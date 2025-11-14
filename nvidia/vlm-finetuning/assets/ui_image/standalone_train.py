import os
import yaml
import torch
import argparse
from PIL import Image, ImageFile
from unsloth import FastLanguageModel, FastVisionModel
from datasets import load_dataset, Image as HFImage
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainingArguments
import io
import re

# 손상된(truncated) 이미지 허용
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 상수 정의
REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"
BASE_CONFIG_FILE = "src/image_vlm_config.yaml"
# 고정된 프롬프트를 상수로 정의합니다.
FIXED_PROMPT = "산불 피해 지역을 파악합니다."

# --- 보상 함수 정의 (하나의 통합 함수로 변경) ---

def combined_reward_function(prompts, completions, answer, **kwargs):
    """
    형식과 정답 정확도를 모두 평가하여 단일 보상 점수를 반환합니다.
    가중치를 적용하여 각 요소의 중요도를 조절합니다.
    """
    format_weight = kwargs.get("format_weight", 1.0)
    correctness_weight = kwargs.get("correctness_weight", 1.0)

    thinking_pattern = f'{REASONING_START}(.*?){REASONING_END}'
    answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'
    
    final_scores = []
    
    # completions와 answer의 길이를 맞추기 위해 zip 사용
    # num_generations 만큼 answer가 확장되어 전달된다고 가정
    for completion, ground_truth in zip(completions, answer):
        # 1. 형식 점수 계산
        format_score = 0.0
        if isinstance(completion, str):
            thinking_matches = re.findall(thinking_pattern, completion, re.DOTALL)
            answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
            # 형식 점수를 0 또는 1로 단순화
            if len(thinking_matches) == 1 and len(answer_matches) == 1:
                format_score = 1.0
        
        # 2. 정확도 점수 계산
        correctness_score = 0.0
        if isinstance(completion, str):
            extracted_solutions = re.findall(answer_pattern, completion, re.DOTALL)
            # 정답을 맞혔을 때 훨씬 큰 보상을 주도록 점수 상향
            if len(extracted_solutions) == 1 and ground_truth.lower() == extracted_solutions[0].strip().lower():
                correctness_score = 10.0

        # 3. 가중치를 적용하여 최종 점수 합산
        total_score = (format_score * format_weight) + (correctness_score * correctness_weight)
        final_scores.append(total_score)

    # 디버깅 로그
    if completions:
        print("--- combined_reward_function DEBUG ---")
        print(f"Completions (sample): {completions[0]}")
        print(f"Ground Truth (sample): {answer[0]}")
        print(f"Final Scores (sample): {final_scores[:4]}")
        print(f"Scores Mean: {sum(final_scores)/len(final_scores) if final_scores else 0}")
        print(f"Scores Std: {torch.tensor(final_scores).std().item() if final_scores else 0}")
        print("------------------------------------")

    return final_scores


# --- 학습 메인 함수 ---
def train(config):
    """설정 파일을 기반으로 모델 학습을 수행합니다."""
    model_config = config['model']
    data_config = config['data']
    hyperparams_config = config['hyperparameters']

    # 1. 모델 및 토크나이저 로드
    print("1. 모델 및 토크나이저를 로드합니다...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_config['model_id'],
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    
    # Unsloth의 PEFT (LoRA) 설정 적용
    if model_config['use_lora']:
        print("   - LoRA 설정을 적용합니다.")
        model = FastVisionModel.get_peft_model(
            model,
            r=model_config['lora_config']['rank'],
            lora_alpha=model_config['lora_config']['alpha'],
            lora_dropout=model_config['lora_config']['dropout'],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            use_gradient_checkpointing=True,
            finetune_vision=model_config['finetune_vision_layers'],
            vision_target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "mlp.fc1", "mlp.fc2"
            ] if model_config['finetune_vision_layers'] else None,
        )

    # 2. 데이터셋 로드 및 전처리
    print("2. 데이터셋을 로드하고 전처리합니다...")
    raw_dataset = load_dataset(data_config['dataset_id'])

    # 2-1) 이미지 디코딩 비활성화(경로/바이트만 보유)
    ds = raw_dataset["train"].cast_column("image", HFImage(decode=False))

    # 2-2) 손상 이미지 필터링 (우리가 직접 열어 검사)
    def is_image_valid(sample):
        img = sample["image"]
        try:
            if isinstance(img, dict):
                if img.get("path"):
                    with open(img["path"], "rb") as f:
                        im = Image.open(f)
                        im.load()  # truncated 허용 상태에서 강제 로드
                elif img.get("bytes") is not None:
                    im = Image.open(io.BytesIO(img["bytes"]))
                    im.load()
                else:
                    return False
            else:
                # 혹시 PIL 객체로 들어온 경우
                img.load()
            return True
        except Exception:
            return False

    print("   - 이미지 유효성 검사를 시작합니다...")
    original_size = len(ds)
    filtered_dataset = ds.filter(is_image_valid, batched=False, num_proc=max(1, (os.cpu_count() or 2)//2))
    filtered_size = len(filtered_dataset)
    print(f"   - 유효성 검사 완료. 제외: {original_size - filtered_size} / 남김: {filtered_size}")

    # 2-3) 통과한 샘플만 다시 디코딩 활성화
    filtered_dataset = filtered_dataset.cast_column("image", HFImage(decode=True))

    # 클래스 이름(폴더명)
    class_names = filtered_dataset.features["label"].names

    def create_conversation(sample):
        label_text = class_names[sample["label"]]
        is_wildfire = "wildfire" in label_text.lower()
        answer_text = "Yes" if is_wildfire else "No"
        
        # 모델에게 역할을 부여하고, 출력 형식을 명확하게 지시하는 시스템 프롬프트
        system_prompt = f"""귀하는 전문 위성 이미지 분석가입니다. 제공된 이미지에 산불 피해가 있는지 확인하는 것이 귀하의 임무입니다.

다음 형식으로만 응답해 주십시오.
1. 먼저 {REASONING_START} 및 {REASONING_END} 태그 안에 단계별 추론 과정을 작성해 주십시오.
2. 그런 다음 {SOLUTION_START} 및 {SOLUTION_END} 태그 안에 최종 답변('예' 또는 '아니요')을 작성해 주십시오.

예:
{REASONING_START}
1. 이미지는 삼림 지대를 보여줍니다.
2. 검게 그을린 땅과 새까맣게 탄 나무들이 넓게 보입니다.
3. 눈에 띄는 연기나 아지랑이가 있는데, 이는 최근 화재가 발생했음을 나타냅니다.
{REASONING_END}
{SOLUTION_START}Yes{SOLUTION_END}
"""
        
        # 모델 입력 형식인 messages 리스트를 구성합니다.
        # GRPO 학습 시에는 assistant 응답을 제외하고 user의 질문까지만 제공합니다.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": FIXED_PROMPT}]},
        ]

        # messages를 토크나이저를 통해 실제 모델 입력 텍스트(prompt)로 변환합니다.
        # add_generation_prompt=True가 assistant의 응답을 시작하도록 유도합니다.
        prompt_str = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
        )
        
        # GRPOTrainer가 사용할 최종 데이터 컬럼들을 반환합니다.
        return {
            "prompt": prompt_str,
            "image": sample["image"],
            "answer": answer_text, 
        }

    # 'label' 컬럼만 제거하고 'prompt', 'image', 'answer'는 남깁니다.
    dataset = filtered_dataset.map(create_conversation, remove_columns=["label"])
    
    # 3. GRPO 학습 설정
    print("3. GRPO 학습을 설정합니다...")

    # 하이퍼파라미터 타입을 확실히 숫자로 강제 변환
    steps = int(hyperparams_config['steps'])
    batch_size = int(hyperparams_config['batch_size'])
    learning_rate = float(hyperparams_config['learning_rate'])
    num_generations = int(hyperparams_config['num_generations'])
    format_reward_w = float(hyperparams_config['format_reward'])
    correctness_reward_w = float(hyperparams_config['correctness_reward'])
    max_seq_length = int(model_config['max_seq_length'])

    grpo_config = GRPOConfig(
        output_dir=hyperparams_config['output_dir'],
        num_train_epochs=1,
        max_steps=steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=float(0.1),
        logging_steps=int(1),
        save_steps=int(5),
        save_total_limit=int(2),
        optim=str(hyperparams_config['optimizer']),
        bf16=True,
        remove_unused_columns=False,
        report_to="none",

        logging_strategy="steps",
        save_strategy="steps",

        # GRPO 설정
        beta=float(0.1),
        num_generations=num_generations,
        temperature=0.9, # 다양성 확보를 위해 유지
    )

    # 4. GRPOTrainer 초기화
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        # === 수정: 단일 통합 보상 함수 사용 ===
        reward_funcs=[combined_reward_function],
        # === 수정: 가중치를 reward_kwargs로 전달 ===
        reward_kwargs={
            "format_weight": format_reward_w,
            "correctness_weight": correctness_reward_w,
        },
        max_length=max_seq_length,
        max_prompt_length=max_seq_length // 2,
    )

    # 5. 학습 시작
    print("\n" + "="*20 + " 학습 시작 " + "="*20)
    trainer.train()
    print("="*20 + " 학습 완료 " + "="*20 + "\n")

    # 6. 모델 저장
    output_dir = hyperparams_config['output_dir']
    print(f"6. 학습된 모델을 '{output_dir}'에 저장합니다...")
    if model_config['use_lora']:
        print("   - LoRA 가중치를 병합하고 16비트로 저장합니다.")
        trainer.model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    else:
        trainer.model.save_pretrained(output_dir)
    
    print("모든 과정이 완료되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM 모델 파인튜닝을 위한 독립 실행 스크립트")
    parser.add_argument(
        '--config_path', 
        type=str, 
        default=BASE_CONFIG_FILE, 
        help=f"학습 설정이 포함된 YAML 파일 경로. 기본값: {BASE_CONFIG_FILE}"
    )
    args = parser.parse_args()

    try:
        with open(args.config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        if 'train' not in full_config:
            raise KeyError("'train' 섹션을 설정 파일에서 찾을 수 없습니다.")
            
        training_config = full_config['train']
        train(training_config)

    except FileNotFoundError:
        print(f"에러: 설정 파일 '{args.config_path}'을(를) 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")








