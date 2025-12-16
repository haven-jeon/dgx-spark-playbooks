FROM nvcr.io/nvidia/pytorch:25.09-py3

# CUDA 환경 변수 설정
# 경고 해결: 기존 변수에 덧붙이는 방식(:$VAR) 대신 직접 경로를 할당하거나, 빈 값일 경우를 대비해야 합니다.
# 여기서는 초기화하는 방식으로 수정하여 경고를 없앱니다.
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV CUDA_PATH=$CUDA_HOME
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH=$CUDA_HOME/include
ENV CPLUS_INCLUDE_PATH=$CUDA_HOME/include
ENV TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

# vLLM 관련 환경 변수
ENV VLLM_TARGET_DEVICE="cuda"
ENV MAX_JOBS=4

# 1. Install triton from source
# 기존에 지정하신 커밋 해시 유지
RUN git clone https://github.com/triton-lang/triton.git && \
	cd triton && \
	git checkout c5d671f91d90f40900027382f98b17a3e04045f6 && \
	pip install -r python/requirements.txt && \
	pip install . --no-build-isolation && \
	cd ..


# 2. Install xformers from source (수정됨)
# 문제 해결: PyTorch 2.9 호환성을 위해 v0.0.33 태그를 체크아웃합니다.
# --depth=1을 제거하고 전체를 clone한 뒤 특정 태그로 checkout 합니다.
RUN git clone https://github.com/facebookresearch/xformers.git --recursive && \
	cd xformers && \
	git checkout v0.0.33 && \
	export TORCH_CUDA_ARCH_LIST="12.1" && \
	pip install . --no-deps --no-build-isolation && \
	cd ..

  
# 3. Install vLLM
RUN pip install "cmake>=3.21" ninja packaging setuptools-scm>=8 wheel jinja2

# vLLM 설치
# xformers 버전을 낮췄으므로 vLLM 설치 시 호환성 문제가 발생할 수 있습니다.
# 소스 빌드를 유지하되, 만약 실패한다면 vLLM 버전도 조정이 필요할 수 있습니다.
RUN git clone https://github.com/vllm-project/vllm.git && \
	cd vllm && \
	export TORCH_CUDA_ARCH_LIST="12.1" && \
	# --no-deps를 사용하여 torch 재설치를 막습니다.
	pip install . --no-build-isolation --no-deps && \
	cd ..


# 5. Install Runtime Dependencies (수정됨: 순서를 뒤로 뺌)
# 여기가 변경되어도 위 4번 단계(vLLM 빌드)는 다시 실행되지 않습니다.
# gguf, outlines 등 실행에 필요한 라이브러리를 여기서 설치합니다.
RUN pip install \
	"regex" \
	"cachetools" \
	"psutil" \
	"sentencepiece" \
	"numpy" \
	"requests>=2.26.0" \
	"tqdm" \
	"blake3" \
	"py-cpuinfo" \
	"transformers>=4.56.0,<5" \
	"tokenizers>=0.21.1" \
	"protobuf" \
	"fastapi[standard]>=0.115.0" \
	"aiohttp" \
	"openai>=1.99.1" \
	"pydantic>=2.12.0" \
	"prometheus_client>=0.18.0" \
	"pillow" \
	"prometheus-fastapi-instrumentator>=7.0.0" \
	"tiktoken>=0.6.0" \
	"lm-format-enforcer==0.11.3" \
	"llguidance>=1.3.0,<1.4.0" \
	"outlines_core==0.2.11" \
	"diskcache==5.6.3" \
	"lark==1.2.2" \
	"xgrammar==0.1.27" \
	"typing_extensions>=4.10" \
	"filelock>=3.16.1" \
	"partial-json-parser" \
	"pyzmq>=25.0.0" \
	"msgspec" \
	"gguf>=0.17.0" \
	"mistral_common[image]>=1.8.5" \
	"opencv-python-headless>=4.11.0" \
	"pyyaml" \
	"six>=1.16.0" \
	"setuptools>=77.0.3,<81.0.0" \
	"einops" \
	"compressed-tensors==0.12.2" \
	"depyf==0.20.0" \
	"cloudpickle" \
	"watchfiles" \
	"python-json-logger" \
	"scipy" \
	"ninja" \
	"pybase64" \
	"cbor2" \
	"ijson" \
	"setproctitle" \
	"openai-harmony>=0.0.3" \
	"anthropic==0.71.0" \
	"model-hosting-container-standards>=0.1.9,<1.0.0" \
	"nvidia-ml-py" \
	"ray" \
	"pandas" \
	"pyarrow" \
	"wandb"


# 4. Install unsloth and other dependencies
#RUN pip install --no-deps bitsandbytes==0.48.0 transformers==4.56.2 trl==0.22.2
RUN pip install unsloth unsloth_zoo qwen-vl-utils

# Launch the shell
CMD ["/bin/bash"]
