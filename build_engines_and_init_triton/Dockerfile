FROM nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3

RUN apt-get update && apt-get -y install python3.10 python3-pip git git-lfs 

RUN apt-get update && apt-get install -y --no-install-recommends rapidjson-dev python-is-python3 ccache 

RUN pip3 uninstall -y tensorrt tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs tensorrt-llm torch 

RUN pip3 install tensorrt_llm==0.13.0.dev2024082000 -U --pre --extra-index-url https://pypi.nvidia.com

RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git

WORKDIR /TensorRT-LLM

RUN pip3 install tiktoken

RUN pip install -U transformers==4.44.2

EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

CMD ["/bin/bash"]
