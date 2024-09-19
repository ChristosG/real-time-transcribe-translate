
# Real-Time Transcriber and Translator Guide

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [Step 1: Prepare Docker Environment](#step-1-prepare-docker-environment)
  - [Step 2: Build and Deploy Models](#step-2-build-and-deploy-models)
    - [2.a LLaMA 3.1 Engine](#2a-llama-31-engine)
    - [2.b Whisper Engine](#2b-whisper-engine)
- [Proxy Setup](#proxy-setup)
- [React Web App](#react-web-app)
- [Running the Project](#running-the-project)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

## Description

Welcome to the **Real-Time Transcriber and Translator Guide**! This guide will walk you through building, deploying, and using a real-time transcription and translation system. In this example, we focus on translating from German to English, but the setup can be easily adjusted for other languages supported by `whisper-large-v3`.

## Features

- **Real-Time Transcription:** Converts spoken German into text.
- **Real-Time Translation:** Translates transcribed text from German to English.
- **Dockerized Environment:** Ensures consistency and ease of deployment.
- **Scalable Proxy:** Handles multiple users via WebSockets.
- **React Web App:** User-friendly interface for interaction.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Operating System:** Ubuntu
- **Docker:** Installed and running
- **NVIDIA Drivers:** Correct version installed
- **NVIDIA Container Toolkit:** Installed
- **FFmpeg:** For audio processing
- **Python 3.8+**
- **Node.js and npm:** For the React web app

### Install NVIDIA Drivers and Container Toolkit

```bash
# Update package lists
sudo apt update

# Install NVIDIA drivers (replace with the required version)
sudo apt install -y nvidia-driver-525

# Add the NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install NVIDIA Container Toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Restart Docker to apply changes
sudo systemctl restart docker


### Install FFmpeg

```bash
sudo apt update
sudo apt install -y ffmpeg
```

### Install Python and Node.js

```bash
# Install Python 3 and pip
sudo apt install -y python3 python3-pip

# Install Node.js and npm
sudo apt install -y nodejs npm
```

## Setup

### Step 1: Prepare Docker Environment

1. **Verify NVIDIA Drivers in Docker**

   Ensure Docker can access your NVIDIA drivers by running:

   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

   You should see the output from `nvidia-smi`. If not, revisit the NVIDIA driver and container toolkit installation.

### Step 2: Build and Deploy Models

<details>
<summary>Shout out to [Sherpa Triton Whisper Guide](https://github.com/k2-fsa/sherpa/tree/master/triton/whisper) for their slick instructions on building the Whisper TRT engine.</summary>
  
</details>

#### 2.a LLaMA 3.1 Engine

1. **Create Engines Directory**

   ```bash
   mkdir -p engines
   cd engines
   ```

2. **Download Models**

   ```bash
   # Create assets directory
   mkdir -p assets

   # Download Whisper Large V3 model
   wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt

   # Download LLaMA 3.1 Instruct model
   wget --directory-prefix=assets https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/resolve/main/model.bin
   ```

3. **Build Docker Image**

   ```bash
   cd build_engines_and_init_triton
   docker build -t llama_whisper .
   ```

4. **Start Builder Script**

   ```bash
   ./start_builder.sh
   ```

5. **Convert LLaMA 3.1 Checkpoint**

   Ensure all dependencies are installed. If you encounter errors related to Python libraries, install them accordingly using `pip`.

   ```bash
   docker exec -it triton-trtllm-container bash
   cd TensorRT-LLM

   python3 examples/llama/convert_checkpoint.py \
     --model_dir /engines/Meta-Llama-3.1-8B-Instruct \
     --output_dir /engines/llama31_checkpoint \
     --dtype float16 \
     --use_weight_only \
     --weight_only_precision int4
   ```

   > **Note:** The quantized model is set to `int4` for translation and short sentence semantics with `max_seq_len` set to 2000 tokens. Adjust these settings as needed by modifying `tensorrt_llm/config.pbtxt` in the `tensorrtllm_backend` directory.

6. **Build TRTLLM Engine**

   ```bash
   trtllm-build \
     --checkpoint_dir /engines/llama31_checkpoint \
     --output_dir /engines/llama31_engine \
     --gemm_plugin float16 \
     --max_batch_size 1 \
     --max_input_len 1000 \
     --max_seq_len 2000 \
     --use_paged_context_fmha enable
   ```

7. **Test the Build**

   ```bash
   tritonserver --model-repository=tensorrtllm_backend/ \
     --model-control-mode=explicit \
     --load-model=preprocessing \
     --load-model=postprocessing \
     --load-model=tensorrt_llm \
     --load-model=tensorrt_llm_bls \
     --load-model=ensemble \
     --log-verbose=2 \
     --log-info=1 \
     --log-warning=1 \
     --log-error=1 \
     --http-port 8000 \
     --grpc-port 8001 \
     --metrics-port 8002
   ```

   If you see the following logs, the server is ready for inference:

   ```
   I0919 10:12:29.013889 126 grpc_server.cc:2463] "Started GRPCInferenceService at 0.0.0.0:8001"
   I0919 10:12:29.014019 126 http_server.cc:4692] "Started HTTPService at 0.0.0.0:8000"
   I0919 10:12:29.055157 126 http_server.cc:362] "Started Metrics Service at 0.0.0.0:8002"
   ```

8. **Test Inference**

   Open another terminal and run:

   ```bash
   curl -X POST "http://localhost:8000/v2/models/ensemble/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "text_input": "Ich heiße Chris",
       "parameters": {
         "max_tokens": 100,
         "bad_words": [""],
         "stop_words": [""],
         "temperature": 0.0,
         "top_k": 50
       }
     }'
   ```

#### 2.b Whisper Engine

1. **Stop Triton Server with LLaMA**

   Press `Ctrl+C` in the terminal running the Triton server to stop it.

2. **Build Whisper Engine**

   ```bash
   docker exec -it triton-trtllm-container bash
   cd TensorRT-LLM

   # Create Whisper Checkpoint
   python3 examples/whisper/convert_checkpoint.py \
     --model_dir /engines/assets \
     --output_dir /engines/whisper_large_checkpoint \
     --model_name large-v3 \
     --use_weight_only \
     --weight_only_precision int8

   # Build TRT Encoder Engine
   trtllm-build \
     --checkpoint_dir /engines/whisper_large_checkpoint/encoder \
     --output_dir /engines/whisper_large/encoder \
     --paged_kv_cache disable \
     --moe_plugin disable \
     --enable_xqa disable \
     --max_batch_size 8 \
     --gemm_plugin disable \
     --bert_attention_plugin float16 \
     --remove_input_padding disable \
     --max_input_len 1500

   # Build TRT Decoder Engine
   trtllm-build \
     --checkpoint_dir /engines/whisper_large_checkpoint/decoder \
     --output_dir /engines/whisper_large/decoder \
     --paged_kv_cache disable \
     --moe_plugin disable \
     --enable_xqa disable \
     --max_beam_width 4 \
     --max_batch_size 8 \
     --max_seq_len 114 \
     --max_input_len 14 \
     --max_encoder_input_len 1500 \
     --gemm_plugin float16 \
     --bert_attention_plugin float16 \
     --gpt_attention_plugin float16 \
     --remove_input_padding disable
   ```

3. **Download Required Files**

   Ensure the following files are present in the `tensorrtllm_backend/whisper/1/` directory:

   ```bash
   wget --directory-prefix=tensorrtllm_backend/whisper/1/ https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
   wget --directory-prefix=tensorrtllm_backend/whisper/1/ https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
   ```

4. **Deploy Whisper on Triton for Testing**

   ```bash
   tritonserver --model-repository=tensorrtllm_backend/ \
     --model-control-mode=explicit \
     --load-model=whisper \
     --log-verbose=2 \
     --log-info=1 \
     --log-warning=1 \
     --log-error=1 \
     --http-port 8000 \
     --grpc-port 8001 \
     --metrics-port 8002
   ```

   If it runs without errors, press `Ctrl+C` to stop the server and exit the Docker container.

## Proxy Setup

This section sets up a proxy to handle communication between the frontend web page and Triton server, supporting WebSockets for multiple users. **Note:** This project records audio from your PC's microphone only, but it can be adjusted to your needs.

1. **(Optional) Set Up Python Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r venv_mods/requirements.txt
   ```

2. **Test Dependencies**

   Run the following command to ensure all dependencies are installed correctly:

   ```bash
   uvicorn venv_mods/perfecto:app --host 0.0.0.0 --port 7000 --loop uvloop --workers 4
   ```

   If it runs without errors, press `Ctrl+C` to stop the server.

## React Web App

**Note:** This section is optional. You can adjust `App.tsx` as needed and open it directly in a browser. This approach may require modifying API endpoints.

### Prerequisites

- **Node.js and npm:** Ensure they are installed.

### Steps

1. **Install Node Modules**

   ```bash
   npm install
   ```

2. **Start the React App**

   ```bash
   npm start
   ```

   Alternatively, you can use Expo:

   ```bash
   npx expo start
   ```

   Press `W` in the terminal to open the app in your browser.

## Running the Project

Assuming all steps have been completed successfully:

1. **Close All Services:** Ensure Docker containers and servers are stopped.

2. **Run Orchestrator Script**

   ```bash
   ./orchestrator.sh
   ```

   > **Note:** Adjust the paths inside `orchestrator.sh` as needed.

3. **Use the Application**

   - Speak into your microphone.
   - View the German transcription and English translation on the web app.

   **Tip:** For better audio capture during Zoom calls, set up a Virtual Sink environment. This captures audio directly from Zoom instead of playing it through speakers, reducing noise and delay for improved transcription quality.

## Usage

- **Real-Time Interaction:** Speak into your microphone, and the system will transcribe and translate your speech in real-time.
- **Web Interface:** Use the React web app to view transcriptions and translations.
- **Customization:** Adjust model parameters and configurations to suit different languages and use cases.

## Dependencies

Ensure the following dependencies are installed:

- **Ubuntu Packages:**
  - `docker`
  - `nvidia-driver`
  - `nvidia-container-toolkit`
  - `ffmpeg`
- **Python Libraries:**
  - Listed in `venv_mods/requirements.txt`
- **Node.js and npm**
- **Additional Tools:**
  - `wget`
  - `curl`
  - `git`

## Troubleshooting

- **Docker NVIDIA Access Issues:**
  - Verify NVIDIA drivers and container toolkit are correctly installed.
  - Run `nvidia-smi` inside a Docker container to check access.

- **Python Dependency Errors:**
  - Ensure all required Python libraries are installed.
  - Use `pip install -r venv_mods/requirements.txt` to install dependencies.

- **Triton Server Errors:**
  - Check logs for specific error messages.
  - Ensure model paths and configurations are correct.

- **React App Issues:**
  - Verify Node.js and npm are correctly installed.
  - Check API endpoint configurations in `App.tsx`.


## Contact

- **Author:** Christos Grigoriadis
- **Email:** cgrigoriadis@outlook.com
```

