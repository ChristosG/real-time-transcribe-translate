#### uvicorn perfecto:app --host 0.0.0.0 --port 7000 --loop uvloop --workers 4

import time
import asyncio
import threading
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

# Import your existing transcription functions
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import sounddevice as sd
import numpy as np
import webrtcvad
import collections

# Additional imports for translation
from langchain.llms.base import LLM
from pydantic import Extra
from typing import Any, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
from transformers import AutoTokenizer
import json

app = FastAPI()

# Create a global asyncio queue for inter-thread communication
message_queue = asyncio.Queue()

class TritonLLM(LLM):
    llm_url = "http://localhost:8000/v2/models/ensemble/generate"  # Adjust the URL if needed

    class Config:
        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        return "Triton LLM"

    def _call(
        self,
        prompt: str,
        temperature: float,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        payload = {
            "text_input": prompt,
            "parameters": {
                "max_tokens": 100,
                "temperature": temperature,
                "top_k": 50,
            }
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(self.llm_url, json=payload, headers=headers)
        response.raise_for_status()

        return response.json()['text_output']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"llmUrl": self.llm_url}

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected: {websocket.client}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"Client disconnected: {websocket.client}")
    
    async def broadcast(self, message: str):
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Error sending message to {connection.client}: {e}")
                self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection open
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.on_event("startup")
async def startup_event():
    # Start the background task to broadcast messages
    asyncio.create_task(broadcast_messages())
    # Get the current event loop
    loop = asyncio.get_event_loop()
    # Start the transcription function in a separate thread
    transcription_thread = threading.Thread(
        target=record_and_transcribe_with_vad,
        args=(whisper_prompt, language_code, model, triton_server, 20, 800, 16000, loop, message_queue),
        daemon=True
    )

    transcription_thread.start()

async def broadcast_messages():
    while True:
        message = await message_queue.get()
        try:
            await manager.broadcast(message)  # message is already JSON serialized
        except Exception as e:
            print(f"Error broadcasting message: {e}")

def transcribe_audio(
    audio_data: np.ndarray,
    whisper_prompt: str,
    language: str,
    model_name: str = "whisper",
    server_url: str = "localhost:8001"
) -> str:
    """
    Sends audio data to the Triton server via gRPC for transcription.

    Args:
        audio_data (np.ndarray): 1D NumPy array of normalized float32 audio samples.
        whisper_prompt (str): Custom prompt for Whisper.
        language (str): Language code (e.g., 'de' for German).
        model_name (str): Name of the Whisper model deployed on Triton.
        server_url (str): Triton server address.

    Returns:
        str: Transcribed text.
    """
    try:
        # Initialize Triton client
        triton_client = grpcclient.InferenceServerClient(url=server_url, verbose=False)

        # Prepare audio data without padding
        samples = audio_data.astype(np.float32)
        samples = np.expand_dims(samples, axis=0)  # Add batch dimension

        # Prepare TEXT_PREFIX input
        text_prefix = f"<|startoftranscript|><|{language}|><|transcribe|><|notimestamps|>"

        # Create Triton inputs
        inputs = []
        
        # WAV input
        input_wav = grpcclient.InferInput("WAV", samples.shape, np_to_triton_dtype(samples.dtype))
        input_wav.set_data_from_numpy(samples)
        inputs.append(input_wav)
        
        # TEXT_PREFIX input
        input_text = grpcclient.InferInput("TEXT_PREFIX", [1, 1], "BYTES")
        input_text.set_data_from_numpy(np.array([[text_prefix.encode()]], dtype=object))
        inputs.append(input_text)
        
        # Specify the output
        outputs = [grpcclient.InferRequestedOutput("TRANSCRIPTS")]

        # Perform inference
        response = triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )

        # Extract transcription
        transcription = response.as_numpy("TRANSCRIPTS")[0]
        if isinstance(transcription, np.ndarray):
            transcription = b" ".join(transcription).decode("utf-8")
        else:
            transcription = transcription.decode("utf-8")
        
        return transcription

    except Exception as e:
        print(f"Transcription Error: {e}")
        return ""


def record_and_transcribe_with_vad(
    whisper_prompt: str,
    language: str,
    model_name: str,
    server_url: str,
    frame_duration: int,
    padding_duration: int,
    sample_rate: int,
    loop,
    message_queue
):
    # Initialize VAD and other variables
    vad = webrtcvad.Vad(0)  # Less aggressive mode
    channels = 1
    dtype = 'int16'
    frame_size = int(sample_rate * frame_duration / 1000) * 2  # 16-bit audio
    padding_frames = int(padding_duration / frame_duration)
    ring_buffer = collections.deque(maxlen=padding_frames)
    triggered = False
    buffer = b''
    buffer_duration = 0.0  # Initialize buffer duration
    overlap_duration = 0.2  # 500 milliseconds
    overlap_frames = int(overlap_duration * sample_rate * 2)

    def transcribe(buffer_bytes):
        # Convert bytes to numpy array
        audio_int16 = np.frombuffer(buffer_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / np.iinfo(np.int16).max  # Normalize

        # Log buffer length
        # buffer_length = len(buffer_bytes) / (sample_rate * 2)
        # print(f"Transcribing buffer of length {buffer_length:.2f} seconds")

        # Transcribe using Triton
        transcription = transcribe_audio(
            audio_data=audio_float32,
            whisper_prompt=whisper_prompt,
            language=language,
            model_name=model_name,
            server_url=server_url
        )

        if transcription and 'Vielen Dank.' not in transcription:
            print("\n=== New Transcription ===")
            print("Transcription (German):", transcription)

            # Now perform translation
            translation = translate_text(transcription)

            # Put the transcription and translation into the asyncio queue
            message = {
                'transcription': transcription,
                'translation': translation
            }
            # Serialize the message as JSON
            asyncio.run_coroutine_threadsafe(message_queue.put(json.dumps(message)), loop)
        else:
            print(f"\n=== Transcription Failed === vd?{transcription}")

        # Reset buffer
        nonlocal buffer, buffer_duration
        #buffer = b''
        #buffer_duration = 0.0
        buffer = buffer_bytes[-overlap_frames:]
        buffer_duration = len(buffer) / (sample_rate * 2)

    def translate_text(transcription):
        # Build the prompt
        messages = [
            {"role": "system", "content": """You are an AI virtual Assistant, and your task is to translate given sentences written in German to English. Just answer the English translation without any other explanation or comments."""},
            {"role": "user", "content": transcription},
        ]

        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        translation = llm(prompt=prompt, temperature=0.0)
        return translation.strip()


    def audio_callback(indata, frames, time_info, status):
        nonlocal triggered, buffer, buffer_duration
        if status:
            print(f"Recording Status: {status}")
        frame = indata.flatten().tobytes()
        is_speech = vad.is_speech(frame, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.7 * ring_buffer.maxlen:
                triggered = True
                buffer = b''.join([f for f, s in ring_buffer])
                buffer_duration = len(buffer) / (sample_rate * 2)
                ring_buffer.clear()
        else:
            buffer += frame
            buffer_duration += frame_duration / 1000.0  # Convert ms to seconds

            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.7 * ring_buffer.maxlen:
                triggered = False
                transcribe(buffer)
                ring_buffer.clear()
            else:
                # Check if buffer duration exceeds maximum
                max_buffer_duration = 5.0  # seconds (adjusted)
                if buffer_duration >= max_buffer_duration:
                    transcribe(buffer)
                    # Continue being in triggered state

    # Start the input stream
    with sd.InputStream(
        callback=audio_callback,
        channels=channels,
        samplerate=sample_rate,
        dtype=dtype,
        blocksize=int(sample_rate * frame_duration / 1000)
    ):
        print("Recording with VAD... Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping recording...")
            if buffer:
                transcribe(buffer)

# def record_and_transcribe_with_vad(
#     whisper_prompt: str,
#     language: str,
#     model_name: str,
#     server_url: str,
#     frame_duration: int,
#     padding_duration: int,
#     sample_rate: int,
#     loop,
#     message_queue
# ):
#     # Initialize VAD and other variables
#     vad = webrtcvad.Vad(1)  # Aggressiveness mode (0-3)
#     channels = 1
#     dtype = 'int16'
#     frame_size = int(sample_rate * frame_duration / 1000) * 2  # 16-bit audio
#     padding_frames = int(padding_duration / frame_duration)
#     ring_buffer = collections.deque(maxlen=padding_frames)
#     triggered = False
#     buffer = b''
#     buffer_duration = 0.0  # Initialize buffer duration

#     def transcribe(buffer_bytes):
#         # Convert bytes to numpy array
#         audio_int16 = np.frombuffer(buffer_bytes, dtype=np.int16)
#         audio_float32 = audio_int16.astype(np.float32) / np.iinfo(np.int16).max  # Normalize

#         # Transcribe using Triton
#         transcription = transcribe_audio(
#             audio_data=audio_float32,
#             whisper_prompt=whisper_prompt,
#             language=language,
#             model_name=model_name,
#             server_url=server_url
#         )

#         if transcription and 'Vielen Dank.' not in transcription:
#             print("\n=== New Transcription ===")
#             print("Transcription (German):", transcription)

            
#             # Now perform translation
#             translation = translate_text(transcription)

#             # Put the transcription and translation into the asyncio queue
#             message = {
#                 'transcription': transcription,
#                 'translation': translation
#             }
#             # Serialize the message as JSON
#             asyncio.run_coroutine_threadsafe(message_queue.put(json.dumps(message)), loop)
#         else:
#             print("\n=== Transcription Failed === vd?{transcription}")

#     def translate_text(transcription):
#         # Build the prompt
#         messages = [
#             {"role": "system", "content": """You are an AI virtual Assistant, and your task is to translate given sentences written in German to English. Just answer the English translation without any other explanation or comments."""},
#             {"role": "user", "content": transcription},
#         ]

#         prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

#         translation = llm(prompt=prompt, temperature=0.0)
#         return translation.strip()

#     def audio_callback(indata, frames, time_info, status):
#         nonlocal triggered, buffer, buffer_duration
#         if status:
#             print(f"Recording Status: {status}")
#         frame = indata.flatten().tobytes()
#         is_speech = vad.is_speech(frame, sample_rate)

#         if not triggered:
#             ring_buffer.append((frame, is_speech))
#             num_voiced = len([f for f, speech in ring_buffer if speech])
#             if num_voiced > 0.9 * ring_buffer.maxlen:
#                 triggered = True
#                 buffer = b''.join([f for f, s in ring_buffer])
#                 buffer_duration = len(buffer) / (sample_rate * 2)
#                 ring_buffer.clear()
#         else:
#             buffer += frame
#             buffer_duration += frame_duration / 1000.0  # Convert ms to seconds

#             ring_buffer.append((frame, is_speech))
#             num_unvoiced = len([f for f, speech in ring_buffer if not speech])
#             if num_unvoiced > 0.9 * ring_buffer.maxlen:
#                 triggered = False
#                 transcribe(buffer)
#                 buffer = b''.join([f for f, s in ring_buffer if s])
#                 buffer_duration = len(buffer) / (sample_rate * 2)
#                 ring_buffer.clear()
#             else:
#                 # Check if buffer duration exceeds maximum
#                 max_buffer_duration = 3.0  # seconds (adjust as needed)
#                 if buffer_duration >= max_buffer_duration:
#                     transcribe(buffer)
#                     buffer = b''
#                     buffer_duration = 0.0
#                     # Continue being in triggered state

#     # Start the input stream
#     with sd.InputStream(
#         callback=audio_callback,
#         channels=channels,
#         samplerate=sample_rate,
#         dtype=dtype,
#         blocksize=int(sample_rate * frame_duration / 1000)
#     ):
#         print("Recording with VAD... Press Ctrl+C to stop.")
#         try:
#             while True:
#                 time.sleep(0.1)
#         except KeyboardInterrupt:
#             print("\nStopping recording...")
#             if buffer:
#                 transcribe(buffer)

# Configuration variables
whisper_prompt = "<|startoftranscript|><|de|><|transcribe|><|notimestamps|>"
language_code = "de"
model = "whisper"
triton_server = "localhost:8001"

# Initialize the LLM and tokenizer
llm = TritonLLM()
tokenizer = AutoTokenizer.from_pretrained("/home/chris/engines/Meta-Llama-3.1-8B-Instruct")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)
