"""
FastAPI Server for VieNeu-TTS
Supports streaming, multiple users, and configurable parameters
"""

import os
import io
import asyncio
import uuid
import base64
import time
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vieneu_tts import VieNeuTTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/vieneu_tts_performance.log')
    ]
)
logger = logging.getLogger("VieNeuTTS-API")

# Global variables
tts_model: Optional[Any] = None
user_sessions: Dict[str, Dict[str, Any]] = {}
preset_voice_cache: Dict[str, Dict[str, Any]] = {}  # Cache for preset voices
preset_voice_cache: Dict[str, Dict[str, Any]] = {}  # Cache for preset voices

# Configuration
BACKBONE_REPO = os.getenv("BACKBONE_REPO", "pnnbao-ump/VieNeu-TTS")
CODEC_REPO = os.getenv("CODEC_REPO", "neuphonic/neucodec")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
ENABLE_TRITON = os.getenv("ENABLE_TRITON", "true").lower() == "true"
MODEL_TYPE = os.getenv("MODEL_TYPE", "pytorch")  # "pytorch" or "gguf"


def is_gguf_model(backbone_repo: str) -> bool:
    """Check if the model is a GGUF quantized model"""
    return "gguf" in backbone_repo.lower() or backbone_repo.lower().endswith(".gguf")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup TTS model"""
    global tts_model

    print("🚀 Initializing VieNeu-TTS model...")
    print(f"   Model Type: {MODEL_TYPE}")
    print(f"   Backbone: {BACKBONE_REPO}")
    print(f"   Codec: {CODEC_REPO}")
    print(f"   Device: {DEVICE}")

    try:
        is_quantized = is_gguf_model(BACKBONE_REPO) or MODEL_TYPE.lower() == "gguf"

        if is_quantized:
            # GGUF quantized model (llama.cpp backend)
            print("📦 Loading GGUF quantized model with llama.cpp backend...")

            # For GGUF models, backbone uses 'gpu' or 'cpu' for llama.cpp
            backbone_device = "gpu" if (DEVICE == "cuda" or DEVICE.startswith("cuda")) else "cpu"

            # For GGUF models, use ONNX codec on CPU for better performance
            # ONNX codec is faster than PyTorch codec and only supports CPU
            codec_repo = CODEC_REPO
            codec_device = "cpu"

            # Auto-switch to ONNX codec if using default codec
            if CODEC_REPO == "neuphonic/neucodec":
                codec_repo = "neuphonic/neucodec-onnx-decoder"
                print("   Using ONNX codec for better GGUF performance")

            tts_model = VieNeuTTS(
                backbone_repo=BACKBONE_REPO,
                backbone_device=backbone_device,
                codec_repo=codec_repo,
                codec_device=codec_device
            )
            print(f"✅ GGUF model loaded successfully on {backbone_device}!")

        elif DEVICE == "cuda" or DEVICE.startswith("cuda:"):
            # Try to use FastVieNeuTTS (GPU-optimized) for PyTorch models
            try:
                from vieneu_tts.vieneu_tts import FastVieNeuTTS
                tts_model = FastVieNeuTTS(
                    backbone_repo=BACKBONE_REPO,
                    backbone_device=DEVICE,
                    codec_repo=CODEC_REPO,
                    codec_device=DEVICE,
                    enable_triton=ENABLE_TRITON,
                    max_batch_size=MAX_BATCH_SIZE,
                )
                print("✅ FastVieNeuTTS (GPU-optimized) loaded successfully!")
            except ImportError:
                print("⚠️ LMDeploy not available, falling back to standard VieNeuTTS")
                tts_model = VieNeuTTS(
                    backbone_repo=BACKBONE_REPO,
                    backbone_device=DEVICE,
                    codec_repo=CODEC_REPO,
                    codec_device=DEVICE
                )
                print("✅ Standard VieNeuTTS loaded successfully!")
        else:
            # CPU PyTorch model
            tts_model = VieNeuTTS(
                backbone_repo=BACKBONE_REPO,
                backbone_device=DEVICE,
                codec_repo=CODEC_REPO,
                codec_device=DEVICE
            )
            print("✅ VieNeuTTS loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load TTS model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Warmup the model for better first-request performance
    print("🔥 Starting model warmup...")
    warmup_model()

    yield
    
    # Cleanup
    print("🧹 Cleaning up resources...")
    if hasattr(tts_model, 'cleanup_memory'):
        tts_model.cleanup_memory()
    user_sessions.clear()
    preset_voice_cache.clear()
    preset_voice_cache.clear()


app = FastAPI(
    title="VieNeu-TTS API",
    description="Vietnamese Text-to-Speech API with streaming support",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory and mount static files
os.makedirs("./output_audio", exist_ok=True)
app.mount("/audio", StaticFiles(directory="./output_audio"), name="audio")


# ============================================================================
# Request/Response Models
# ============================================================================

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    user_id: Optional[str] = Field(None, description="User ID for session management")
    voice_name: Optional[str] = Field(None, description="Preset voice name")
    ref_audio_base64: Optional[str] = Field(None, description="Base64 encoded reference audio")
    ref_text: Optional[str] = Field(None, description="Reference text for custom voice")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    streaming: bool = Field(False, description="Enable streaming response")


class RegisterVoiceRequest(BaseModel):
    user_id: str = Field(..., description="Unique user ID")
    voice_name: str = Field(..., description="Voice name for this user")
    ref_text: str = Field(..., description="Reference text")
    ref_audio_base64: str = Field(..., description="Base64 encoded reference audio")

class BatchSynthesizeRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to synthesize")
    voice_name: str = Field(..., description="Voice name")
    user_id: Optional[str] = Field(None, description="User ID")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    active_sessions: int


# ============================================================================
# Helper Functions
# ============================================================================

def get_preset_voice_cached(voice_name: str) -> tuple[np.ndarray, str]:
    """Get cached preset voice data (codes and text) - optimized for Docker deployment"""
    global preset_voice_cache

    # Check cache first
    if voice_name in preset_voice_cache:
        cached_data = preset_voice_cache[voice_name]
        return cached_data["ref_codes"], cached_data["ref_text"]

    # Load and cache if not found
    sample_dir = "./sample"
    audio_path = os.path.join(sample_dir, f"{voice_name}.wav")
    text_path = os.path.join(sample_dir, f"{voice_name}.txt")
    codes_path = os.path.join(sample_dir, f"{voice_name}.pt")

    if not os.path.exists(audio_path) or not os.path.exists(text_path):
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")

    # Load reference text
    with open(text_path, "r", encoding="utf-8") as f:
        ref_text = f.read()

    # Load or encode reference codes
    if os.path.exists(codes_path):
        ref_codes = torch.load(codes_path, map_location="cpu", weights_only=True)
        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.cpu().numpy()
    else:
        ref_codes = encode_reference(audio_path)

    # Cache the data
    preset_voice_cache[voice_name] = {
        "ref_codes": ref_codes,
        "ref_text": ref_text
    }

    return ref_codes, ref_text

def get_preset_voice_path(voice_name: str) -> tuple[str, str, Optional[str]]:
    """Get preset voice paths (legacy function for compatibility)"""
    sample_dir = "./sample"
    audio_path = os.path.join(sample_dir, f"{voice_name}.wav")
    text_path = os.path.join(sample_dir, f"{voice_name}.txt")
    codes_path = os.path.join(sample_dir, f"{voice_name}.pt")
    
    if not os.path.exists(audio_path) or not os.path.exists(text_path):
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
    
    codes_path = codes_path if os.path.exists(codes_path) else None
    return audio_path, text_path, codes_path


def encode_reference(audio_path: str) -> np.ndarray:
    """Encode reference audio to codes"""
    start_time = time.time()

    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    ref_codes = tts_model.encode_reference(audio_path)

    if isinstance(ref_codes, torch.Tensor):
        ref_codes = ref_codes.cpu().numpy()

    encode_time = time.time() - start_time
    print(f"🎵 Reference encoding: {encode_time:.3f}s - {audio_path}")
    
    return ref_codes


def adjust_speed(audio: np.ndarray, speed: float, sample_rate: int = 24000) -> np.ndarray:
    """Adjust audio speed using resampling"""
    if speed == 1.0:
        return audio
    
    try:
        import librosa
        return librosa.effects.time_stretch(audio, rate=speed)
    except ImportError:
        # Fallback: simple resampling (lower quality)
        from scipy import signal
        new_length = int(len(audio) / speed)
        return signal.resample(audio, new_length)


async def generate_audio_stream(text: str, ref_codes: np.ndarray, ref_text: str, speed: float = 1.0):
    """Generate audio stream"""
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    
    # Check if streaming is supported
    if not hasattr(tts_model, 'infer_stream'):
        raise HTTPException(status_code=400, detail="Streaming not supported by current model")
    
    sample_rate = 24000
    
    for audio_chunk in tts_model.infer_stream(text, ref_codes, ref_text):
        if audio_chunk is None or len(audio_chunk) == 0:
            continue
        
        # Apply speed adjustment
        if speed != 1.0:
            audio_chunk = adjust_speed(audio_chunk, speed, sample_rate)
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_chunk, sample_rate, format='WAV')
        buffer.seek(0)
        
        yield buffer.read()
        
        # Reduce delay for better responsiveness
        await asyncio.sleep(0.001)

def warmup_model():
    """Warmup the model with all available voices"""
    if tts_model is None:
        return

    try:
        # Use a short Vietnamese sentence for warmup
        warmup_text = "Xin chào"
        warmed_up_count = 0
        failed_count = 0

        # Warm up ALL preset voices
        sample_dir = "./sample"
        if os.path.exists(sample_dir):
            # Get all voice files and sort them for consistent order
            voice_files = [f for f in os.listdir(sample_dir) if f.endswith(".wav")]
            voice_files.sort()

            for file in voice_files:
                voice_name = file.replace(".wav", "")
                try:
                    ref_codes, ref_text = get_preset_voice_cached(voice_name)
                    print(f"🔥 Warming up model with voice '{voice_name}'...")

                    # Perform warmup inference
                    _ = tts_model.infer(warmup_text, ref_codes, ref_text)
                    warmed_up_count += 1

                except Exception as e:
                    print(f"⚠️ Failed to warm up voice '{voice_name}': {str(e)}")
                    failed_count += 1
                    continue

        if warmed_up_count > 0:
            print(f"✅ Model warmup completed! Successfully warmed up {warmed_up_count} voices.")
            if failed_count > 0:
                print(f"⚠️ {failed_count} voices failed to warm up.")
        else:
            print("⚠️ No preset voices found for warmup")

    except Exception as e:
        print(f"⚠️ Model warmup failed: {e}")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_info = {
        "backbone": BACKBONE_REPO,
        "codec": CODEC_REPO,
        "model_type": MODEL_TYPE,
        "is_gguf": is_gguf_model(BACKBONE_REPO)
    }

    return HealthResponse(
        status="healthy" if tts_model is not None else "unhealthy",
        model_loaded=tts_model is not None,
        device=DEVICE,
        active_sessions=len(user_sessions)
    )


@app.get("/model-info")
async def get_model_info():
    """Get current model information"""
    return {
        "backbone_repo": BACKBONE_REPO,
        "codec_repo": CODEC_REPO,
        "device": DEVICE,
        "model_type": MODEL_TYPE,
        "is_gguf": is_gguf_model(BACKBONE_REPO),
        "max_batch_size": MAX_BATCH_SIZE if not is_gguf_model(BACKBONE_REPO) else "N/A",
        "triton_enabled": ENABLE_TRITON if not is_gguf_model(BACKBONE_REPO) else "N/A"
    }


@app.get("/voices")
async def list_voices():
    """List available preset voices"""
    sample_dir = "./sample"

    if not os.path.exists(sample_dir):
        return {"voices": []}

    voices = []
    for file in os.listdir(sample_dir):
        if file.endswith(".wav"):
            voice_name = file.replace(".wav", "")
            voices.append(voice_name)

    return {"voices": voices}


@app.post("/register-voice")
async def register_voice(request: RegisterVoiceRequest):
    """
    Register a custom voice for a user (JSON input)

    Parameters:
    - user_id: Unique user ID
    - voice_name: Voice name for this user
    - ref_text: Reference text
    - ref_audio_base64: Base64-encoded reference audio (WAV format)
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    # Save uploaded audio temporarily
    temp_audio_path = f"/tmp/{uuid.uuid4()}.wav"

    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.ref_audio_base64)
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)

        # Encode reference
        ref_codes = encode_reference(temp_audio_path)

        # Store in user session
        if request.user_id not in user_sessions:
            user_sessions[request.user_id] = {}

        user_sessions[request.user_id][request.voice_name] = {
            "ref_codes": ref_codes,
            "ref_text": request.ref_text
        }

        return JSONResponse({
            "status": "success",
            "message": f"Voice '{request.voice_name}' registered for user '{request.user_id}'",
            "user_id": request.user_id,
            "voice_name": request.voice_name
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register voice: {str(e)}")

    finally:
        # Cleanup temp file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    """
    Synthesize speech from text (JSON input/output)

    Accepts JSON request body and returns base64-encoded audio in JSON response or streaming audio.

    Parameters:
    - text: Text to synthesize
    - user_id: User ID (for registered voices)
    - voice_name: Voice name (preset or registered)
    - ref_text: Reference text (required for custom audio with ref_audio_base64)
    - ref_audio_base64: Base64-encoded reference audio (WAV format)
    - speed: Speech speed (0.5-2.0)
    - streaming: Enable streaming response (returns audio/wav stream instead of JSON)
    """
    request_start = time.time()
    request_id = str(uuid.uuid4())[:8]

    print(f"🚀 [{request_id}] TTS Request - Text: '{request.text[:50]}...' | Voice: {request.voice_name} | Speed: {request.speed}")

    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    # Handle streaming mode
    if request.streaming:
        # Determine reference source (same logic as non-streaming)
        ref_codes = None
        ref_text_final = None

        if request.ref_audio_base64:
            if not request.ref_text:
                raise HTTPException(status_code=400, detail="ref_text required for custom audio")

            temp_audio_path = f"/tmp/{uuid.uuid4()}.wav"
            try:
                audio_bytes = base64.b64decode(request.ref_audio_base64)
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_bytes)
                ref_codes = encode_reference(temp_audio_path)
                ref_text_final = request.ref_text
            finally:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

        elif request.user_id and request.voice_name:
            if request.user_id not in user_sessions or request.voice_name not in user_sessions[request.user_id]:
                raise HTTPException(
                    status_code=404,
                    detail=f"Voice '{request.voice_name}' not found for user '{request.user_id}'"
                )
            voice_data = user_sessions[request.user_id][request.voice_name]
            ref_codes = voice_data["ref_codes"]
            ref_text_final = voice_data["ref_text"]

        elif request.voice_name:
            ref_codes, ref_text_final = get_preset_voice_cached(request.voice_name)
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either: ref_audio_base64 + ref_text, user_id + voice_name, or voice_name"
            )

        # For streaming, create a unique stream URL
        stream_id = f"{request_id}_{int(time.time())}"
        stream_url = f"/stream/{stream_id}"

        prep_time = time.time() - request_start
        print(f"🌊 [{request_id}] Streaming prep: {prep_time:.3f}s")

        # Store stream generator for later access
        if not hasattr(app.state, 'active_streams'):
            app.state.active_streams = {}

        app.state.active_streams[stream_id] = {
            'generator': generate_audio_stream(request.text, ref_codes, ref_text_final, request.speed),
            'created_at': time.time()
        }

        # Create full stream URL
        base_url = os.getenv("BASE_URL", "http://localhost:8000")
        full_stream_url = f"{base_url}/stream/{stream_id}"

        return JSONResponse({
            "success": True,
            "audio_url": full_stream_url,
            "stream_id": stream_id,
            "format": "wav",
            "text": request.text,
            "voice_name": request.voice_name
        })

    # Determine reference source
    ref_source_start = time.time()
    ref_codes = None
    ref_text_final = None

    # Priority 1: Custom audio (base64)
    if request.ref_audio_base64:
        if not request.ref_text:
            raise HTTPException(status_code=400, detail="ref_text required for custom audio")

        temp_audio_path = f"/tmp/{uuid.uuid4()}.wav"
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(request.ref_audio_base64)
            with open(temp_audio_path, "wb") as f:
                f.write(audio_bytes)

            ref_codes = encode_reference(temp_audio_path)
            ref_text_final = request.ref_text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode audio: {str(e)}")
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    # Priority 2: Registered voice
    elif request.user_id and request.voice_name:
        if request.user_id not in user_sessions or request.voice_name not in user_sessions[request.user_id]:
            raise HTTPException(
                status_code=404,
                detail=f"Voice '{request.voice_name}' not found for user '{request.user_id}'"
            )

        voice_data = user_sessions[request.user_id][request.voice_name]
        ref_codes = voice_data["ref_codes"]
        ref_text_final = voice_data["ref_text"]

    # Priority 3: Preset voice
    elif request.voice_name:
        ref_codes, ref_text_final = get_preset_voice_cached(request.voice_name)

    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either: ref_audio_base64 + ref_text, user_id + voice_name, or voice_name"
        )

    ref_source_time = time.time() - ref_source_start

    # Generate speech
    inference_start = time.time()
    try:
        audio = tts_model.infer(request.text, ref_codes, ref_text_final)
        inference_time = time.time() - inference_start

        # Apply speed adjustment
        if request.speed != 1.0:
            audio = adjust_speed(audio, request.speed)

        # Save audio to file
        os.makedirs("./output_audio", exist_ok=True)
        audio_filename = f"{request_id}_{int(time.time())}.wav"
        audio_filepath = os.path.join("./output_audio", audio_filename)

        sf.write(audio_filepath, audio, 24000)

        # Create full file URL
        base_url = os.getenv("BASE_URL", "http://localhost:8000")
        audio_url = f"{base_url}/audio/{audio_filename}"

        total_time = time.time() - request_start

        # Performance logging
        print(f"⏱️  [{request_id}] Timing - Ref: {ref_source_time:.3f}s | Inference: {inference_time:.3f}s | Total: {total_time:.3f}s")
        print(f"📊 [{request_id}] Audio: {len(audio)} samples | Voice: {request.voice_name} | File: {audio_filename}")

        return JSONResponse({
            "success": True,
            "audio_url": audio_url,
            "sample_rate": 24000,
            "format": "wav",
            "text": request.text,
            "voice_name": request.voice_name,
            "timing": {
                "total_time": round(total_time, 3),
                "inference_time": round(inference_time, 3),
                "ref_prep_time": round(ref_source_time, 3)
            }
        })

    except Exception as e:
        error_time = time.time() - request_start
        print(f"❌ [{request_id}] Failed after {error_time:.3f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.post("/batch-synthesize")
async def batch_synthesize(request: BatchSynthesizeRequest):
    """
    Batch synthesize multiple texts (JSON input/output)

    Accepts JSON request body and returns base64-encoded audio in JSON response.

    Parameters:
    - texts: List of texts to synthesize
    - voice_name: Voice name (preset or registered)
    - user_id: User ID (for registered voices)
    - speed: Speech speed (0.5-2.0)
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    if not hasattr(tts_model, 'infer_batch'):
        raise HTTPException(status_code=400, detail="Batch processing not supported")

    # Get reference
    if request.user_id and request.voice_name:
        if request.user_id not in user_sessions or request.voice_name not in user_sessions[request.user_id]:
            raise HTTPException(
                status_code=404,
                detail=f"Voice '{request.voice_name}' not found for user '{request.user_id}'"
            )
        voice_data = user_sessions[request.user_id][request.voice_name]
        ref_codes = voice_data["ref_codes"]
        ref_text = voice_data["ref_text"]
    else:
        ref_codes, ref_text = get_preset_voice_cached(request.voice_name)

    try:
        # Batch generation
        audio_list = tts_model.infer_batch(request.texts, ref_codes, ref_text)

        # Combine all audio with silence padding
        sample_rate = 24000
        silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)

        combined_audio = []
        for audio in audio_list:
            if request.speed != 1.0:
                audio = adjust_speed(audio, request.speed)
            combined_audio.append(audio)
            combined_audio.append(silence)

        final_audio = np.concatenate(combined_audio[:-1])  # Remove last silence

        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, final_audio, sample_rate, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.read()

        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        return JSONResponse({
            "status": "success",
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
            "format": "wav",
            "texts": request.texts,
            "voice_name": request.voice_name,
            "num_texts": len(request.texts)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch synthesis failed: {str(e)}")

@app.get("/audio/{filename}")
async def serve_audio_file(filename: str):
    """Serve generated audio files"""
    file_path = os.path.join("./output_audio", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav", filename=filename)
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")

@app.get("/stream/{stream_id}")
async def serve_audio_stream(stream_id: str):
    """Serve audio stream"""
    if not hasattr(app.state, 'active_streams') or stream_id not in app.state.active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")

    stream_data = app.state.active_streams[stream_id]
    generator = stream_data['generator']

    # Wrap generator to include cleanup
    async def generator_with_cleanup():
        try:
            async for chunk in generator:
                yield chunk
        finally:
            # Clean up the stream after serving
            if hasattr(app.state, 'active_streams') and stream_id in app.state.active_streams:
                del app.state.active_streams[stream_id]

    return StreamingResponse(
        generator_with_cleanup(),
        media_type="audio/wav"
    )

@app.delete("/user/{user_id}")
async def delete_user_session(user_id: str):
    """Delete user session and registered voices"""
    if user_id in user_sessions:
        del user_sessions[user_id]
        return {"status": "success", "message": f"User '{user_id}' session deleted"}
    else:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)
