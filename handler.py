"""
RunPod Serverless Handler for LongCat Avatar Video Generation

Accepts: Image (base64) + Audio (base64 WAV)
Returns: Video URL (uploaded to R2)
"""

import runpod
import torch
import os
import uuid
import base64
import io
import time
import traceback
import math
import numpy as np
from PIL import Image
from pathlib import Path

# Import LongCat modules
from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video.audio_process.wav2vec2 import Wav2Vec2ModelWrapper
from longcat_video.audio_process.torch_utils import save_video_ffmpeg
from transformers import AutoTokenizer, UMT5EncoderModel, Wav2Vec2FeatureExtractor
from audio_separator.separator import Separator
import librosa

from r2_client import R2Client

# Global state - loaded once at container startup
PIPELINE = None
MODELS = {}
R2 = None
DEVICE = "cuda"
DTYPE = torch.bfloat16


def load_models():
    """Load all models at container startup."""
    global PIPELINE, MODELS, R2

    model_path = os.environ.get("MODEL_PATH", "/runpod-volume/models")
    base_path = f"{model_path}/LongCat-Video"
    avatar_path = f"{model_path}/LongCat-Video-Avatar"

    print(f"Loading models from {model_path}...")
    start = time.time()

    # Shared components (from base model)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, subfolder="tokenizer", torch_dtype=DTYPE
    )

    print("Loading text encoder...")
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_path, subfolder="text_encoder", torch_dtype=DTYPE
    ).to(DEVICE)

    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        base_path, subfolder="vae", torch_dtype=DTYPE
    ).to(DEVICE)

    print("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_path, subfolder="scheduler", torch_dtype=DTYPE
    )

    # Avatar DiT model
    print("Loading Avatar DiT model...")
    dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
        avatar_path, subfolder="avatar_single", torch_dtype=DTYPE
    ).to(DEVICE)

    # Audio models
    print("Loading audio encoder...")
    wav2vec_path = f"{avatar_path}/chinese-wav2vec2-base"
    audio_encoder = Wav2Vec2ModelWrapper(wav2vec_path).to(DEVICE)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        wav2vec_path, local_files_only=True
    )

    # Vocal separator for extracting clean vocals from audio
    print("Loading vocal separator...")
    vocal_sep_path = f"{avatar_path}/vocal_separator"
    audio_output_dir = Path("/tmp/audio_temp")
    audio_output_dir.mkdir(exist_ok=True)

    vocal_separator = Separator(
        output_dir=audio_output_dir / "vocals",
        output_single_stem="vocals",
        model_file_dir=vocal_sep_path,
    )
    vocal_separator.load_model("Kim_Vocal_2.onnx")
    MODELS["vocal_separator"] = vocal_separator
    MODELS["audio_output_dir"] = audio_output_dir

    # Create the pipeline
    print("Creating pipeline...")
    PIPELINE = LongCatVideoAvatarPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
        audio_encoder=audio_encoder,
        wav2vec_feature_extractor=wav2vec_extractor
    )
    PIPELINE.to(DEVICE)

    # Initialize R2 client
    print("Initializing R2 client...")
    R2 = R2Client()

    print(f"All models loaded in {time.time() - start:.1f} seconds")


def decode_base64_image(b64_string):
    """Decode base64 string to PIL Image."""
    # Handle data URL prefix if present (e.g., "data:image/jpeg;base64,...")
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def decode_base64_audio(b64_string):
    """Decode base64 audio to a temp file and return the path."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    audio_bytes = base64.b64decode(b64_string)
    path = f"/tmp/audio_{uuid.uuid4()}.wav"
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path


def extract_vocals(audio_path):
    """Extract vocals from audio using the vocal separator."""
    outputs = MODELS["vocal_separator"].separate(audio_path)
    if len(outputs) <= 0:
        print("Audio separation failed, using raw audio")
        return audio_path

    vocal_path = MODELS["audio_output_dir"] / "vocals" / outputs[0]
    return str(vocal_path.resolve())


def generate_avatar_video(
    image,
    audio_path,
    prompt,
    negative_prompt,
    resolution,
    num_frames,
    num_inference_steps,
    text_guidance_scale,
    audio_guidance_scale,
    seed
):
    """Generate avatar video from image and audio."""

    # Default settings from the demo
    save_fps = 16
    audio_stride = 2

    # Extract vocals from the audio
    print("Extracting vocals...")
    vocal_path = extract_vocals(audio_path)

    # Load and process audio
    print("Loading audio...")
    speech_array, sr = librosa.load(vocal_path, sr=16000)

    # Pad audio if shorter than video duration
    generate_duration = num_frames / save_fps
    source_duration = len(speech_array) / sr
    added_samples = math.ceil((generate_duration - source_duration) * sr)
    if added_samples > 0:
        speech_array = np.append(speech_array, [0.0] * added_samples)

    # Get audio embedding
    print("Computing audio embedding...")
    audio_emb = PIPELINE.get_audio_embedding(
        speech_array, fps=save_fps * audio_stride, device=DEVICE, sample_rate=sr
    )

    if torch.isnan(audio_emb).any():
        raise ValueError("Invalid audio embedding (contains NaN values)")

    # Prepare audio embedding slice (from demo code)
    indices = torch.arange(5) - 2
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames
    center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=audio_emb.shape[0] - 1)
    audio_emb_slice = audio_emb[center_indices][None, ...].to(DEVICE)

    # Generator for reproducibility
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Generate video using ai2v (audio + image to video)
    print(f"Generating video ({num_frames} frames, {num_inference_steps} steps)...")
    output_tuple = PIPELINE.generate_ai2v(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        resolution=resolution,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        text_guidance_scale=text_guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        output_type='np',
        generator=generator,
        audio_emb=audio_emb_slice
    )

    output = output_tuple[0]  # Get numpy array

    # Convert to video frames
    video_frames = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
    video_tensor = torch.from_numpy(np.array(video_frames))

    # Save video with original audio
    print("Encoding video with audio...")
    output_path = f"/tmp/output_{uuid.uuid4()}"
    save_video_ffmpeg(video_tensor, output_path, audio_path, fps=save_fps, quality=5)

    # Read video bytes
    video_file = f"{output_path}.mp4"
    with open(video_file, "rb") as f:
        video_bytes = f.read()

    # Cleanup temp files
    os.remove(video_file)
    if vocal_path != audio_path and os.path.exists(vocal_path):
        os.remove(vocal_path)
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return video_bytes


def handler(job):
    """Main RunPod serverless handler."""
    try:
        job_input = job["input"]

        # Required inputs
        if "image" not in job_input:
            return {"status": "error", "error": "Missing required field: image"}
        if "audio" not in job_input:
            return {"status": "error", "error": "Missing required field: audio"}

        image_b64 = job_input["image"]
        audio_b64 = job_input["audio"]

        # Optional inputs with defaults
        prompt = job_input.get("prompt", "A person speaking naturally to the camera")
        negative_prompt = job_input.get(
            "negative_prompt",
            "Close-up, Bright tones, overexposed, static, blurred details, subtitles, "
            "worst quality, low quality, ugly, deformed, disfigured"
        )
        resolution = job_input.get("resolution", "480p")
        num_frames = job_input.get("num_frames", 93)
        num_inference_steps = job_input.get("num_inference_steps", 50)
        text_guidance_scale = job_input.get("text_guidance_scale", 4.0)
        audio_guidance_scale = job_input.get("audio_guidance_scale", 4.0)
        seed = job_input.get("seed", 42)

        start_time = time.time()

        # Decode inputs
        print("Decoding image...")
        image = decode_base64_image(image_b64)
        print(f"Image size: {image.size}")

        print("Decoding audio...")
        audio_path = decode_base64_audio(audio_b64)

        # Generate video
        video_bytes = generate_avatar_video(
            image=image,
            audio_path=audio_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            seed=seed
        )

        # Upload to R2
        print("Uploading to R2...")
        filename = f"{uuid.uuid4()}.mp4"
        video_url = R2.upload_video(video_bytes, filename)

        generation_time = time.time() - start_time
        print(f"Total generation time: {generation_time:.1f}s")

        return {
            "status": "success",
            "video_url": video_url,
            "resolution": resolution,
            "num_frames": num_frames,
            "generation_time_seconds": round(generation_time, 1)
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Initialize models at container startup
print("=" * 50)
print("Starting LongCat Avatar Handler")
print("=" * 50)
load_models()
print("=" * 50)
print("Handler ready to receive requests!")
print("=" * 50)

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
