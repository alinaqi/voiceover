import os
import uuid
import openai
import requests
import shutil
import base64
import json
import subprocess

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from moviepy.editor import VideoFileClip, AudioFileClip
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

from celery_app import celery  # Import the Celery instance
import time

app = FastAPI()

origins = [
    "*",  
    "*",  

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows requests from these origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# Directory for storing uploaded videos
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Set API keys for OpenAI and ElevenLabs
openai.api_key = os.getenv("OPENAI_API_KEY", "<YOUR_OPENAI_API_KEY>")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "<YOUR_ELEVEN_LABS_API_KEY>")
ELEVENLABS_VOICE_ID = "nsQAxyXwUKBvqtEK9MfK"


# Model for generating narration request
class NarrationRequest(BaseModel):
    language: str
    description: str  # User-provided description to make narration more accurate
    webhook_url: Optional[str] = None  # URL to call when processing is complete

def convert_to_mp4(input_path, output_path):
    try:
        # Use the ffmpeg command to convert the input .webm file to an output .mp4 file
        command = ["ffmpeg", "-y", "-i", input_path, output_path]  # `-y` flag to overwrite if needed
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e}")
        raise HTTPException(status_code=500, detail="Error converting video")


# Helper function to extract frames and timestamps (from the original code)
def extract_frames_with_timestamps(video_path, frame_interval=30):
    import cv2
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    base64_frames_with_timestamps = []

    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frame = base64.b64encode(buffer).decode("utf-8")
            timestamp = frame_count / frame_rate
            base64_frames_with_timestamps.append((base64_frame, timestamp))

        frame_count += 1

    video_capture.release()
    return base64_frames_with_timestamps

# Helper function to generate voiceover script (from the original code)
def generate_voiceover_script(base64_frames_with_timestamps, language="English", description=""):
    frame_descriptions = "\n".join([f"Frame at {timestamp:.2f} seconds." for _, timestamp in base64_frames_with_timestamps[0::50]])

    prompt_messages = [
        {
            "role": "user",
            "content": f"""
            This is an explainer video. Narrate in {language}. 
            
            Here is the brief description and background of the video to make the narration more accurate: {description}

            Focus on explaining what's happening on the screen and in the video.
            Assume you are demo'ing the software to an audience so use second person language e.g "you can do this..."
            Pay attention to the button texts, labels etc and mention it in your narration so users can follow you. 
            Make sure you narrate what is on the screen and everything is in sync with frame timestamps.
            Guide user through what's happening on the screen accurately following the mouse cursor. Assume you are an instructor guiding a student.

            Don'ts: 
            Do not mention timestamps and frames. 
            Do not mention "the video... " or "the screen...". 
            Do not greet the audience, go straight to the video.
            {frame_descriptions}
            """
        },
    ]

    params = {
        "model": "gpt-4o-mini",
        "messages": prompt_messages
    }

    result = openai.chat.completions.create(**params)
    return result.choices[0].message.content
# Helper function to generate audio using ElevenLabs API (from the original code)
def generate_voiceover_audio_elevenlabs(narration):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/with-timestamps"
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
    }
    payload = {
        "text": narration,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print(f"Error generating voiceover: {response.status_code}, {response.text}")
        return None, None

    response_dict = response.json()
    audio_bytes = base64.b64decode(response_dict["audio_base64"])
    audio_file = f"{uuid.uuid4()}.mp3"
    with open(audio_file, "wb") as f:
        f.write(audio_bytes)

    return audio_file, response_dict['alignment']

# Helper function to combine video and audio (from the original code)
def add_voiceover_to_video(video_path, audio_file, output_path):
    video = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_file)
    final_video = video.set_audio(audio_clip)
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path

# Endpoint to upload video
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        temp_file_path = os.path.join(UPLOAD_DIR, f"{file_id}-temp.webm")
        mp4_file_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")

        # Save the uploaded file as a temporary .webm file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Convert the temporary .webm file to .mp4
        convert_to_mp4(temp_file_path, mp4_file_path)

        # Remove the temporary .webm file after conversion
        os.remove(temp_file_path)

        return {"file_id": file_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading and converting file: {e}")

# Endpoint to generate narration and return both video and audio files
# Endpoint to start background processing
@app.post("/generate_narration/{file_id}")
async def generate_narration(file_id: str, narration_request: NarrationRequest):
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Launch the Celery task for processing in the background
    print("Starting video processing task")
    process_video_task.delay(file_id, narration_request.language, narration_request.description, narration_request.webhook_url)
    
    return {"message": "Video processing started", "file_id": file_id}

@celery.task(name="process_video_task")
def process_video_task(file_id: str, language: str, description: str,  webhook_url: Optional[str] = None):
    print(f"Processing video: {file_id}")
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")

    try:
        # Step 1: Extract frames with timestamps
        print("Extracting frames with timestamps for ", file_id)
        base64_frames_with_timestamps = extract_frames_with_timestamps(video_path)

        if base64_frames_with_timestamps:
            # Step 2: Generate voiceover script using OpenAI
            print("Generating voiceover script for ", file_id)
            voiceover_script = generate_voiceover_script(base64_frames_with_timestamps, language=language, description=description)
            
            # Step 3: Generate audio using ElevenLabs API
            audio_file, alignment = generate_voiceover_audio_elevenlabs(voiceover_script)

            if audio_file:
                # Step 4: Combine the video and audio
                output_video_path = os.path.join(UPLOAD_DIR, f"{file_id}_{language}_narrated.mp4")
                print("Combining video and audio for ", file_id)
                final_video_path = add_voiceover_to_video(video_path, audio_file, output_video_path)
                
                # Send notification to the webhook
                # If webhook_url is provided, notify the webhook
                if webhook_url:
                    notify_webhook(webhook_url, {"file_id": file_id, "output_video": final_video_path, "voiceover_script": voiceover_script})
                else:
                    print(f"Processing complete for file_id: {file_id}. Video saved at: {final_video_path}")
            else:
                raise Exception("Error generating audio")

    except Exception as e:
        notify_webhook(webhook_url, {"error": str(e), "file_id": file_id})

def notify_webhook(url, data):
    try:
        print(f"Calling webhook: {url}")
        response = requests.post(url, json=data)
        if response.status_code != 200:
            print(f"Error calling webhook: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Webhook call failed: {e}")


# Endpoint to get the video with narration
@app.get("/video/{file_id}")
async def get_video(file_id: str):
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}_narrated.mp4")

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Narrated video not found")

    return {"video_path": video_path}

@celery.task(name="test_task")
def test_task(x, y):
    print(f"Adding {x} + {y}")
    return x + y

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
