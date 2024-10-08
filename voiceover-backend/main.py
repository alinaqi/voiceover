import os
import uuid
import openai
import requests
import shutil
import base64
import json
import subprocess
import nltk
import tempfile
import io
from pydub import AudioSegment
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from moviepy.editor import VideoFileClip, AudioFileClip
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

from celery_app import celery  # Import the Celery instance
from elevenlabs import ElevenLabs, VoiceSettings



app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

origins = ["*"]

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
openai.api_key = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "FvmvwvObRqIHojkEGh5N"

elevenlabs_client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)

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

def extract_audio(video_path, output_audio_path):
    try:
        command = ["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", output_audio_path]
        subprocess.run(command, check=True)
        
        # Check if the audio file was actually created and has content
        if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
            return output_audio_path
        else:
            print("No audio track found in the video.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        return None

def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file_to_transcribe:
        transcript = openai.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file_to_transcribe
            )
        return transcript.text

def translate_text(text, target_language):
    prompt_messages = [
        {
            "role": "user",
            "content": f"Translate the following text into {target_language}:\n\n{text}"
        }
    ]
    params = {"model": "gpt-4", "messages": prompt_messages}
    logger.info("Translating text using OpenAI")
    result = openai.chat.completions.create(**params)
    logger.info("Translation complete")
    return result.choices[0].message.content

# Helper function to combine video and new audio
def add_voiceover_to_video(video_path, audio_file, output_path):
    video = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_file)
    final_video = video.set_audio(audio_clip)
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path

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
            Guide user through what's happening on the screen accurately following the mouse cursor. Make narration based on the text on the screen covering buttons, text areas and what user is typing. 

            Don'ts: 
            Do not mention timestamps and frames. 
            Do not mention "the video... " or "the screen...". 
            Do not greet the audience, go straight to the video.
            {frame_descriptions}
            """
        },
    ]

    params = {
        "model": "gpt-4o",
        "messages": prompt_messages
    }

    result = openai.chat.completions.create(**params)
    return result.choices[0].message.content
# Helper function to generate audio using ElevenLabs API (from the original code)
def generate_voiceover_audio_elevenlabs(narration):
    try: 
        logger.info("Generating voiceover audio using ElevenLabs API")
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
        logger.info("Voiceover audio generated")

        return audio_file, response_dict['alignment']
    except Exception as e:
        logger.error(f"Error generating voiceover: {e}")
        return None, None

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
        original_extension = file.filename.split('.')[-1].lower()
        supported_formats = ['mp4', 'webm', 'mov', 'avi', 'mkv']

        if original_extension not in supported_formats:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {original_extension}")

        original_file_path = os.path.join(UPLOAD_DIR, f"{file_id}.{original_extension}")
        mp4_file_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")

        # Save the uploaded file
        with open(original_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Convert to MP4 if the file is not already in MP4 format
        if original_extension != "mp4":
            print("Converting video to MP4 format")
            convert_to_mp4(original_file_path, mp4_file_path)
            os.remove(original_file_path)  # Remove the original file after conversion
        else:
            print("No conversion needed, file is already in MP4 format")
            mp4_file_path = original_file_path  # No conversion needed

        return {"file_id": file_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading and converting file: {e}")

# Endpoint to generate narration and return both video and audio files
@app.post("/generate_narration/{file_id}")
async def generate_narration(file_id: str, narration_request: NarrationRequest):
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Launch the Celery task for processing in the background
    print("Starting video processing task")
    #process_video_task.delay(file_id, narration_request.language, narration_request.description, narration_request.webhook_url)
    if narration_request.webhook_url:
        process_video_task_synchronous(file_id, narration_request.language, narration_request.description, narration_request.webhook_url)
    else:
        file = process_video_task_synchronous(file_id, narration_request.language, narration_request.description)
    
    print("file: ", file)
    return {"message": "Video processing started", "file_id": file_id}

def remove_audio_from_video(input_video_path, output_video_path):
    """
    Removes audio from the video file and creates a new video with no sound.
    """
    try:
        # FFmpeg command to remove the audio stream
        command = [
            "ffmpeg", "-y", "-i", input_video_path, 
            "-c", "copy", "-an", output_video_path  # `-an` removes the audio track
        ]
        subprocess.run(command, check=True)
        return output_video_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Error removing audio from video: {e}")
        raise Exception("Failed to remove audio from video")


def text_to_speech_save_mp3(text: str , output_dir: str) -> str:
    """
    Converts text to speech and saves the audio data as an MP3 file.
    
    Args:
        text (str): The text content to be converted into speech.
        output_dir (str): The directory where the MP3 file will be saved.
    
    Returns:
        str: The path to the saved MP3 file.
    """
        
        
        # Perform the text-to-speech conversion
        # response = elevenlabs_client.text_to_speech.convert(
        #     voice_id=ELEVENLABS_VOICE_ID,
        #     optimize_streaming_latency="0",
        #     output_format="mp3_22050_32",
        #     text=text,
        #     model_id="eleven_multilingual_v2",
        #     voice_settings=VoiceSettings(
        #         stability=0.0,
        #         similarity_boost=1.0,
        #         style=0.0,
        #         use_speaker_boost=True,
        #     ),
        # )
        
        # logger.info("Streaming and saving audio data...")
        
        # # Open the file in binary write mode
        # with open(filepath, 'wb') as audio_file:
        #     # Write each chunk of audio data to the file
        #     for chunk in response:
        #         if chunk:
        #             audio_file.write(chunk)
        
        # logger.info(f"Audio saved successfully to {filepath}")

        # Generate a unique filename for the MP3
    try:
        logger.info("Starting text-to-speech conversion")
        
        # Split the text into sentences
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > 4000:  # Leave some buffer
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"Text split into {len(chunks)} chunks")
        
        # Process each chunk
        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_filename = f"chunk_{i}_{uuid.uuid4()}.mp3"
            chunk_filepath = os.path.join(output_dir, chunk_filename)
            
            response = openai.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=chunk
            )
            
            response.stream_to_file(chunk_filepath)
            chunk_files.append(chunk_filepath)
            logger.info(f"Processed chunk {i+1}/{len(chunks)}")
        
        # Combine all audio files
        combined = AudioSegment.empty()
        for chunk_file in chunk_files:
            segment = AudioSegment.from_mp3(chunk_file)
            combined += segment
        
        # Save the final combined audio
        final_filename = f"final_{uuid.uuid4()}.mp3"
        final_filepath = os.path.join(output_dir, final_filename)
        combined.export(final_filepath, format="mp3")
        
        logger.info(f"Final audio saved successfully to {final_filepath}")
        
        # Clean up chunk files
        for chunk_file in chunk_files:
            os.remove(chunk_file)
        
        return final_filepath
    
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {str(e)}")
        raise

def process_video_task_synchronous(file_id: str, language: str, description: str, webhook_url: Optional[str] = None):
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    extracted_audio_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp3")
    muted_video_path = os.path.join(UPLOAD_DIR, f"{file_id}_muted.mp4")  # Video without audio


    try:
        # Check if audio track exists in the video
        logger.info("Checking for audio track in the video")
        has_audio = check_audio_track(video_path)
        translated_text = None
        if has_audio:
            # Step 1: Extract Audio
            logger.info("Extracting audio from video")
            try:
                audio_path = extract_audio(video_path, extracted_audio_path)
                logger.info("removing audio from video")
                remove_audio_from_video(video_path, muted_video_path)
                video_path = muted_video_path 
                logger.info("audio removed")
                logger.info("Audio extraction done: ", audio_path)
                # Step 2: Transcribe and translate the audio
                if audio_path:
                    logger.info("Transcribing audio")
                    transcription = transcribe_audio(audio_path)
                    logger.info("Transcription done")
                    logger.info("Translating text")
                    translated_text = translate_text(transcription, language)
                    logger.info("Translation done")
                    logger.info("Translated text: ", translated_text)
            except Exception as e:
                logger.error("Error processing audio: ", e)
        else:
            # Follow the original process (no audio track exists)
            logger.info("No audio track found, generating voiceover script from video frames and description")

            # Step 1: Extract frames with timestamps
            base64_frames_with_timestamps = extract_frames_with_timestamps(video_path)
            
            if not base64_frames_with_timestamps:
                raise Exception("Failed to extract frames")

            # Step 2: Generate voiceover script using OpenAI
            translated_text = generate_voiceover_script(base64_frames_with_timestamps, language=language, description=description)

        # Step 3: Generate voiceover for the translated text
        logger.info("Generating voiceover audio")

        translated_audio_file = text_to_speech_save_mp3(translated_text,UPLOAD_DIR) #generate_voiceover_audio_elevenlabs(translated_text)

        logger.info(f"Audio file generated: {translated_audio_file}")
        if not translated_audio_file:
            raise Exception("Error generating audio")

        # Step 4: Combine original video with translated voiceover
        output_video_path = os.path.join(UPLOAD_DIR, f"{file_id}_{language}_narrated.mp4")
        logger.info("Combining video and audio for", file_id)
        final_video_path = add_voiceover_to_video(video_path, translated_audio_file, output_video_path)

        # Send notification to the webhook
        if webhook_url:
            logger.info("Notifying webhook")
            notify_webhook(webhook_url, {"file_id": file_id, "output_video": final_video_path, "voiceover_script": translated_text})
        else:
            logger.info(f"Processing complete for file_id: {file_id}. Video saved at: {final_video_path}")
    except Exception as e:
        notify_webhook(webhook_url, {"error": str(e), "file_id": file_id})


@celery.task(name="process_video_task")
def process_video_task(file_id: str, language: str, description: str, webhook_url: Optional[str] = None):
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    extracted_audio_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp3")
    muted_video_path = os.path.join(UPLOAD_DIR, f"{file_id}_muted.mp4")  # Video without audio


    try:
        # Check if audio track exists in the video
        logger.info("Checking for audio track in the video")
        has_audio = check_audio_track(video_path)
        translated_text = None
        if has_audio:
            # Step 1: Extract Audio
            logger.info("Extracting audio from video")
            try:
                audio_path = extract_audio(video_path, extracted_audio_path)
                logger.info("removing audio from video")
                remove_audio_from_video(video_path, muted_video_path)
                video_path = muted_video_path 
                logger.info("audio removed")
                logger.info("Audio extraction done: ", audio_path)
                # Step 2: Transcribe and translate the audio
                if audio_path:
                    logger.info("Transcribing audio")
                    transcription = transcribe_audio(audio_path)
                    logger.info("Transcription done")
                    logger.info("Translating text")
                    translated_text = translate_text(transcription, language)
                    logger.info("Translation done")
                    logger.info("Translated text: ", translated_text)
            except Exception as e:
                logger.error("Error processing audio: ", e)
        else:
            # Follow the original process (no audio track exists)
            logger.info("No audio track found, generating voiceover script from video frames and description")

            # Step 1: Extract frames with timestamps
            base64_frames_with_timestamps = extract_frames_with_timestamps(video_path)
            
            if not base64_frames_with_timestamps:
                raise Exception("Failed to extract frames")

            # Step 2: Generate voiceover script using OpenAI
            translated_text = generate_voiceover_script(base64_frames_with_timestamps, language=language, description=description)

        # Step 3: Generate voiceover for the translated text
        logger.info("Generating voiceover audio")

        translated_audio_file = text_to_speech_save_mp3(translated_text,UPLOAD_DIR) #generate_voiceover_audio_elevenlabs(translated_text)

        return translated_audio_file
        logger.info(f"Audio file generated: {translated_audio_file}")
        #translated_audio_file = "3c03d613-9090-4581-8a2c-5561fba9fd77.mp3" 
        if not translated_audio_file:
            raise Exception("Error generating audio")

        # Step 4: Combine original video with translated voiceover
        output_video_path = os.path.join(UPLOAD_DIR, f"{file_id}_{language}_narrated.mp4")
        logger.info("Combining video and audio for", file_id)
        final_video_path = add_voiceover_to_video(video_path, translated_audio_file, output_video_path)

        # Send notification to the webhook
        if webhook_url:
            logger.info("Notifying webhook")
            notify_webhook(webhook_url, {"file_id": file_id, "output_video": final_video_path, "voiceover_script": translated_text})
        else:
            logger.info(f"Processing complete for file_id: {file_id}. Video saved at: {final_video_path}")
    except Exception as e:
        notify_webhook(webhook_url, {"error": str(e), "file_id": file_id})

def check_audio_track(video_path: str) -> bool:
    """
    Check if a video file contains an audio track using ffmpeg.
    Returns True if an audio track is found, otherwise False.
    """
    try:
        print(f"Checking audio track in video: {video_path}")
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error checking audio track: {e}")
        return False

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