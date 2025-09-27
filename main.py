# main.py
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import base64
import io
import json
import asyncio
import time
import random
from datetime import datetime
from urllib.parse import quote

from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect, Body, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import yt_dlp
import cv2
import pandas as pd
from PIL import Image
import httpx
import re
from dotenv import load_dotenv


import io
from fastapi import UploadFile, File
from openai import OpenAI

# OpenAI SDK v1.x (pip install openai>=1.30)
from openai import OpenAI  # uses OPENAI_API_KEY env var by default

# --------- Configuration ----------
BASE_DIR = Path(__file__).parent.resolve()
DOWNLOADS_DIR = BASE_DIR / "downloads"
TEMPLATES_DIR = BASE_DIR / "templates"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Load env vars from .env (if present) and .env.txt (explicit file requested by user)
load_dotenv()
load_dotenv(".env.txt")
# Add these lines after your existing load_dotenv() calls
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123") 
EXPERT_PASSWORD = os.getenv("EXPERT_PASSWORD", "expert123")

def get_openai_client() -> OpenAI:
    """
    Create a singleton OpenAI client using ONLY the environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError("Missing OPENAI_API_KEY in environment (.env / .env.txt).")
    return OpenAI(api_key=api_key)

# Initialize once (fail fast if key missing)
OPENAI_CLIENT = get_openai_client()

app = FastAPI(title="YouTube Downloader")

# Serve the downloads directory so the user can click the files
app.mount("/downloads", StaticFiles(directory=str(DOWNLOADS_DIR)), name="downloads")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov"}
EXPERT_QUESTION_TYPES = [
    ("character", "Character"),
    ("setting", "Setting"),
    ("feeling", "Feeling"),
    ("action", "Action"),
    ("causal", "Causal"),
    ("outcome", "Outcome"),
    ("prediction", "Prediction"),
]
EXPERT_QUESTION_TYPE_VALUES = {value for value, _ in EXPERT_QUESTION_TYPES}
EXPERT_QUESTION_TYPE_LABELS = {value: label for value, label in EXPERT_QUESTION_TYPES}

def normalize_segment_value(value: Any) -> float:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return 0.0


# -----------------------------
# Download helpers
# -----------------------------
def download_youtube(url: str) -> Dict[str, Any]:
    """
    Download best-quality video and English subtitles (WebVTT) from a YouTube URL.
    Returns a dict with success flag, message, video_id, and list of file paths (relative to downloads/).
    """
    result = {"success": False, "message": "Download error", "video_id": None, "files": []}

    if not url:
        result["message"] = "No URL provided."
        return result

    if not (url.startswith("http") and ("youtube.com" in url or "youtu.be" in url)):
        result["message"] = "Please provide a valid YouTube URL."
        return result

    try:
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            video_id = info.get("id", "unknown")

        result["video_id"] = video_id
        video_dir = DOWNLOADS_DIR / video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        # Use WebVTT for browser <track> captions
        ydl_opts = {
            "outtmpl": str(video_dir / f"{video_id}.%(ext)s"),
            "format": "best",
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "subtitlesformat": "vtt",
            "ignoreerrors": True,
            "no_warnings": True,
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Collect created files (relative to downloads/)
        created: List[str] = []
        if video_dir.exists():
            for p in sorted(video_dir.iterdir()):
                if p.is_file():
                    rel = p.relative_to(DOWNLOADS_DIR).as_posix()
                    created.append(rel)

        if created:
            result["success"] = True
            result["message"] = "Video downloaded"
            result["files"] = created
        else:
            result["message"] = "Download finished but no files were found."

        return result

    except yt_dlp.utils.DownloadError as e:
        result["message"] = f"Download error: {e}"
        return result
    except Exception as e:
        result["message"] = f"Unexpected error: {e}"
        return result


def list_all_downloads() -> List[dict]:
    """
    Enumerate all per-video download folders and files under downloads/.
    Returns: [{"video_id": "...", "files": ["/downloads/.../file.ext", ...]}]
    """
    results: List[dict] = []
    if not DOWNLOADS_DIR.exists():
        return results

    for item in sorted(DOWNLOADS_DIR.iterdir()):
        if item.is_dir():
            vid = item.name
            links = []
            # top-level files
            for p in sorted(item.iterdir()):
                if p.is_file():
                    rel = p.relative_to(DOWNLOADS_DIR).as_posix()
                    links.append(f"/downloads/{rel}")
            # extracted_frames files (if any)
            frames_dir = item / "extracted_frames"
            if frames_dir.exists():
                for p in sorted(frames_dir.iterdir()):
                    if p.is_file():
                        rel = p.relative_to(DOWNLOADS_DIR).as_posix()
                        links.append(f"/downloads/{rel}")

            results.append({"video_id": vid, "files": links})
    return results


def find_current_video_and_sub(video_id: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Given a video_id, return URLs to the first playable video file and English WebVTT subtitle (if any).
    """
    video_url = None
    sub_url = None
    if not video_id:
        return {"video": None, "sub": None}

    video_dir = DOWNLOADS_DIR / video_id
    if not video_dir.exists():
        return {"video": None, "sub": None}

    # First available video file
    video_path = None
    for pat in ("*.mp4", "*.webm", "*.mkv"):
        vids = sorted(video_dir.glob(pat))
        if vids:
            video_path = vids[0]
            break
    if video_path:
        video_url = f"/downloads/{video_path.relative_to(DOWNLOADS_DIR).as_posix()}"

    # Prefer English WebVTT (yt-dlp usually names like <id>.en.vtt)
    subs = sorted(video_dir.glob("*.en.vtt")) + sorted(video_dir.glob("*.vtt"))
    if subs:
        sub_url = f"/downloads/{subs[0].relative_to(DOWNLOADS_DIR).as_posix()}"

    return {"video": video_url, "sub": sub_url}


# -----------------------------
# Frame extraction
# -----------------------------
def extract_frames_per_second_for_video(video_id: str) -> Dict[str, Any]:
    """
    Extract 1 frame per second from the downloaded video in downloads/<video_id>.
    """
    folder_path = DOWNLOADS_DIR / video_id
    if not folder_path.exists():
        return {"success": False, "message": f"Folder '{video_id}' not found.", "files": []}

    video_files = []
    for ext in ("*.mp4", "*.webm", "*.mkv"):
        video_files.extend(folder_path.glob(ext))
    if not video_files:
        return {"success": False, "message": f"No video files found in '{video_id}'.", "files": []}

    video_file = video_files[0]
    output_dir = folder_path / "extracted_frames"
    output_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        return {"success": False, "message": f"Error opening video file: {video_file.name}", "files": []}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps is None or fps <= 0:
        cap.release()
        return {"success": False, "message": "Invalid FPS detected.", "files": []}

    duration = total_video_frames / fps
    total_seconds = int(duration)

    frame_data = []
    for second in range(total_seconds):
        frame_number = int(second * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_filename = f"frame_{second:04d}s.jpg"
        frame_path = output_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)

        frame_info = {
            "frame_number": second + 1,
            "timestamp_seconds": second,
            "timestamp_formatted": f"{second // 60:02d}:{second % 60:02d}",
            "filename": frame_filename,
            "file_path": str(frame_path),
        }
        frame_data.append(frame_info)

    cap.release()

    json_path = output_dir / "frame_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video_info": {
                    "filename": video_file.name,
                    "duration_seconds": duration,
                    "total_frames": total_video_frames,
                    "fps": fps,
                    "extracted_frames": len(frame_data),
                },
                "frames": frame_data,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    csv_path = output_dir / "frame_data.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Frame,Timestamp,Time_Formatted,Filename\n")
        for fr in frame_data:
            f.write(
                f"{fr['frame_number']},{fr['timestamp_seconds']},"
                f"{fr['timestamp_formatted']},{fr['filename']}\n"
            )

    links: List[str] = []
    if output_dir.exists():
        for p in sorted(output_dir.iterdir()):
            if p.is_file():
                rel = p.relative_to(DOWNLOADS_DIR).as_posix()
                links.append(f"/downloads/{rel}")

    return {
        "success": True,
        "message": f"Extracted {len(frame_data)} frames to '{output_dir.name}'.",
        "files": links,
        "video_id": video_id,
        "output_dir": f"/downloads/{video_id}/extracted_frames",
        "count": len(frame_data),
    }


# -----------------------------
# Question generation helpers
# -----------------------------
def encode_image_to_base64(image_path, max_size=(512, 512)):
    """Convert image to base64 string with optional resizing for efficiency"""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()
            return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def time_to_seconds(time_str):
    """Convert time string (HH:MM:SS or MM:SS) to seconds"""
    try:
        parts = time_str.split(':')
        if len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        else:  # Just seconds
            return int(parts[0])
    except:
        return 0


def parse_iso8601_duration_to_seconds(duration: str) -> int:
    """Parse ISO8601 duration like PT1H2M3S to total seconds."""
    if not duration:
        return 0
    # Support hours, minutes, seconds (YouTube typical)
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not m:
        return 0
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    seconds = int(m.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def format_hhmmss(total_seconds: int) -> str:
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


def read_frame_data_from_csv(folder_name, start_time, end_time):
    """Read frame data from CSV file and get frames within specified time range"""
    folder_path = Path(folder_name)
    frames_dir = folder_path / "extracted_frames"
    csv_path = frames_dir / "frame_data.csv"

    if not csv_path.exists():
        return [], ""

    try:
        df = pd.read_csv(csv_path)

        # Convert time strings to seconds for filtering
        if 'Time_Formatted' in df.columns:
            df['Time_Seconds'] = df['Time_Formatted'].apply(time_to_seconds)
        elif 'Time_Seconds' in df.columns:
            pass
        else:
            df['Time_Seconds'] = df.index  # fallback

        filtered_frames = df[(df['Time_Seconds'] >= start_time) & (df['Time_Seconds'] <= end_time)]
        if len(filtered_frames) == 0:
            return [], ""

        frame_data = []
        transcript_parts = []

        for _, row in filtered_frames.iterrows():
            image_path = frames_dir / row['Filename']
            frame_info = {
                'image_path': image_path,
                'subtitle_text': row.get('Subtitle_Text', 'No transcript available'),
                'time_seconds': row.get('Time_Seconds', 0),
                'time_formatted': row.get('Time_Formatted', '')
            }
            frame_data.append(frame_info)

            subtitle = row.get('Subtitle_Text', '')
            if subtitle and subtitle not in ['No transcript available', 'No subtitle at this time', 'No subtitles available']:
                time_label = row.get('Time_Formatted', f"{row.get('Time_Seconds', 0)}s")
                transcript_parts.append(f"[{time_label}] {subtitle}")

        complete_transcript = "\n".join(transcript_parts) if transcript_parts else "No transcript available for this video segment."
        return frame_data, complete_transcript

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return [], ""


def generate_questions_for_segment(video_id: str, start_time: int, end_time: int) -> Optional[str]:
    """
    Analyze frames + transcript for a time window and return JSON text with the questions.
    Uses env-provided OPENAI_API_KEY only. Optimized for rate limits with retry logic.
    """
    folder_name = str(DOWNLOADS_DIR / video_id)
    try:
        client = OPENAI_CLIENT
    except Exception as e:
        print(f"Error creating OpenAI client: {e}")
        return None

    frame_data, complete_transcript = read_frame_data_from_csv(folder_name, start_time, end_time)
    if not frame_data:
        return None

    duration = end_time - start_time + 1  # inclusive window
    
    # First attempt with standard prompt
    base_prompt = f"""You are an early childhood educator designing comprehension questions for children ages 6–8. 
    Analyze the video content using both the visual frames and the complete transcript provided below.

COMPLETE TRANSCRIPT:
==========================================
{complete_transcript}
==========================================

TASK:
I am providing you with {len(frame_data)} sequential frames from a {duration}-second segment ({start_time}s to {end_time}s) of a video, 
along with the complete transcript above.

Please do the following:

1. Provide ONE short, child-friendly comprehension question for EACH of the following categories:
   - Character
   - Setting
   - Feeling
   - Action
   - Causal Relationship
   - Outcome
   - Prediction

2. After creating the questions, rank the questions based on how relevant and good it is to test comprehension and active viewing, the best question will be ranked 1

3. Return JSON only (no extra text) in this structure:
{{
  "questions": {{
    "character": {{ "q": "...", "a": "...", "rank":"" }},
    "setting": {{ "q": "...", "a": "...", "rank":"" }},
    "feeling": {{ "q": "...", "a": "...", "rank":"" }},
    "action": {{ "q": "...", "a": "...", "rank":"" }},
    "causal": {{ "q": "...", "a": "...", "rank":"" }},
    "outcome": {{ "q": "...", "a": "...", "rank":"" }},
    "prediction": {{ "q": "...", "a": "...", "rank":"" }}
  }},
  "best_question": "..."
}}
"""

    # Second attempt with more persuasive prompt
    polite_prompt = f"""Please help me create educational questions for young children. This is a children's educational video with no violence or inappropriate content - it's designed to teach kids about nature and the environment.

COMPLETE TRANSCRIPT:
==========================================
{complete_transcript}
==========================================

I am providing you with {len(frame_data)} sequential frames from a {duration}-second segment ({start_time}s to {end_time}s) of this educational children's video, along with the complete transcript above.

Please create ONE short, child-friendly comprehension question for EACH of the following categories:
- Character
- Setting  
- Feeling
- Action
- Causal Relationship
- Outcome
- Prediction

After creating the questions, please rank the questions based on how relevant and good it is to test comprehension and active viewing, the best question will be ranked 1

Return JSON only (no extra text) in this structure:
{{
  "questions": {{
    "character": {{ "q": "...", "a": "...", "rank":"" }},
    "setting": {{ "q": "...", "a": "...", "rank":"" }},
    "feeling": {{ "q": "...", "a": "...", "rank":"" }},
    "action": {{ "q": "...", "a": "...", "rank":"" }},
    "causal": {{ "q": "...", "a": "...", "rank":"" }},
    "outcome": {{ "q": "...", "a": "...", "rank":"" }},
    "prediction": {{ "q": "...", "a": "...", "rank":"" }}
  }},
  "best_question": "..."
}}
"""

    content = []

    # Sample frames to stay under token limits (max 5 frames)
    max_frames = 5
    if len(frame_data) > max_frames:
        step = len(frame_data) // max_frames
        sampled_frames = [frame_data[i] for i in range(0, len(frame_data), step)][:max_frames]
    else:
        sampled_frames = frame_data

    # Add sampled frames as low-detail inline images
    successful_frames = 0
    for fr in sampled_frames:
        b64 = encode_image_to_base64(fr['image_path'])
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low"
                }
            })
            successful_frames += 1

    if successful_frames == 0:
        return None

    # Try both prompts with retry logic
    prompts_to_try = [base_prompt, polite_prompt]
    
    for attempt_round, prompt in enumerate(prompts_to_try):
        content_with_prompt = [{"type": "text", "text": prompt}] + content
        
        # Retry logic with exponential backoff for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": content_with_prompt}],
                    max_tokens=1500,
                    temperature=0.3
                )
                result_content = resp.choices[0].message.content
                
                # Check if the response is a refusal
                if result_content and not any(refusal in result_content.lower() for refusal in 
                    ["i'm sorry", "i can't", "i'm unable", "cannot assist", "can't assist"]):
                    return result_content
                    
                # If first prompt failed, try second prompt
                if attempt_round == 0:
                    print(f"First prompt attempt failed for segment {start_time}-{end_time}s, trying polite prompt")
                    break
                    
            except Exception as e:
                if "rate_limit_exceeded" in str(e) and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                    print(f"Rate limit hit, waiting {wait_time:.1f} seconds before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Error calling OpenAI API: {e}")
                    if attempt_round == 0:
                        break  # Try second prompt
                    return None
    
    print(f"Both prompt attempts failed for segment {start_time}-{end_time}s")
    return None

def build_segments_from_duration(duration_seconds: int, interval_seconds: int, start_offset: int = 0) -> List[tuple]:
    """
    Build inclusive segments like (0, 60), (61, 120), ... until duration_seconds.
    """
    segments = []
    start = max(0, int(start_offset))
    step = max(1, int(interval_seconds))
    while start <= duration_seconds:
        end = min(start + step - 1, duration_seconds)
        segments.append((start, end))
        if end >= duration_seconds:
            break
        start = end + 1
    return segments

# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    """Home page with user type selection"""
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/home", response_class=HTMLResponse)
def home_redirect(request: Request):
    """Alternative home page route"""
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/children", response_class=HTMLResponse)
def children_page(request: Request):
    """Children's learning interface - no password required"""
    return templates.TemplateResponse("children.html", {"request": request})

@app.post("/api/verify-password")
async def verify_password(request: Request, user_type: str = Form(...), password: str = Form(...)):
    """Verify password for admin/expert access"""
    valid_passwords = {
        "admin": ADMIN_PASSWORD,
        "expert": EXPERT_PASSWORD
    }
    
    if user_type in valid_passwords and password == valid_passwords[user_type]:
        if user_type == "admin":
            return JSONResponse({"success": True, "redirect": "/admin"})
        elif user_type == "expert":
            return JSONResponse({"success": True, "redirect": "/expert-preview"})
    else:
        return JSONResponse({"success": False, "message": "Invalid password"})





# -----------------------------
# YouTube Search API (child-safe with duration filters)
# -----------------------------
@app.get("/api/yt_search")
async def yt_search(
    request: Request,
    q: str,
    min_minutes: int = 5,  # More permissive minimum
    max_minutes: int = 120,  # More permissive maximum
    max_results: int = 50,
    page_token: Optional[str] = None,
    page_size: Optional[int] = 20,
):
    """
    Search YouTube for videos with more permissive filtering.
    Requires YOUTUBE_API_KEY env var.
    """
    api_key = YOUTUBE_API_KEY
    if not api_key:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Missing YOUTUBE_API_KEY on server.",
                "items": [],
            },
        )

    q = (q or "").strip()
    if not q:
        # Provide default educational suggestions if no query
        q = "educational kids learning children"

    min_seconds = max(0, int(min_minutes)) * 60
    max_seconds = max(0, int(max_minutes)) * 60
    if max_seconds and max_seconds < min_seconds:
        min_seconds, max_seconds = max_seconds, min_seconds

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # 1) Search for videos with more permissive settings
            size = int(page_size) if page_size is not None else 20
            if size <= 0:
                size = 20
            if size > 50:
                size = 50

            search_params = {
                "part": "snippet",
                "q": q + " educational kids children learning",  # Add educational keywords
                "type": "video",
                "maxResults": min(size * 2, 50),  # Get more results to filter from
                "safeSearch": "strict",
                "videoEmbeddable": "true",
                "order": "relevance",
                "key": api_key,
            }
            if page_token:
                search_params["pageToken"] = page_token
            search_resp = await client.get(
                "https://www.googleapis.com/youtube/v3/search", params=search_params
            )
            search_resp.raise_for_status()
            search_data = search_resp.json()

            video_ids = [
                item.get("id", {}).get("videoId")
                for item in search_data.get("items", [])
                if item.get("id", {}).get("kind") == "youtube#video"
                and item.get("id", {}).get("videoId")
            ]
            if not video_ids:
                return JSONResponse(
                    content={
                        "success": True,
                        "items": [],
                        "message": "No results found.",
                    }
                )

            # 2) Fetch details with more permissive filtering
            videos_params = {
                "part": "snippet,contentDetails,status",
                "id": ",".join(video_ids),
                "key": api_key,
                "maxResults": 50,
            }
            videos_resp = await client.get(
                "https://www.googleapis.com/youtube/v3/videos", params=videos_params
            )
            videos_resp.raise_for_status()
            videos_data = videos_resp.json()

            items_out: List[Dict[str, Any]] = []
            for v in videos_data.get("items", []):
                vid = v.get("id")
                snippet = v.get("snippet", {})
                content_details = v.get("contentDetails", {})
                status = v.get("status", {})

                duration_iso = content_details.get("duration")
                duration_seconds = parse_iso8601_duration_to_seconds(duration_iso)
                if duration_seconds <= 0:
                    continue

                # More flexible duration filtering - allow 80% variance
                duration_variance = 0.8
                adjusted_min = max(0, min_seconds * duration_variance)
                adjusted_max = max_seconds * (1 + duration_variance) if max_seconds else float('inf')
                
                if duration_seconds < adjusted_min:
                    continue
                if max_seconds and duration_seconds > adjusted_max:
                    continue

                # Much more permissive content filtering - only block explicitly restricted content
                content_rating = content_details.get("contentRating", {})
                if content_rating.get("ytRating") == "ytAgeRestricted":
                    continue
                
                # Check for obviously inappropriate content in title/description
                title_lower = snippet.get("title", "").lower()
                description_lower = snippet.get("description", "").lower()
                inappropriate_keywords = ["violence", "horror", "scary", "adult", "mature"]
                
                if any(keyword in title_lower or keyword in description_lower for keyword in inappropriate_keywords):
                    continue

                thumbnails = snippet.get("thumbnails", {})
                thumb = (
                    thumbnails.get("medium", {}).get("url")
                    or thumbnails.get("high", {}).get("url")
                    or thumbnails.get("default", {}).get("url")
                )
                items_out.append(
                    {
                        "videoId": vid,
                        "title": snippet.get("title"),
                        "channel": snippet.get("channelTitle"),
                        "durationSeconds": duration_seconds,
                        "durationFormatted": format_hhmmss(duration_seconds),
                        "thumbnail": thumb,
                        "url": f"https://www.youtube.com/watch?v={vid}",
                    }
                )

            return JSONResponse(
                content={
                    "success": True,
                    "items": items_out[:size],
                    "count": len(items_out[:size]),
                    "page_size": size,
                    "nextPageToken": search_data.get("nextPageToken"),
                    "searchTotal": search_data.get("pageInfo", {}).get("totalResults"),
                    "message": f"Showing up to {size} results ordered by relevance.",
                }
            )

    except httpx.HTTPStatusError as e:
        return JSONResponse(
            status_code=502,
            content={"success": False, "message": f"YouTube API error: {e}"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Server error: {e}"},
        )
    """
    Search YouTube for kid-safe videos and filter by duration window.
    Requires YOUTUBE_API_KEY env var.
    """
    api_key = YOUTUBE_API_KEY
    if not api_key:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Missing YOUTUBE_API_KEY on server.",
                "items": [],
            },
        )

    q = (q or "").strip()
    if not q:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Empty query.", "items": []},
        )

    min_seconds = max(0, int(min_minutes)) * 60
    max_seconds = max(0, int(max_minutes)) * 60
    if max_seconds and max_seconds < min_seconds:
        min_seconds, max_seconds = max_seconds, min_seconds

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # 1) Search for videos (safeSearch=strict, embeddable)
            size = int(page_size) if page_size is not None else 20
            if size <= 0:
                size = 20
            if size > 50:
                size = 50

            search_params = {
                "part": "snippet",
                "q": q,
                "type": "video",
                "maxResults": min(size, 50),
                "safeSearch": "strict",
                "videoEmbeddable": "true",
                "order": "relevance",
                "key": api_key,
            }
            if page_token:
                search_params["pageToken"] = page_token
            search_resp = await client.get(
                "https://www.googleapis.com/youtube/v3/search", params=search_params
            )
            search_resp.raise_for_status()
            search_data = search_resp.json()

            video_ids = [
                item.get("id", {}).get("videoId")
                for item in search_data.get("items", [])
                if item.get("id", {}).get("kind") == "youtube#video"
                and item.get("id", {}).get("videoId")
            ]
            if not video_ids:
                return JSONResponse(
                    content={
                        "success": True,
                        "items": [],
                        "message": "No results found.",
                    }
                )

            # 2) Fetch details (duration, madeForKids, thumbnails)
            videos_params = {
                "part": "snippet,contentDetails,status",
                "id": ",".join(video_ids),
                "key": api_key,
                "maxResults": 50,
            }
            videos_resp = await client.get(
                "https://www.googleapis.com/youtube/v3/videos", params=videos_params
            )
            videos_resp.raise_for_status()
            videos_data = videos_resp.json()

            items_out: List[Dict[str, Any]] = []
            for v in videos_data.get("items", []):
                vid = v.get("id")
                snippet = v.get("snippet", {})
                content_details = v.get("contentDetails", {})
                status = v.get("status", {})

                duration_iso = content_details.get("duration")
                duration_seconds = parse_iso8601_duration_to_seconds(duration_iso)
                if duration_seconds <= 0:
                    continue

                if duration_seconds < min_seconds:
                    continue
                if max_seconds and duration_seconds > max_seconds:
                    continue

                # More permissive child-safe filter - only block age-restricted content
                content_rating = content_details.get("contentRating", {})
                if content_rating.get("ytRating") == "ytAgeRestricted":
                    continue
                # Note: Removed strict madeForKids requirement to get more results

                # YouTube may also mark age-restricted; guard just in case
                content_rating = content_details.get("contentRating", {})
                if content_rating.get("ytRating") == "ytAgeRestricted":
                    continue

                thumbnails = snippet.get("thumbnails", {})
                thumb = (
                    thumbnails.get("medium", {}).get("url")
                    or thumbnails.get("high", {}).get("url")
                    or thumbnails.get("default", {}).get("url")
                )
                items_out.append(
                    {
                        "videoId": vid,
                        "title": snippet.get("title"),
                        "channel": snippet.get("channelTitle"),
                        "durationSeconds": duration_seconds,
                        "durationFormatted": format_hhmmss(duration_seconds),
                        "thumbnail": thumb,
                        "url": f"https://www.youtube.com/watch?v={vid}",
                    }
                )

            return JSONResponse(
                content={
                    "success": True,
                    "items": items_out[:size],
                    "count": len(items_out[:size]),
                    "page_size": size,
                    "nextPageToken": search_data.get("nextPageToken"),
                    "searchTotal": search_data.get("pageInfo", {}).get("totalResults"),
                    "message": f"Showing up to {size} results ordered by relevance.",
                }
            )

    except httpx.HTTPStatusError as e:
        return JSONResponse(
            status_code=502,
            content={"success": False, "message": f"YouTube API error: {e}"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Server error: {e}"},
        )
    
@app.get("/expert-preview", response_class=HTMLResponse)
def expert_preview(
    request: Request,
    file: Optional[str] = Query(None),
    video: Optional[str] = Query(None),
    mode: Optional[str] = Query("review")  # Add mode parameter
):
    question_files = list_question_json_files()
    selected_file_path: Optional[Path] = None
    selected_file_rel = None
    selection_error = None

    # Handle video selection for create mode
    if mode == "create" and video:
        # For create mode, we don't need existing question files
        # Just set up for video-based creation
        pass
    elif not file and video:
        for item in question_files:
            if item["video_id"] == video:
                file = item["rel_path"]
                break

    if file:
        candidate = resolve_question_file_param(file)
        if candidate and candidate.exists():
            selected_file_path = candidate
            selected_file_rel = candidate.relative_to(DOWNLOADS_DIR).as_posix()
        else:
            selection_error = "Selected question JSON could not be found."

    segments_info: List[Dict[str, Any]] = []
    segments_for_js: List[Dict[str, Any]] = []
    existing_annotations: List[Dict[str, Any]] = []
    existing_annotations_map: Dict[str, Any] = {}
    selected_json_pretty = None
    video_url = None
    annotation_rel_path = None
    selected_video_id = None
    selected_file_name = None

    if selected_file_path:
        selected_file_name = selected_file_path.name
        selected_video_dir = selected_file_path.parent.parent
        selected_video_id = selected_video_dir.name
        try:
            raw_data = json.loads(selected_file_path.read_text(encoding="utf-8"))
        except Exception:
            raw_data = {}
        segments_info = serialize_question_segments(raw_data)
        for segment in segments_info:
            parsed = segment.get("parsed")
            best_question = None
            questions_payload = None
            if isinstance(parsed, dict):
                questions_payload = parsed.get("questions")
                best_question = parsed.get("best_question")
            segments_for_js.append(
                {
                    "index": segment["index"],
                    "start": segment["start"], 
                    "end": segment["end"],
                    "questions": questions_payload,
                    "best_question": best_question,
                }
            )
        selected_json_pretty = json.dumps(raw_data, indent=2, ensure_ascii=False)

        video_candidate = find_primary_video_file(selected_video_dir)
        if video_candidate:
            video_url = f"/downloads/{video_candidate.relative_to(DOWNLOADS_DIR).as_posix()}"

        annotations_bundle = load_expert_annotations(selected_file_path, selected_video_id)
        annotations_data = annotations_bundle["data"]
        annotations_list = annotations_data.get("annotations", [])
        if isinstance(annotations_list, list):
            existing_annotations = annotations_list
            for entry in annotations_list:
                key = f"{entry.get('start')}-{entry.get('end')}"
                existing_annotations_map[key] = entry
        try:
            annotation_rel_path = annotations_bundle["path"].relative_to(DOWNLOADS_DIR).as_posix()
        except ValueError:
            annotation_rel_path = None

    # Handle create mode video selection
    elif mode == "create" and video:
        selected_video_id = video
        video_dir = DOWNLOADS_DIR / video
        if video_dir.exists():
            video_candidate = find_primary_video_file(video_dir)
            if video_candidate:
                video_url = f"/downloads/{video_candidate.relative_to(DOWNLOADS_DIR).as_posix()}"
            
            # Load existing expert annotations for create mode
            expert_questions_dir = video_dir / "expert_questions"
            expert_file = expert_questions_dir / f"expert_{video}.json"
            
            if expert_file.exists():
                try:
                    expert_data = json.loads(expert_file.read_text(encoding="utf-8"))
                    annotations_list = expert_data.get("annotations", [])
                    if isinstance(annotations_list, list):
                        existing_annotations = annotations_list
                        for entry in annotations_list:
                            key = f"{entry.get('start')}-{entry.get('end')}"
                            existing_annotations_map[key] = entry
                    try:
                        annotation_rel_path = expert_file.relative_to(DOWNLOADS_DIR).as_posix()
                    except ValueError:
                        annotation_rel_path = None
                except Exception:
                    pass

    context = {
        "request": request,
        "question_files": question_files,
        "selected_file_rel": selected_file_rel,
        "selected_file_name": selected_file_name,
        "selected_video_id": selected_video_id,
        "video_url": video_url,
        "segments": segments_info,
        "segments_for_js": segments_for_js,
        "existing_annotations": existing_annotations,
        "existing_annotations_map": existing_annotations_map,
        "selected_json_pretty": selected_json_pretty,
        "annotations_rel_path": annotation_rel_path,
        "selection_error": selection_error,
        "question_type_options": [
            {"value": value, "label": label} for value, label in EXPERT_QUESTION_TYPES
        ],
        "question_file_url": f"/downloads/{selected_file_rel}" if selected_file_rel else None,
        "mode": mode,
    }
    return templates.TemplateResponse("expert_preview.html", context)

@app.post("/api/expert-annotations")
async def save_expert_annotation(payload: Dict[str, Any] = Body(...)):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload.")

    mode = payload.get("mode", "review")  # "create" or "review"
    
    if mode == "create":
        # Handle Create mode - save to expert_video_id file
        video_id = payload.get("video_id")
        if not video_id:
            raise HTTPException(status_code=400, detail="Missing video_id for create mode.")
            
        video_dir = DOWNLOADS_DIR / video_id
        if not video_dir.exists():
            raise HTTPException(status_code=400, detail="Video directory not found.")
            
        # Create expert questions file
        expert_questions_dir = video_dir / "expert_questions"
        expert_questions_dir.mkdir(exist_ok=True)
        
        expert_file = expert_questions_dir / f"expert_{video_id}.json"
        
        # Load existing expert data
        if expert_file.exists():
            try:
                expert_data = json.loads(expert_file.read_text(encoding="utf-8"))
            except Exception:
                expert_data = {"video_id": video_id, "mode": "create", "annotations": []}
        else:
            expert_data = {"video_id": video_id, "mode": "create", "annotations": []}
            
        annotations_list = expert_data.setdefault("annotations", [])
        
    else:
        # Handle Review mode - existing logic
        question_file = resolve_question_file_param(payload.get("file"))
        if not question_file or not question_file.exists():
            raise HTTPException(status_code=400, detail="Invalid question file.")
            
        video_dir = question_file.parent.parent
        video_id = video_dir.name
        
        annotations_bundle = load_expert_annotations(question_file, video_id)
        expert_data = annotations_bundle["data"]
        expert_data["video_id"] = video_id
        expert_data["question_file"] = question_file.name
        annotations_list = expert_data.setdefault("annotations", [])
        expert_file = annotations_bundle["path"]

    # Common processing for both modes
    try:
        start = int(payload.get("start"))
        end = int(payload.get("end"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid segment bounds.")

    skip_requested = bool(payload.get("skip"))
    segment_index = payload.get("segment_index")
    try:
        segment_index = int(segment_index) if segment_index is not None else None
    except (TypeError, ValueError):
        segment_index = None

    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # Create annotation entry
    if skip_requested:
        entry = {
            "segment_index": segment_index,
            "start": start,
            "end": end,
            "question_type": "skip",
            "question_type_label": "Skipped", 
            "question": "(skipped)",
            "answer": "",
            "skipped": True,
            "saved_at": timestamp,
            "mode": mode
        }
    else:
        question = (payload.get("question") or "").strip()
        answer = (payload.get("answer") or "").strip()
        question_type_raw = (payload.get("question_type") or "").strip().lower()
        
        if not question or not answer:
            raise HTTPException(status_code=400, detail="Question and answer are required.")
        if question_type_raw not in EXPERT_QUESTION_TYPE_VALUES:
            raise HTTPException(status_code=400, detail="Invalid question type.")

        entry = {
            "segment_index": segment_index,
            "start": start,
            "end": end,
            "question_type": question_type_raw,
            "question_type_label": EXPERT_QUESTION_TYPE_LABELS.get(
                question_type_raw, question_type_raw.title()
            ),
            "question": question,
            "answer": answer,
            "skipped": False,
            "saved_at": timestamp,
            "mode": mode
        }

        # Handle best question for review mode only
        if mode == "review":
            best_question_payload = payload.get("best_question")
            if isinstance(best_question_payload, dict):
                best_question_question = (best_question_payload.get("question") or "").strip()
                best_question_answer = (best_question_payload.get("answer") or "").strip()
                approved_raw = best_question_payload.get("approved")
                
                if isinstance(approved_raw, bool):
                    approved_value = approved_raw
                elif isinstance(approved_raw, str):
                    approved_value = approved_raw.lower() in {"true", "1", "yes", "approved"}
                else:
                    approved_value = None
                    
                comment_text = (best_question_payload.get("comment") or "").strip()
                
                if approved_value is False and not comment_text:
                    raise HTTPException(status_code=400, detail="Provide a comment when disapproving the best question.")
                    
                if any([best_question_question, best_question_answer, approved_value is not None, comment_text]):
                    if approved_value is None:
                        approved_value = True
                        
                    entry["best_question"] = {
                        "question": best_question_question,
                        "answer": best_question_answer,
                        "approved": approved_value,
                        "comment": comment_text if not approved_value else "",
                    }

    # Update annotations list
    updated = False
    for idx, existing in enumerate(list(annotations_list)):
        if isinstance(existing, dict) and existing.get("start") == start and existing.get("end") == end:
            if not skip_requested and mode == "review" and entry.get("best_question") is None and existing.get("best_question") is not None:
                entry["best_question"] = existing.get("best_question")
            annotations_list[idx] = entry
            updated = True
            break
            
    if not updated:
        annotations_list.append(entry)

    annotations_list.sort(key=lambda item: (item.get("start", 0), item.get("end", 0)))

    # Save the file
    expert_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        expert_file.write_text(
            json.dumps(expert_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store annotation: {exc}")

    try:
        annotation_rel = expert_file.relative_to(DOWNLOADS_DIR).as_posix()
    except ValueError:
        annotation_rel = None


    return JSONResponse({
        "success": True,
        "annotation": entry,
        "annotations_file": annotation_rel,
        "updated": updated,
        "mode": mode
    })
# -----------------------------
# WebSocket endpoint for streaming interval results
# -----------------------------
@app.websocket("/ws/questions/{video_id}")
async def ws_questions(websocket: WebSocket, video_id: str):
    await websocket.accept()
    try:
        # Expect one JSON message with params
        params = await websocket.receive_json()
        start_seconds = int(params.get("start_seconds", 0))
        interval_seconds = int(params.get("interval_seconds", 60))
        full_duration = bool(params.get("full_duration", False))

        frames_dir = DOWNLOADS_DIR / video_id / "extracted_frames"
        if not frames_dir.exists():
            await websocket.send_json({"type": "error", "message": "Frames not found. Please extract frames first."})
            await websocket.close()
            return

        # Load duration (for full-duration mode / clamping)
        duration_seconds = None
        json_path = frames_dir / "frame_data.json"
        if json_path.exists():
            try:
                info = json.loads(json_path.read_text(encoding="utf-8"))
                duration_seconds = int(float(info.get("video_info", {}).get("duration_seconds", 0)))
            except Exception:
                duration_seconds = None

        # SINGLE INTERVAL (non-loop)
        if not full_duration:
            start = max(0, int(start_seconds))
            end = start + max(1, int(interval_seconds)) - 1
            if duration_seconds is not None and end > duration_seconds:
                end = duration_seconds

            await websocket.send_json({"type": "status", "message": f"Generating questions for {start}-{end}s..."})
            result_text = await asyncio.to_thread(generate_questions_for_segment, video_id, start, end)
            result_obj = _maybe_parse_json(result_text)
            
            # Don't auto-save for single intervals
            await websocket.send_json({
                "type": "segment_result",
                "start": start,
                "end": end,
                "result": result_obj
            })
            await websocket.send_json({"type": "done", "auto_saved": False})
            await websocket.close()
            return

        # FULL DURATION LOOP
        if duration_seconds is None or duration_seconds <= 0:
            await websocket.send_json({"type": "error", "message": "Unable to determine video duration."})
            await websocket.close()
            return

        segments = build_segments_from_duration(duration_seconds, interval_seconds, start_seconds)
        await websocket.send_json({"type": "status", "message": f"Starting full-duration generation over {len(segments)} segments."})

        aggregated = {
            "video_id": video_id,
            "interval_seconds": int(interval_seconds),
            "start_offset": int(start_seconds),
            "duration_seconds": duration_seconds,
            "segments": []
        }

        for idx, (seg_start, seg_end) in enumerate(segments, start=1):
            await websocket.send_json({"type": "status", "message": f"[{idx}/{len(segments)}] {seg_start}-{seg_end}s"})
            result_text = await asyncio.to_thread(generate_questions_for_segment, video_id, seg_start, seg_end)
            result_obj = _maybe_parse_json(result_text)
            aggregated["segments"].append({
                "start": seg_start,
                "end": seg_end,
                "result": result_obj
            })
            await websocket.send_json({
                "type": "segment_result",
                "start": seg_start,
                "end": seg_end,
                "result": result_obj
            })

        # Don't auto-save - send data for manual submission
        await websocket.send_json({
            "type": "done",
            "segments_count": len(segments),
            "auto_saved": False,
            "data": aggregated
        })
        await websocket.close()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close()
        except Exception:
            pass
def _maybe_parse_json(text: Optional[str]):
    if text is None:
        return None
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return text
    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = cleaned[3:].lstrip()
        if cleaned.lower().startswith('json'):
            cleaned = cleaned[4:].lstrip()
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3].rstrip()
    try:
        return json.loads(cleaned)
    except Exception:
        return text  # return raw text if not valid JSON


def persist_segment_questions_json(video_id: str, start: int, end: int, payload: Any) -> Optional[str]:
    """Persist a single segment's questions JSON to disk and return a downloads URL."""
    if payload is None:
        return None

    if isinstance(payload, (dict, list)):
        data = payload
    elif isinstance(payload, str):
        try:
            data = json.loads(payload)
        except Exception:
            return None
    else:
        return None

    try:
        start_int = int(start)
    except Exception:
        start_int = None
    try:
        end_int = int(end)
    except Exception:
        end_int = None

    if start_int is not None and end_int is not None:
        filename = f"questions_{start_int:05d}-{end_int:05d}.json"
    else:
        filename = f"questions_{start}-{end}.json"

    questions_dir = DOWNLOADS_DIR / video_id / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)
    out_path = questions_dir / filename

    try:
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        print(f"Failed to write questions JSON for {video_id} {start}-{end}: {exc}")
        return None

    return f"/downloads/{out_path.relative_to(DOWNLOADS_DIR).as_posix()}"


def resolve_question_file_param(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    cleaned = value.strip()
    if cleaned.startswith('/'):
        cleaned = cleaned.lstrip('/')
    if cleaned.startswith('downloads/'):
        cleaned = cleaned[len('downloads/'):]
    rel_path = Path(cleaned)
    if rel_path.is_absolute() or '..' in rel_path.parts:
        return None
    candidate = DOWNLOADS_DIR / rel_path
    if candidate.is_file() and candidate.suffix.lower() == '.json':
        try:
            candidate.relative_to(DOWNLOADS_DIR)
        except ValueError:
            return None
        return candidate
    return None


def list_question_json_files() -> List[Dict[str, str]]:
    files: List[Dict[str, str]] = []
    if not DOWNLOADS_DIR.exists():
        return files
    for video_dir in sorted(DOWNLOADS_DIR.iterdir()):
        if not video_dir.is_dir():
            continue
        questions_dir = video_dir / 'questions'
        if not questions_dir.is_dir():
            continue
        for json_file in sorted(questions_dir.glob('*.json')):
            try:
                rel_path = json_file.relative_to(DOWNLOADS_DIR).as_posix()
            except ValueError:
                continue
            files.append({
                'video_id': video_dir.name,
                'name': json_file.name,
                'rel_path': rel_path,
            })
    files.sort(key=lambda item: (item['video_id'], item['name']))
    return files


def find_primary_video_file(video_dir: Path) -> Optional[Path]:
    if not video_dir.exists() or not video_dir.is_dir():
        return None
    for candidate in sorted(video_dir.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
            return candidate
    return None


def load_expert_annotations(question_file: Path, video_id: str) -> Dict[str, Any]:
    annotations_path = question_file.with_suffix(question_file.suffix + '.expert.json')
    payload: Dict[str, Any] = {
        'video_id': video_id,
        'question_file': question_file.name,
        'annotations': [],
    }
    if annotations_path.exists():
        try:
            loaded = json.loads(annotations_path.read_text(encoding='utf-8'))
            if isinstance(loaded, dict):
                payload.update({
                    'annotations': loaded.get('annotations', []),
                })
        except Exception:
            pass
    return {
        'path': annotations_path,
        'data': payload,
    }


def serialize_question_segments(question_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for idx, seg in enumerate(question_data.get('segments', [])):
        start = int(seg.get('start', 0))
        end = int(seg.get('end', start))
        result_raw = seg.get('result')
        parsed = _maybe_parse_json(result_raw)
        if isinstance(parsed, dict):
            display_payload = json.dumps(parsed, indent=2, ensure_ascii=False)
            parsed_for_js = parsed
        elif isinstance(parsed, list):
            display_payload = json.dumps(parsed, indent=2, ensure_ascii=False)
            parsed_for_js = parsed
        else:
            display_payload = result_raw if isinstance(result_raw, str) else json.dumps(result_raw, indent=2, ensure_ascii=False)
            parsed_for_js = None
        segments.append({
            'index': idx,
            'start': start,
            'end': end,
            'parsed': parsed_for_js,
            'display': display_payload,
        })
    return segments


def build_expert_preview_link(video_id: Optional[str], file_path: Optional[str]) -> str:
    if file_path:
        cleaned = file_path.lstrip('/')
        if cleaned.startswith('downloads/'):
            cleaned = cleaned[len('downloads/'):]
        return f"/expert-preview?file={quote(cleaned)}"
    if video_id:
        return f"/expert-preview?video={quote(str(video_id))}"
    return "/expert-preview"


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    # Get all downloads for the admin interface
    all_downloads = list_all_downloads()
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "all_downloads": all_downloads
    })

@app.post("/api/download")
async def api_download(url: str = Form(...)):
    outcome = download_youtube(url)
    return JSONResponse(content=outcome)

@app.post("/api/frames/{video_id}")
async def api_extract_frames(video_id: str):
    outcome = extract_frames_per_second_for_video(video_id)
    return JSONResponse(content=outcome)


@app.post("/api/submit-questions")
async def submit_questions(payload: Dict[str, Any] = Body(...)):
    """Submit and save finalized questions"""
    video_id = payload.get("video_id")
    questions_data = payload.get("questions", [])
    
    if not video_id or not questions_data:
        raise HTTPException(status_code=400, detail="Missing video_id or questions")
    
    # Create the aggregated structure
    aggregated = {
        "video_id": video_id,
        "submitted_at": datetime.utcnow().isoformat(),
        "status": "submitted",
        "segments": questions_data
    }
    
    # Save to the questions directory
    questions_dir = DOWNLOADS_DIR / video_id / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = questions_dir / f"{video_id}.json"
    
    try:
        out_path.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8")
        output_url = f"/downloads/{out_path.relative_to(DOWNLOADS_DIR).as_posix()}"
        
        return JSONResponse({
            "success": True,
            "message": "Questions submitted successfully",
            "file_url": output_url,
            "file_path": str(out_path)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save: {e}")
    

@app.get("/api/kids_videos")
def list_kids_videos():
    """Return JSON of all locally available kids videos"""
    videos = []
    if not DOWNLOADS_DIR.exists():
        return {"success": True, "count": 0, "videos": videos}

    for item in sorted(DOWNLOADS_DIR.iterdir()):
        if not item.is_dir():
            continue

        vid = item.name

        # Find video file
        video_files = list(item.glob("*.mp4")) + list(item.glob("*.webm")) + list(item.glob("*.mkv"))
        if not video_files:
            continue
        video_file = video_files[0]

        # Get title from metadata if available
        title = vid
        info_json = next(item.glob("*.info.json"), None)
        if info_json:
            try:
                info = json.loads(info_json.read_text(encoding="utf-8"))
                title = info.get("title", vid)
            except Exception:
                pass

        # Find thumbnail
        thumb_url = None
        for ext in ("jpg", "png", "webp"):
            thumb_file = next(item.glob(f"*.{ext}"), None)
            if thumb_file:
                thumb_url = f"/downloads/{vid}/{thumb_file.name}"
                break
        
        # Get duration from frame data if available
        duration = None
        duration_sec = None
        frame_json = item / "extracted_frames" / "frame_data.json"
        if frame_json.exists():
            try:
                info = json.loads(frame_json.read_text(encoding="utf-8"))
                duration_sec = int(float(info["video_info"]["duration_seconds"]))
                m, s = divmod(duration_sec, 60)
                duration = f"{m:02d}:{s:02d}"
            except Exception:
                pass

        videos.append({
            "video_id": vid,
            "title": title,
            "duration": duration,
            "duration_seconds": duration_sec,
            "local_path": f"/downloads/{vid}/{video_file.name}",
            "thumbnail": thumb_url or "/static/default-thumbnail.png",
        })

    return {"success": True, "count": len(videos), "videos": videos}



from rapidfuzz import fuzz
import re

# Add these helper functions
NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

SCALE_WORDS = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
STOPWORDS = {"the", "a", "an", "is", "are", "and", "of", "to", "it"}

def words_to_numbers(text: str) -> list[int]:
    """Extract ALL numbers (digits or words) from a phrase."""
    text = text.lower().strip()
    numbers = []

    # Digits
    for d in re.findall(r"\d+", text):
        numbers.append(int(d))

    # Words
    tokens = re.split(r"[-\s]+", text)
    total, current = 0, 0
    found_number = False

    for token in tokens + ["end"]:  # sentinel flush
        if token in NUM_WORDS:
            found_number = True
            current += NUM_WORDS[token]
        elif token in SCALE_WORDS:
            found_number = True
            scale = SCALE_WORDS[token]
            if current == 0:
                current = 1
            current *= scale
            if scale > 100:  # thousand/million
                total += current
                current = 0
        else:
            if found_number:
                total += current
                numbers.append(total)
                total, current, found_number = 0, 0, False

    return numbers

def clean_text(text: str) -> str:
    """Remove stopwords and punctuation for fuzzy matching."""
    return " ".join(
        w for w in re.findall(r"[a-z]+", text.lower())
        if w not in STOPWORDS and w not in NUM_WORDS and w not in SCALE_WORDS
    )

@app.post("/api/check_answer")
async def check_answer(payload: dict = Body(...)):
    expected = payload.get("expected", "").strip().lower()
    user = payload.get("user", "").strip().lower()

    if not expected or not user:
        return {
            "similarity": 0.0,
            "expected": expected,
            "user": user,
            "is_numeric": False,
        }

    # Extract numbers
    exp_nums = words_to_numbers(expected)
    usr_nums = words_to_numbers(user)

    # If expected has numbers, enforce strict equality
    if exp_nums:
        if sorted(exp_nums) != sorted(usr_nums):
            return {
                "similarity": 0.0,
                "expected": expected,
                "user": user,
                "is_numeric": True,
                "reason": f"Numbers mismatch (expected {exp_nums}, got {usr_nums})",
            }

        # Numbers match → check words
        exp_clean = clean_text(expected)
        usr_clean = clean_text(user)

        if not exp_clean or not usr_clean:
            return {
                "similarity": 1.0,
                "expected": expected,
                "user": user,
                "is_numeric": True,
            }

        pr = fuzz.partial_ratio(exp_clean, usr_clean) / 100.0
        tsr = fuzz.token_set_ratio(exp_clean, usr_clean) / 100.0
        ratio = max(pr, tsr)

        return {
            "similarity": round(ratio, 3),
            "expected": expected,
            "user": user,
            "is_numeric": True,
        }

    # No numbers: plain fuzzy check
    pr = fuzz.partial_ratio(expected, user) / 100.0
    tsr = fuzz.token_set_ratio(expected, user) / 100.0
    ratio = max(pr, tsr)

    return {
        "similarity": round(ratio, 3),
        "expected": expected,
        "user": user,
        "is_numeric": False,
    }

@app.get("/api/videos-list")
async def list_videos():
    """List all downloaded videos available for expert question creation"""
    try:
        videos = []
        if not DOWNLOADS_DIR.exists():
            return JSONResponse({"success": True, "videos": []})
        
        for video_dir in sorted(DOWNLOADS_DIR.iterdir()):
            if not video_dir.is_dir():
                continue

            video_id = video_dir.name

            # Find video file
            video_file = None
            for ext in ("*.mp4", "*.webm", "*.mkv"):
                video_files = list(video_dir.glob(ext))
                if video_files:
                    video_file = video_files[0]
                    break

            if not video_file:
                continue

            questions_dir = video_dir / "questions"
            if not questions_dir.exists() or not questions_dir.is_dir():
                continue

            question_files = [p for p in questions_dir.glob('*.json') if p.is_file()]
            if not question_files:
                continue

            question_count = len(question_files)

            # Get video duration from frame data if available
            duration = 0
            frames_dir = video_dir / "extracted_frames"
            json_path = frames_dir / "frame_data.json"
            if json_path.exists():
                try:
                    info = json.loads(json_path.read_text(encoding="utf-8"))
                    duration = int(float(info.get("video_info", {}).get("duration_seconds", 0)))
                except Exception:
                    duration = 0

            # Count files
            file_count = len([f for f in video_dir.iterdir() if f.is_file()])

            # Create video URL
            video_url = f"/downloads/{video_file.relative_to(DOWNLOADS_DIR).as_posix()}"

            videos.append({
                "id": video_id,
                "title": video_id,  # Could be enhanced to get actual title
                "duration": duration,
                "fileCount": file_count,
                "questionCount": question_count,
                "videoUrl": video_url
            })

        return JSONResponse({
            "success": True,
            "videos": videos
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error listing videos: {e}",
            "videos": []
        })
    

@app.get("/api/expert-questions/{video_id}")
async def get_expert_questions(video_id: str):
    video_dir = DOWNLOADS_DIR / video_id
    questions_dir = video_dir / "expert_questions"
    file_path = questions_dir / "expert_questions.json"

    if not video_dir.exists() or not questions_dir.exists() or not file_path.exists():
        return JSONResponse({"success": True, "video_id": video_id, "questions": []})

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return JSONResponse({"success": False, "message": f"Unable to read expert questions: {exc}", "questions": []}, status_code=500)

    questions = data.get("questions") if isinstance(data, dict) else []
    if not isinstance(questions, list):
        questions = []

    return JSONResponse({"success": True, "video_id": video_id, "questions": questions})


@app.post("/api/expert-questions")
async def save_expert_question(payload: Dict[str, Any] = Body(...)):
    video_id = str(payload.get("videoId") or payload.get("video_id") or "").strip()
    if not video_id:
        return JSONResponse({"success": False, "message": "videoId is required"}, status_code=400)

    video_dir = DOWNLOADS_DIR / video_id
    if not video_dir.exists():
        return JSONResponse({"success": False, "message": "Video not found"}, status_code=404)

    segment_start_value = normalize_segment_value(payload.get("segmentStart"))
    segment_end_value = normalize_segment_value(payload.get("segmentEnd"))
    timestamp_value = normalize_segment_value(payload.get("timestamp", segment_end_value))

    skipped = bool(payload.get("skipped") or payload.get("skip") or payload.get("isSkipped"))
    skip_reason = str(payload.get("skipReason") or payload.get("skip_reason") or "").strip()

    if segment_end_value <= segment_start_value:
        segment_end_value = segment_start_value

    question_type = str(payload.get("questionType") or payload.get("question_type") or "").strip().lower()
    question_text = str(payload.get("question") or "").strip()
    answer_text = str(payload.get("answer") or "").strip()

    if skipped:
        question_type = ""
        question_text = ""
        answer_text = ""
    else:
        if question_type not in EXPERT_QUESTION_TYPE_VALUES:
            return JSONResponse({"success": False, "message": "Invalid question type"}, status_code=400)

        if not question_text or not answer_text:
            return JSONResponse({"success": False, "message": "Question and answer are required"}, status_code=400)

    questions_dir = video_dir / "expert_questions"
    questions_dir.mkdir(parents=True, exist_ok=True)
    file_path = questions_dir / "expert_questions.json"

    try:
        stored = json.loads(file_path.read_text(encoding="utf-8")) if file_path.exists() else {}
    except Exception:
        stored = {}

    if not isinstance(stored, dict):
        stored = {}

    questions_list = stored.get("questions")
    if not isinstance(questions_list, list):
        questions_list = []

    def matches_existing(entry: Dict[str, Any]) -> bool:
        existing_start = normalize_segment_value(entry.get("segment_start"))
        existing_end = normalize_segment_value(entry.get("segment_end"))
        return abs(existing_start - segment_start_value) < 1e-3 and abs(existing_end - segment_end_value) < 1e-3

    questions_list = [q for q in questions_list if not matches_existing(q)]

    entry = {
        "segment_start": segment_start_value,
        "segment_end": segment_end_value,
        "timestamp": timestamp_value,
        "question_type": question_type if not skipped else None,
        "question": question_text,
        "answer": answer_text,
        "skipped": skipped,
        "skip_reason": skip_reason,
        "updated_at": datetime.utcnow().isoformat()
    }

    questions_list.append(entry)
    questions_list.sort(key=lambda q: normalize_segment_value(q.get("segment_start")))

    stored["video_id"] = video_id
    stored["questions"] = questions_list

    file_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")

    message = "Segment marked as skipped." if skipped else "Expert question saved."
    return JSONResponse({"success": True, "message": message, "updatedAt": entry["updated_at"], "skipped": skipped})

@app.post("/api/save-final-questions")
async def save_final_questions(payload: Dict[str, Any] = Body(...)):
    """Save final reviewed questions to a dedicated folder"""
    video_id = str(payload.get("videoId") or "").strip()
    if not video_id:
        return JSONResponse({"success": False, "message": "videoId is required"}, status_code=400)
    
    video_dir = DOWNLOADS_DIR / video_id
    if not video_dir.exists():
        return JSONResponse({"success": False, "message": "Video not found"}, status_code=404)
    
    # Get the final questions data
    final_data = payload.get("data")
    if not final_data:
        return JSONResponse({"success": False, "message": "No data provided"}, status_code=400)
    
    # Create final_questions directory
    final_questions_dir = video_dir / "final_questions"
    final_questions_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to final_questions.json
    final_file_path = final_questions_dir / "final_questions.json"
    
    try:
        # Add metadata
        final_data["saved_at"] = datetime.utcnow().isoformat()
        final_data["video_id"] = video_id
        
        # Write the file
        final_file_path.write_text(json.dumps(final_data, indent=2), encoding="utf-8")
        
        return JSONResponse({
            "success": True,
            "message": "Final questions saved successfully",
            "filepath": f"downloads/{video_id}/final_questions/final_questions.json",
            "saved_at": final_data["saved_at"]
        })
        
    except Exception as exc:
        return JSONResponse(
            {"success": False, "message": f"Failed to save final questions: {exc}"}, 
            status_code=500
        )

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts audio (webm, wav, mp3, etc), sends to Whisper,
    no temp file saved (in-memory BytesIO).
    """
    try:
        # Read file into memory
        contents = await file.read()
        audio_bytes = io.BytesIO(contents)

        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Send to Whisper
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=("speech.webm", audio_bytes, file.content_type)
        )

        return {"success": True, "text": transcription.text}

    except Exception as e:
        print("❌ Whisper transcription error:", e)
        return {"success": False, "error": str(e)}
    

app.mount("/static", StaticFiles(directory="static"), name="static")
