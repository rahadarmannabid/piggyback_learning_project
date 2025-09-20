# main.py
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import base64
import io
import json
import asyncio
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


def generate_questions_for_segment(video_id: str, start_time: int, end_time: int, api_key: Optional[str] = None) -> Optional[str]:
    """
    Analyze frames + transcript for a time window and return JSON text with the questions.
    """
    folder_name = str(DOWNLOADS_DIR / video_id)
    try:
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception as e:
        print(f"Error creating OpenAI client: {e}")
        return None

    frame_data, complete_transcript = read_frame_data_from_csv(folder_name, start_time, end_time)
    if not frame_data:
        return None

    duration = end_time - start_time + 1  # inclusive window
    prompt = f"""You are an early childhood educator designing comprehension questions for children ages 4–7. 
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

2. After creating the questions, choose the best single question you think is most appropriate for comprehension.

3. Return JSON only (no extra text) in this structure:
{{
  "questions": {{
    "character": {{ "q": "...", "a": "..." }},
    "setting": {{ "q": "...", "a": "..." }},
    "feeling": {{ "q": "...", "a": "..." }},
    "action": {{ "q": "...", "a": "..." }},
    "causal": {{ "q": "...", "a": "..." }},
    "outcome": {{ "q": "...", "a": "..." }},
    "prediction": {{ "q": "...", "a": "..." }}
  }},
  "best_question": "..."
}}
"""

    content = [{"type": "text", "text": prompt}]

    # Add frames as low-detail inline images
    successful_frames = 0
    for fr in frame_data:
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

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            max_tokens=1500,
            temperature=0.3
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
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


def generate_questions_for_full_duration(
    video_id: str,
    interval_seconds: int,
    api_key: Optional[str] = None,
    start_offset: int = 0
) -> Dict[str, Any]:
    """
    Non-streaming aggregator (kept for HTTP POST flow or saving to disk).
    """
    frames_dir = DOWNLOADS_DIR / video_id / "extracted_frames"
    if not frames_dir.exists():
        return {"success": False, "message": "Frames not found. Please extract frames first."}

    # Read duration
    duration_seconds = 0
    json_path = frames_dir / "frame_data.json"
    if json_path.exists():
        try:
            info = json.loads(json_path.read_text(encoding="utf-8"))
            duration_seconds = int(float(info.get("video_info", {}).get("duration_seconds", 0)))
        except Exception:
            duration_seconds = 0

    if duration_seconds <= 0:
        return {"success": False, "message": "Unable to determine video duration."}

    segments = build_segments_from_duration(duration_seconds, interval_seconds, start_offset)

    aggregated: Dict[str, Any] = {
        "video_id": video_id,
        "interval_seconds": int(interval_seconds),
        "start_offset": int(start_offset),
        "duration_seconds": duration_seconds,
        "segments": []
    }

    for (seg_start, seg_end) in segments:
        result_text = generate_questions_for_segment(video_id, seg_start, seg_end, api_key)
        parsed = None
        if result_text:
            try:
                parsed = json.loads(result_text)
            except Exception:
                parsed = None
        aggregated["segments"].append({
            "start": seg_start,
            "end": seg_end,
            "result": parsed if parsed is not None else result_text
        })

    # Save aggregated file
    questions_dir = DOWNLOADS_DIR / video_id / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)
    out_path = questions_dir / f"questions_interval_{int(interval_seconds)}s.json"
    out_path.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "success": True,
        "message": f"Generated {len(segments)} segments of questions.",
        "segments_count": len(segments),
        "output_json": f"/downloads/{out_path.relative_to(DOWNLOADS_DIR).as_posix()}",
        "aggregated": aggregated
    }


# -----------------------------
# Routes (HTML pages)
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("download.html", {"request": request})


@app.post("/download", response_class=HTMLResponse)
def do_download(request: Request, url: str = Form(...)):
    outcome = download_youtube(url)

    # Links for the current download
    links = [f"/downloads/{rel}" for rel in outcome.get("files", [])]

    # Build list of all downloads
    all_downloads = list_all_downloads()

    # Find the current video's playable URL + subtitle URL
    vid = outcome.get("video_id")
    media = find_current_video_and_sub(vid)

    return templates.TemplateResponse(
        "preview.html",
        {
            "request": request,
            "success": outcome["success"],
            "message": outcome["message"],
            "video_id": vid,
            "files": links,
            "all_downloads": all_downloads,
            "current_video_url": media["video"],
            "current_sub_url": media["sub"],
        },
    )


# -----------------------------
# YouTube Search API (child-safe with duration filters)
# -----------------------------
@app.get("/api/yt_search")
async def yt_search(
    request: Request,
    q: str,
    min_minutes: int = 10,
    max_minutes: int = 60,
    max_results: int = 25,
    page_token: Optional[str] = None,
    page_size: Optional[int] = 20,
):
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

                # Child-safe filter: require madeForKids True and not age-restricted
                made_for_kids = bool(status.get("madeForKids", False))
                if not made_for_kids:
                    continue

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


@app.get("/frames/{video_id}", response_class=HTMLResponse)
def frames_page(request: Request, video_id: str):
    return templates.TemplateResponse(
        "frames.html",
        {
            "request": request,
            "video_id": video_id,
            "ran": False,
            "success": None,
            "message": None,
            "count": 0,
            "output_dir": None,
            "files": [],
        },
    )


@app.post("/frames/{video_id}", response_class=HTMLResponse)
def run_extraction(request: Request, video_id: str):
    outcome = extract_frames_per_second_for_video(video_id)
    return templates.TemplateResponse(
        "frames.html",
        {
            "request": request,
            "video_id": video_id,
            "ran": True,
            "success": outcome.get("success", False),
            "message": outcome.get("message", ""),
            "count": outcome.get("count", 0),
            "output_dir": outcome.get("output_dir"),
            "files": outcome.get("files", []),
        },
    )


@app.get("/questions/{video_id}", response_class=HTMLResponse)
def questions_page(request: Request, video_id: str):
    """
    Show a form to pick interval and (optionally) stream full-duration results via WebSocket.
    """
    # Try to read duration (from frame_data.json), otherwise leave blank
    frames_dir = DOWNLOADS_DIR / video_id / "extracted_frames"
    duration_seconds = None
    json_path = frames_dir / "frame_data.json"
    if json_path.exists():
        try:
            info = json.loads(json_path.read_text(encoding="utf-8"))
            duration_seconds = int(float(info.get("video_info", {}).get("duration_seconds", 0)))
        except Exception:
            duration_seconds = None

    return templates.TemplateResponse(
        "questions.html",
        {
            "request": request,
            "video_id": video_id,
            "duration_seconds": duration_seconds,
            "result": None,
            "full_result": None,
            "output_json": None,
            "start_seconds": None,
            "interval_seconds": None,
            "full_duration": False,
            "expert_preview_file": None,
            "expert_preview_link": build_expert_preview_link(video_id, None),
            "error": None,
        },
    )


@app.post("/questions/{video_id}", response_class=HTMLResponse)
def generate_questions_http(
    request: Request,
    video_id: str,
    start_seconds: int = Form(0),
    interval_seconds: int = Form(...),
    full_duration: Optional[str] = Form(None),   # checkbox: "on" if checked
    api_key: Optional[str] = Form(None),
):
    """
    Fallback non-streaming HTTP endpoint (kept for compatibility).
    """
    frames_dir = DOWNLOADS_DIR / video_id / "extracted_frames"
    if not frames_dir.exists():
        return templates.TemplateResponse(
            "questions.html",
            {
                "request": request,
                "video_id": video_id,
                "duration_seconds": None,
                "result": None,
                "full_result": None,
                "output_json": None,
                "start_seconds": start_seconds,
                "interval_seconds": interval_seconds,
                "full_duration": bool(full_duration),
                "expert_preview_file": None,
                "expert_preview_link": build_expert_preview_link(video_id, None),
                "error": "Frames not found. Please extract frames first.",
            },
        )

    # Try to read duration for display/clamping
    duration_seconds = None
    json_path = frames_dir / "frame_data.json"
    if json_path.exists():
        try:
            info = json.loads(json_path.read_text(encoding="utf-8"))
            duration_seconds = int(float(info.get("video_info", {}).get("duration_seconds", 0)))
        except Exception:
            pass

    # FULL DURATION MODE
    if full_duration:
        full_outcome = generate_questions_for_full_duration(
            video_id=video_id,
            interval_seconds=int(interval_seconds),
            api_key=api_key,
            start_offset=int(start_seconds or 0),
        )
        return templates.TemplateResponse(
            "questions.html",
            {
                "request": request,
                "video_id": video_id,
                "duration_seconds": duration_seconds,
                "result": None,
                "full_result": json.dumps(full_outcome.get("aggregated"), indent=2, ensure_ascii=False) if full_outcome.get("aggregated") else None,
                "output_json": full_outcome.get("output_json"),
                "expert_preview_file": full_outcome.get("output_json"),
                "expert_preview_link": build_expert_preview_link(video_id, full_outcome.get("output_json")),
                "start_seconds": int(start_seconds or 0),
                "interval_seconds": int(interval_seconds),
                "full_duration": True,
                "error": None if full_outcome.get("success") else (full_outcome.get("message") or "Failed to generate questions for full duration."),
            },
        )

    # SINGLE WINDOW MODE (inclusive end like the examples)
    start = max(0, int(start_seconds or 0))
    end = start + max(1, int(interval_seconds)) - 1
    if duration_seconds and end > duration_seconds:
        end = duration_seconds

    result_text = generate_questions_for_segment(video_id, start, end, api_key)
    pretty_result = result_text
    saved_json_url = None
    if result_text:
        try:
            parsed_result = json.loads(result_text)
        except Exception:
            parsed_result = None
        if parsed_result is not None:
            pretty_result = json.dumps(parsed_result, indent=2, ensure_ascii=False)
            saved_json_url = persist_segment_questions_json(video_id, start, end, parsed_result)
    return templates.TemplateResponse(
        "questions.html",
        {
            "request": request,
            "video_id": video_id,
            "duration_seconds": duration_seconds,
            "result": pretty_result,
            "full_result": None,
            "output_json": saved_json_url,
            "expert_preview_file": saved_json_url,
            "expert_preview_link": build_expert_preview_link(video_id, saved_json_url),
            "start_seconds": start,
            "interval_seconds": int(interval_seconds),
            "full_duration": False,
            "error": None if result_text else "Failed to generate questions. Verify API key and time window.",
        },
    )


@app.get("/expert-preview", response_class=HTMLResponse)
def expert_preview(
    request: Request,
    file: Optional[str] = Query(None),
    video: Optional[str] = Query(None),
):
    question_files = list_question_json_files()
    selected_file_path: Optional[Path] = None
    selected_file_rel = None
    selection_error = None

    if not file and video:
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
    }
    return templates.TemplateResponse("expert_preview.html", context)


@app.post("/api/expert-annotations")
async def save_expert_annotation(payload: Dict[str, Any] = Body(...)):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload.")

    question_file = resolve_question_file_param(payload.get("file"))
    if not question_file or not question_file.exists():
        raise HTTPException(status_code=400, detail="Invalid question file.")

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

    video_dir = question_file.parent.parent
    video_id = video_dir.name

    annotations_bundle = load_expert_annotations(question_file, video_id)
    annotations_data = annotations_bundle["data"]
    annotations_data["video_id"] = video_id
    annotations_data["question_file"] = question_file.name
    annotations_list = annotations_data.setdefault("annotations", [])

    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    best_question_data = None
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
        }

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
                best_question_data = {
                    "question": best_question_question,
                    "answer": best_question_answer,
                    "approved": approved_value,
                    "comment": comment_text if not approved_value else "",
                }
        if best_question_data is not None:
            entry["best_question"] = best_question_data

    updated = False
    for idx, existing in enumerate(list(annotations_list)):
        if isinstance(existing, dict) and existing.get("start") == start and existing.get("end") == end:
            if not skip_requested and best_question_data is None and existing.get("best_question") is not None:
                entry["best_question"] = existing.get("best_question")
            annotations_list[idx] = entry
            updated = True
            break
    if not updated:
        annotations_list.append(entry)

    annotations_list.sort(key=lambda item: (item.get("start", 0), item.get("end", 0)))

    annotations_path = annotations_bundle["path"]
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        annotations_path.write_text(
            json.dumps(annotations_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store annotation: {exc}")

    try:
        annotation_rel = annotations_path.relative_to(DOWNLOADS_DIR).as_posix()
    except ValueError:
        annotation_rel = None

    return JSONResponse({
        "success": True,
        "annotation": entry,
        "annotations_file": annotation_rel,
        "updated": updated,
    })


# -----------------------------
# NEW: WebSocket endpoint for streaming interval results
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
        api_key = params.get("api_key")

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
            # offload blocking work to thread
            result_text = await asyncio.to_thread(generate_questions_for_segment, video_id, start, end, api_key)
            result_payload = _maybe_parse_json(result_text)
            payload_for_save = result_payload if isinstance(result_payload, (dict, list)) else result_text
            saved_json_url = persist_segment_questions_json(video_id, start, end, payload_for_save)
            await websocket.send_json({
                "type": "segment_result",
                "start": start,
                "end": end,
                "result": result_payload
            })
            await websocket.send_json({"type": "done", "output_json": saved_json_url})
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
            result_text = await asyncio.to_thread(generate_questions_for_segment, video_id, seg_start, seg_end, api_key)
            result_obj = _maybe_parse_json(result_text)
            aggregated["segments"].append({
                "start": seg_start,
                "end": seg_end,
                "result": result_obj
            })
            # push each segment as soon as it's ready
            await websocket.send_json({
                "type": "segment_result",
                "start": seg_start,
                "end": seg_end,
                "result": result_obj
            })

        # Save aggregated JSON to disk
        questions_dir = DOWNLOADS_DIR / video_id / "questions"
        questions_dir.mkdir(parents=True, exist_ok=True)
        out_path = questions_dir / f"questions_interval_{int(interval_seconds)}s.json"
        out_path.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8")
        output_url = f"/downloads/{out_path.relative_to(DOWNLOADS_DIR).as_posix()}"

        await websocket.send_json({
            "type": "done",
            "segments_count": len(segments),
            "output_json": output_url
        })
        await websocket.close()

    except WebSocketDisconnect:
        # client disconnected mid-stream
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

