# admin_routes.py
import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import (
    APIRouter,
    Body,
    Form,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
load_dotenv()

# ----- Local paths (keep consistent with main.py) -----
BASE_DIR = Path(__file__).parent.resolve()
TEMPLATES_DIR = BASE_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Three routers:
#  - pages: mounted under /admin
#  - api:   mounted under /api
#  - ws:    mounted with NO prefix (keeps /ws/... as-is)
router_admin_pages = APIRouter()
router_admin_api = APIRouter()
router_admin_ws = APIRouter()

# ----- Env -----
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


# ===== Helpers duplicated here (tiny / no circulars) =====
def parse_iso8601_duration_to_seconds(duration: str) -> int:
    """Parse ISO8601 duration like PT1H2M3S -> seconds."""
    if not duration:
        return 0
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not m:
        return 0
    h = int(m.group(1) or 0)
    mins = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + mins * 60 + s


def format_hhmmss(total_seconds: int) -> str:
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# =========================================================
# Admin page
# =========================================================
@router_admin_pages.get("/", response_class=HTMLResponse)
def admin_page(request: Request):
    # admin.html is self-contained (fetches data via JS), so no heavy context needed
    return templates.TemplateResponse("admin.html", {"request": request})


# =========================================================
# Admin API
# =========================================================
@router_admin_api.get("/yt_search")
async def yt_search(
    request: Request,
    q: str,
    min_minutes: int = 5,
    max_minutes: int = 120,
    max_results: int = 50,
    page_token: Optional[str] = None,
    page_size: Optional[int] = 20,
):
    """
    Search YouTube for videos with permissive filtering (admin use).
    """
    if not YOUTUBE_API_KEY:
        return {
            "success": False,
            "message": "Missing YOUTUBE_API_KEY on server.",
            "items": [],
        }

    q = (q or "").strip()
    if not q:
        # default seed for admin search
        q = "educational kids learning children"

    min_seconds = max(0, int(min_minutes)) * 60
    max_seconds = max(0, int(max_minutes)) * 60
    if max_seconds and max_seconds < min_seconds:
        min_seconds, max_seconds = max_seconds, min_seconds

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            size = int(page_size or 20)
            size = max(1, min(size, 50))

            # 1) Search
            search_params = {
                "part": "snippet",
                "q": q + " educational kids children learning",
                "type": "video",
                "maxResults": min(size * 2, 50),  # fetch more to filter
                "safeSearch": "strict",
                "videoEmbeddable": "true",
                "order": "relevance",
                "key": YOUTUBE_API_KEY,
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
                return {"success": True, "items": [], "message": "No results found."}

            # 2) Details
            videos_params = {
                "part": "snippet,contentDetails,status",
                "id": ",".join(video_ids),
                "key": YOUTUBE_API_KEY,
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

                duration_iso = content_details.get("duration")
                duration_seconds = parse_iso8601_duration_to_seconds(duration_iso)
                if duration_seconds <= 0:
                    continue

                # flexible duration window (80% variance)
                variance = 0.8
                adjusted_min = max(0, min_seconds * variance)
                adjusted_max = (
                    max_seconds * (1 + variance) if max_seconds else float("inf")
                )
                if duration_seconds < adjusted_min:
                    continue
                if max_seconds and duration_seconds > adjusted_max:
                    continue

                # only block obviously age-restricted
                content_rating = content_details.get("contentRating", {})
                if content_rating.get("ytRating") == "ytAgeRestricted":
                    continue

                title_lower = snippet.get("title", "").lower()
                description_lower = snippet.get("description", "").lower()
                if any(
                    k in title_lower or k in description_lower
                    for k in ["violence", "horror", "adult", "mature"]
                ):
                    continue

                thumbs = snippet.get("thumbnails", {})
                thumb = (
                    thumbs.get("medium", {}).get("url")
                    or thumbs.get("high", {}).get("url")
                    or thumbs.get("default", {}).get("url")
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

            return {
                "success": True,
                "items": items_out[:size],
                "count": len(items_out[:size]),
                "page_size": size,
                "nextPageToken": search_data.get("nextPageToken"),
                "searchTotal": search_data.get("pageInfo", {}).get("totalResults"),
                "message": f"Showing up to {size} results ordered by relevance.",
            }

    except httpx.HTTPStatusError as e:
        return {"success": False, "message": f"YouTube API error: {e}"}
    except Exception as e:
        return {"success": False, "message": f"Server error: {e}"}


@router_admin_api.post("/download")
async def api_download(url: str = Form(...)):
    # Lazy import to avoid circular dependency
    from main import download_youtube

    outcome = download_youtube(url)
    return outcome


@router_admin_api.post("/frames/{video_id}")
async def api_extract_frames(video_id: str):
    from main import extract_frames_per_second_for_video

    return extract_frames_per_second_for_video(video_id)


@router_admin_api.post("/submit-questions")
async def submit_questions(payload: Dict[str, Any] = Body(...)):
    """
    Submit and save finalized questions (admin 'Submit' in UI).
    Saves to downloads/<video_id>/questions/<video_id>.json
    """
    video_id = payload.get("video_id")
    questions_data = payload.get("questions", [])
    if not video_id or not questions_data:
        raise HTTPException(status_code=400, detail="Missing video_id or questions")

    from pathlib import Path
    import json
    from datetime import datetime

    DOWNLOADS_DIR = Path(__file__).parent.resolve() / "downloads"
    questions_dir = DOWNLOADS_DIR / video_id / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)
    out_path = questions_dir / f"{video_id}.json"

    aggregated = {
        "video_id": video_id,
        "submitted_at": datetime.utcnow().isoformat(),
        "status": "submitted",
        "segments": questions_data,
    }

    try:
        out_path.write_text(
            json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save: {e}")

    return {
        "success": True,
        "message": "Questions submitted successfully",
        "file_url": f"/downloads/{video_id}/questions/{out_path.name}",
        "file_path": str(out_path),
    }


# =========================================================
# WebSocket – keep original path: /ws/questions/{video_id}
# =========================================================
@router_admin_ws.websocket("/ws/questions/{video_id}")
async def ws_questions(websocket: WebSocket, video_id: str):
    await websocket.accept()
    try:
        params = await websocket.receive_json()
        start_seconds = int(params.get("start_seconds", 0))
        interval_seconds = int(params.get("interval_seconds", 60))
        full_duration = bool(params.get("full_duration", False))

        # Lazy imports to avoid circulars
        from main import (
            generate_questions_for_segment_with_retry,
            build_segments_from_duration,
            _maybe_parse_json,
        )
        from pathlib import Path
        import json

        DOWNLOADS_DIR = Path(__file__).parent.resolve() / "downloads"
        frames_dir = DOWNLOADS_DIR / video_id / "extracted_frames"
        if not frames_dir.exists():
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Frames not found. Please extract frames first.",
                }
            )
            await websocket.close()
            return

        # Load duration if available
        duration_seconds = None
        json_path = frames_dir / "frame_data.json"
        if json_path.exists():
            try:
                info = json.loads(json_path.read_text(encoding="utf-8"))
                duration_seconds = int(
                    float(info.get("video_info", {}).get("duration_seconds", 0))
                )
            except Exception:
                duration_seconds = None

        # One-shot interval
        if not full_duration:
            start = max(0, int(start_seconds))
            end = start + max(1, int(interval_seconds)) - 1
            if duration_seconds is not None and end > duration_seconds:
                end = duration_seconds

            await websocket.send_json(
                {
                    "type": "status",
                    "message": f"Generating questions for {start}-{end}s...",
                }
            )
            result_text = await asyncio_to_thread(
                generate_questions_for_segment_with_retry, video_id, start, end
            )
            result_obj = _maybe_parse_json(result_text)

            await websocket.send_json(
                {
                    "type": "segment_result",
                    "start": start,
                    "end": end,
                    "result": result_obj,
                }
            )
            await websocket.send_json({"type": "done", "auto_saved": False})
            await websocket.close()
            return

        # Full loop
        if duration_seconds is None or duration_seconds <= 0:
            await websocket.send_json(
                {"type": "error", "message": "Unable to determine video duration."}
            )
            await websocket.close()
            return

        segments = build_segments_from_duration(
            duration_seconds, interval_seconds, start_seconds
        )
        await websocket.send_json(
            {
                "type": "status",
                "message": f"Starting full-duration generation over {len(segments)} segments.",
            }
        )

        aggregated = {
            "video_id": video_id,
            "interval_seconds": int(interval_seconds),
            "start_offset": int(start_seconds),
            "duration_seconds": duration_seconds,
            "segments": [],
        }

        for idx, (seg_start, seg_end) in enumerate(segments, start=1):
            await websocket.send_json(
                {
                    "type": "status",
                    "message": f"[{idx}/{len(segments)}] {seg_start}-{seg_end}s",
                }
            )
            result_text = await asyncio_to_thread(
                generate_questions_for_segment_with_retry,
                video_id,
                seg_start,
                seg_end,
            )
            result_obj = _maybe_parse_json(result_text)
            aggregated["segments"].append(
                {"start": seg_start, "end": seg_end, "result": result_obj}
            )
            await websocket.send_json(
                {
                    "type": "segment_result",
                    "start": seg_start,
                    "end": seg_end,
                    "result": result_obj,
                }
            )

        await websocket.send_json(
            {
                "type": "done",
                "segments_count": len(segments),
                "auto_saved": False,
                "data": aggregated,
            }
        )
        await websocket.close()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close()
        except Exception:
            pass


# small helper: run sync function in thread (keeps this file standalone)
import asyncio


def asyncio_to_thread(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))
