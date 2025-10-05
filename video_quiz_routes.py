import os, re, io, json
from pathlib import Path
from fastapi import APIRouter, Body, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from config import GRADING_CONFIG
from rapidfuzz import fuzz
from openai import OpenAI
from typing import cast, Any, Dict
from dotenv import load_dotenv
load_dotenv()

# ---- Local paths (mirrors your main.py) ----
BASE_DIR = Path(__file__).parent.resolve()
DOWNLOADS_DIR = BASE_DIR / "downloads"
TEMPLATES_DIR = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router_video_quiz = APIRouter()

# Two routers so we can mount API under /api and keep /kids at root
router_api = APIRouter()
router_pages = APIRouter()


# Kids-specific OpenAI client (Whisper + AI fallback for grading)
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY missing. Add it to your .env or environment."
        )
    return OpenAI(api_key=api_key)


# Put these two helpers near the top of the file (once):
def _parse_duration_to_seconds(val) -> int | None:
    """Accept int/float seconds or 'HH:MM:SS'/'MM:SS' strings and return seconds."""
    if val is None:
        return None
    # numeric
    if isinstance(val, (int, float)):
        try:
            return max(0, int(float(val)))
        except Exception:
            return None
    # string
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        # try hh:mm:ss or mm:ss
        parts = val.split(":")
        try:
            parts = [int(p) for p in parts]
        except Exception:
            # maybe it's a numeric string
            try:
                return max(0, int(float(val)))
            except Exception:
                return None
        if len(parts) == 3:
            h, m, s = parts
            return max(0, h * 3600 + m * 60 + s)
        if len(parts) == 2:
            m, s = parts
            return max(0, m * 60 + s)
        if len(parts) == 1:
            return max(0, int(parts[0]))
    return None


def _format_mmss(sec: int | None) -> str:
    if not sec or sec < 0:
        return "00:00"
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


# ============================================================
# Kids library discovery (reimplementation matches your behavior)
# ============================================================
def refresh_kids_videos_json():
    """
    Scan downloads/ and rebuild static/kids_videos.json.
    Reads title/thumbnail/duration from meta.json or fallback files.
    """
    results = []
    if not DOWNLOADS_DIR.exists():
        return results

    for item in sorted(DOWNLOADS_DIR.iterdir()):
        if not item.is_dir():
            continue
        vid = item.name

        # --- Find video file ---
        video_file = None
        for ext in (".mp4", ".webm", ".mkv", ".mov"):
            for cand in item.glob(f"*{ext}"):
                video_file = cand
                break
            if video_file:
                break
        if not video_file:
            continue

        # --- Metadata ---
        title = vid
        duration = None
        thumb_url = None
        meta_json = item / "meta.json"
        info_json = item / f"{vid}.info.json"

        meta_source = None
        if meta_json.exists():
            meta_source = meta_json
        elif info_json.exists():
            meta_source = info_json

        if meta_source:
            try:
                meta = json.loads(meta_source.read_text(encoding="utf-8"))
                title = meta.get("title") or title
                duration = meta.get("duration") or meta.get("duration_string")
                thumb_url = meta.get("thumbnail")
            except Exception:
                pass

        # --- Thumbnail fallback ---
        if not thumb_url:
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                thumb_file = next(item.glob(f"*{ext}"), None)
                if thumb_file:
                    thumb_url = f"/downloads/{vid}/{thumb_file.name}"
                    break

        # --- Fallback to first extracted frame ---
        if not thumb_url:
            frames_dir = item / "extracted_frames"
            first_frame = next(frames_dir.glob("frame_0001s.jpg"), None)
            if first_frame:
                thumb_url = f"/downloads/{vid}/extracted_frames/{first_frame.name}"

        thumb_url = thumb_url or "/static/default-unlock.png"

        # --- Duration (meta/info OR extracted_frames/frame_data.json) ---
        dur_sec = None

        # 1) meta.json or <id>.info.json
        if meta_source:
            try:
                meta = json.loads(meta_source.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            title = meta.get("title") or title
            # these keys commonly exist in yt-dlp info
            for k in ("duration", "duration_string", "approx_duration_ms"):
                if k in meta:
                    cand = meta[k]
                    if k == "approx_duration_ms":
                        try:
                            cand = float(cand) / 1000.0
                        except Exception:
                            cand = None
                    dur_sec = _parse_duration_to_seconds(cand)
                    if dur_sec:
                        break

        # 2) fallback to extracted_frames/frame_data.json (correct path)
        if dur_sec is None:
            frame_json = item / "extracted_frames" / "frame_data.json"
            if frame_json.exists():
                try:
                    fd = json.loads(frame_json.read_text(encoding="utf-8"))
                    raw = fd.get("video_info", {}).get("duration_seconds", 0)
                    dur_sec = _parse_duration_to_seconds(raw) or 0
                except Exception:
                    dur_sec = 0

        duration = _format_mmss(dur_sec)

        results.append(
            {
                "video_id": vid,
                "title": title,
                "duration": duration,
                "local_path": f"/downloads/{vid}/{video_file.name}",
                "thumbnail": thumb_url,
            }
        )

    # Write the JSON file for caching
    out_path = BASE_DIR / "static" / "kids_videos.json"
    os.makedirs(out_path.parent, exist_ok=True)
    out_path.write_text(json.dumps({"videos": results}, indent=2), encoding="utf-8")

    return results


# ============================================================
# Kids: library & page routes
# ============================================================
@router_video_quiz.get("/kids_videos")
def list_kids_videos():
    """Return JSON of all locally available kids videos"""
    videos = refresh_kids_videos_json()
    return {"success": True, "count": len(videos), "videos": videos}


@router_pages.get("/kids", response_class=HTMLResponse)
def kids_page(request: Request):
    """Kids panel page (front-end HTML that loads kids_videos.json)"""
    return templates.TemplateResponse("video_quiz.html", {"request": request})


@router_api.get("/final-questions/{video_id}")
def get_final_questions(video_id: str):
    """
    Loads final_questions.json for the given video_id.
    Returns the best LLM-ranked question per segment (lowest llm_ranking)
    that is not trashed. If trashed=True, skip to the next question.
    """
    path = DOWNLOADS_DIR / video_id / "final_questions" / "final_questions.json"
    if not path.exists():
        return {"success": False, "error": "final_questions.json not found"}

    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    selected_segments = []

    for seg in segments:
        ai_qs = seg.get("aiQuestions", [])
        if not ai_qs:
            continue

        # Sort by llm_ranking (lowest = best)
        sorted_qs = sorted(ai_qs, key=lambda q: q.get("llm_ranking", 999))

        # Find first non-trashed question
        chosen_q = next((q for q in sorted_qs if not q.get("trashed", False)), None)
        if chosen_q:
            selected_segments.append({
                "segment_range_start": seg.get("start"),
                "segment_range_end": seg.get("end"),
                "question": chosen_q.get("editedQuestion"),
                "answer": chosen_q.get("editedAnswer"),
                "llm_ranking": chosen_q.get("llm_ranking"),
                "expert_ranking": chosen_q.get("expert_ranking"),
            })

    return {"success": True, "segments": selected_segments}

# ============================================================
# Answer-checker helpers (moved verbatim from your main.py)
# ============================================================
NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
SCALE_WORDS = {"hundred": 100, "thousand": 1000, "million": 1_000_000}

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "and",
    "of",
    "to",
    "it",
    "in",
    "on",
    "at",
    "for",
    "was",
    "were",
    "be",
    "being",
    "been",
    "am",
    "do",
    "did",
    "does",
    "done",
    "they",
    "them",
    "their",
    "there",
    "here",
    "that",
    "this",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "me",
    "my",
    "your",
    "his",
    "her",
    "our",
    "ours",
    "with",
    "by",
    "from",
}
FILLER_WORDS = {"um", "uh", "like", "you know", "hmm", "well", "okay", "so"}

SYNONYMS = {
    # Feelings
    "scared": "afraid",
    "frightened": "afraid",
    "fearful": "afraid",
    "nervous": "afraid",
    "worried": "afraid",
    "sad": "unhappy",
    "crying": "unhappy",
    "mad": "angry",
    "upset": "angry",
    "annoyed": "angry",
    "happy": "happy",
    "glad": "happy",
    "joyful": "happy",
    "excited": "happy",
    "fun": "happy",
    "laughing": "happy",
    "smiling": "happy",
    # Family
    "mom": "mother",
    "mommy": "mother",
    "dad": "father",
    "daddy": "father",
    "grandma": "grandmother",
    "grandpa": "grandfather",
    "bro": "brother",
    "sis": "sister",
    "sissy": "sister",
    # Animals
    "puppy": "dog",
    "puppies": "dog",
    "kitten": "cat",
    "kitties": "cat",
    "bunny": "rabbit",
    "hare": "rabbit",
    "pony": "horse",
    # Food
    "soda": "drink",
    "juice": "drink",
    "milk": "drink",
    "water": "drink",
    "snack": "food",
    "meal": "food",
    "candy": "sweet",
    "sweets": "sweet",
    "chocolate": "sweet",
    "cookie": "sweet",
    "icecream": "sweet",
    "ice cream": "sweet",
    "cake": "sweet",
    "pie": "sweet",
    # Everyday objects
    "automobile": "car",
    "truck": "car",
    "bus": "car",
    "bike": "bicycle",
    "tv": "television",
    "show": "movie",
    "cartoon": "movie",
    "film": "movie",
    # Size
    "large": "big",
    "huge": "big",
    "giant": "big",
    "enormous": "big",
    "little": "small",
    "tiny": "small",
    "short": "small",
    # Speed
    "quick": "fast",
    "speedy": "fast",
    # Yes/No
    "yeah": "yes",
    "yep": "yes",
    "yup": "yes",
    "nope": "no",
    "nah": "no",
}


# ============================================================
# Helpers
# ============================================================
def words_to_numbers(text: str) -> list[int]:
    """Extract numbers (digits or words) from text."""
    text = text.lower().strip()
    numbers = [int(d) for d in re.findall(r"\d+", text)]

    tokens = re.split(r"[-\s]+", text)
    total, current, found_number = 0, 0, False

    for token in tokens + ["end"]:
        if token in NUM_WORDS:
            found_number = True
            current += NUM_WORDS[token]
        elif token in SCALE_WORDS:
            found_number = True
            scale = SCALE_WORDS[token]
            if current == 0:
                current = 1
            current *= scale
            if scale > 100:
                total += current
                current = 0
        else:
            if found_number:
                total += current
                numbers.append(total)
                total, current, found_number = 0, 0, False

    return numbers


def normalize_text(text: str) -> str:
    """Clean text: lowercase, strip fillers/stopwords, map synonyms."""
    tokens = re.findall(r"[a-z]+", text.lower())
    normalized = []
    for t in tokens:
        if t in STOPWORDS or t in FILLER_WORDS or t in NUM_WORDS or t in SCALE_WORDS:
            continue
        normalized.append(SYNONYMS.get(t, t))
    return " ".join(normalized)


def keyword_overlap(expected: str, user: str) -> float:
    exp_words = set(expected.split())
    usr_words = set(user.split())
    return len(exp_words & usr_words) / max(1, len(exp_words))


def simplify_item(item: str) -> str:
    item = item.strip()
    m = re.search(r"\bcalled\s+(.+)", item)
    if m:
        item = m.group(1)
    norm = normalize_text(item)
    toks = norm.split()
    if len(toks) > 3:
        norm = " ".join(toks[-3:])
    return norm


def extract_items(expected_raw: str) -> list[str]:
    parts = [
        p
        for p in re.split(r",|\sand\s", expected_raw, flags=re.IGNORECASE)
        if p.strip()
    ]
    return [simplify_item(p) for p in parts if simplify_item(p)]


def list_match(expected_raw: str, user_raw: str) -> tuple[int, int, list[str]]:
    items = extract_items(expected_raw)
    user_norm = normalize_text(user_raw)

    matched = set()
    for item in items:
        score = max(
            fuzz.partial_ratio(item, user_norm), fuzz.token_set_ratio(item, user_norm)
        )
        if score >= 60:
            matched.add(item)
    return len(matched), len(items), list(matched)


def required_items_from_question(question: str, expected: str) -> int:
    total_expected = len([p for p in re.split(r",|and", expected) if p.strip()])
    if not question:
        return total_expected
    num_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    for word, val in num_map.items():
        if word in question.lower():
            return min(val, total_expected)
    return total_expected


# ============================================================
# POST /api/check_answer (moved verbatim; decorator adjusted)
# ============================================================
@router_api.post(
    "/api/check_answer".replace("/api", "")
)  # keep original route path under /api prefix
async def check_answer(payload: dict = Body(...)):
    expected = cast(str, payload.get("expected") or "").strip().lower()
    user = cast(str, payload.get("user") or "").strip().lower()
    question = cast(str, payload.get("question") or "").strip().lower()

    print(
        f"🔎 Checking answers | Q='{question}' | Expected='{expected}' | User='{user}'"
    )

    if not expected or not user:
        return {
            "similarity": 0.0,
            "expected": expected,
            "user": user,
            "is_numeric": False,
            "status": "wrong",
            "reason": "Empty input",
        }

    # --- Quick RapidFuzz similarity ---
    exp_clean = normalize_text(expected)
    usr_clean = normalize_text(user)

    pr = fuzz.partial_ratio(exp_clean, usr_clean) / 100.0
    tsr = fuzz.token_set_ratio(exp_clean, usr_clean) / 100.0
    score = max(pr, tsr)

    print(f"  RapidFuzz → pr={pr:.3f}, tsr={tsr:.3f}, final={score:.3f}")

    if score >= GRADING_CONFIG["rapidfuzz_correct"]:
        return {
            "similarity": round(score, 3),
            "expected": expected,
            "user": user,
            "is_numeric": False,
            "status": "correct",
            "reason": f"High RapidFuzz score {score:.2f}",
        }

    if score <= GRADING_CONFIG["rapidfuzz_wrong"]:
        return {
            "similarity": round(score, 3),
            "expected": expected,
            "user": user,
            "is_numeric": False,
            "status": "wrong",
            "reason": f"Low RapidFuzz score {score:.2f}",
        }

    # --- Borderline → escalate to AI ---
    if GRADING_CONFIG["use_ai"]:
        try:
            client = get_openai_client()
            resp = client.chat.completions.create(
                model=GRADING_CONFIG["ai_model"],
                temperature=GRADING_CONFIG["ai_temperature"],
                max_tokens=GRADING_CONFIG["ai_max_tokens"],
                messages=[
                    {
                        "role": "system",
                        "content": "You are a friendly teacher grading a child's comprehension. Be lenient if the child expresses the same idea using different words or more specific examples. Respond with only one word: correct, almost, or wrong.",
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}\nExpected answer: {expected}\nChild's answer: {user}",
                    },
                ],
                timeout=GRADING_CONFIG["ai_timeout"],
            )
            ai_label = resp.choices[0].message.content.strip().lower()  # type: ignore
            if ai_label not in ["correct", "almost", "wrong"]:
                ai_label = "almost"  # default fallback
            return {
                "similarity": round(score, 3),
                "expected": expected,
                "user": user,
                "is_numeric": False,
                "status": ai_label,
                "reason": f"AI judged borderline case (RapidFuzz={score:.2f})",
            }
        except Exception as e:
            print("⚠️ AI call failed:", e)

    # --- Fallback if AI off or failed ---
    return {
        "similarity": round(score, 3),
        "expected": expected,
        "user": user,
        "is_numeric": False,
        "status": "almost",
        "reason": f"Borderline case defaulted (RapidFuzz={score:.2f})",
    }


# ============================================================
# Whisper transcription (moved verbatim; decorator adjusted)
# ============================================================
@router_api.post("/api/transcribe".replace("/api", ""))
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts audio (webm, wav, mp3, etc), sends to Whisper,
    no temp file saved (in-memory BytesIO).
    """
    try:
        contents = await file.read()
        audio_bytes = io.BytesIO(contents)
        client = get_openai_client()
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=("speech.webm", audio_bytes, file.content_type),
        )
        print("Whisper raw response:", transcription)
        return {"success": True, "text": transcription.text}
    except Exception as e:
        print("❌ Whisper transcription error:", e)
        return {"success": False, "error": str(e)}


# ============================================================
# Frontend config (skip prevention + thresholds)
# ============================================================
@router_api.get("/config")
async def get_config():
    # Single source of truth for frontend
    return {"skip_prevention": False, "thresholds": GRADING_CONFIG}
