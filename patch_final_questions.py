import json
from pathlib import Path

VIDEO_ID = "tSAKzVP2IGk"  # change if needed

base = Path("downloads") / VIDEO_ID
final_path = base / "final_questions" / "final_questions.json"
llm_path   = base / "questions" / f"{VIDEO_ID}.json"

print(f"Reading:\n  final = {final_path}\n  llm   = {llm_path}")
final = json.loads(final_path.read_text(encoding="utf-8"))
llm   = json.loads(llm_path.read_text(encoding="utf-8"))

llm_segments = llm.get("segments", [])
final_segments = final.get("segments", [])

def get_llm_rank(seg_idx: int, q_type: str):
    if seg_idx < 0 or seg_idx >= len(llm_segments):
        return None
    seg = llm_segments[seg_idx]
    result = seg.get("result")
    if not result:
        return None
    qs = result.get("questions") or {}
    obj = qs.get(q_type) or {}
    rank = obj.get("rank")
    try:
        return int(rank) if rank is not None else None
    except Exception:
        return None

changes = {"expert_set": 0, "llm_set": 0}

for i, seg in enumerate(final_segments):
    # prefer explicit segmentIndex if present; otherwise fall back to list order
    seg_idx = seg.get("segmentIndex", i)
    for q in seg.get("aiQuestions", []):
        q_type = q.get("type")

        # 1) Preserve/restore expert ranking from old 'ranking' if needed
        if "expert_ranking" not in q:
            if "ranking" in q:
                q["expert_ranking"] = q["ranking"]
                del q["ranking"]
                changes["expert_set"] += 1
        # If expert_ranking already exists, do nothing.

        # 2) Always fill llm_ranking from the LLM file (never from expert)
        llm_rank = get_llm_rank(seg_idx, q_type)
        q["llm_ranking"] = llm_rank
        changes["llm_set"] += 1

# Write back
final_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
print("Done. Changes:", changes)

# Quick sanity print on first non-empty segment
for i, seg in enumerate(final.get("segments", [])):
    if seg.get("aiQuestions"):
        print("\nSanity check on segment", seg.get("segmentIndex", i))
        for q in seg["aiQuestions"]:
            print(f"  {q.get('type'):10s}  expert={q.get('expert_ranking')}  llm={q.get('llm_ranking')}")
        break