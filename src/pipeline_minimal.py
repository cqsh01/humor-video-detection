import os
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.frame_sampling import sample_frames
from src.qwen_api import ask_image, ask_text

# ---- English prompts (建议你逐步换成你论文里要用的英文版) ----
PROMPT_COUNT = "How many persons are visible in this frame? Output ONLY one Arabic numeral."
PROMPT_DESCR = (
    "Describe each visible person using only clearly visible facts. "
    "One person per line. Include clothing color, action, and position (left/right/center). "
    "If unsure, write 'unknown'."
)
PROMPT_CAPTION = "Describe what is happening in this frame in one concise sentence (visible facts only)."
PROMPT_HUMOR = (
    "Based on the frame evidence with timestamps, determine whether the video contains humor. "
    "If humorous, identify ONE most likely humorous moment and output its start and end timestamps "
    "(mm:ss) chosen from the provided frame timestamps. Explain why it is humorous and cite evidence "
    "by referencing at least two timestamps."
)

MM_MODEL = "qwen3.5-plus"

_num_re = re.compile(r"\d+")

def parse_int(text: str) -> Optional[int]:
    if not text:
        return None
    m = _num_re.search(text)
    return int(m.group()) if m else None

def majority_vote(ints: List[int]) -> Optional[int]:
    if not ints:
        return None
    counts: Dict[int, int] = {}
    for x in ints:
        counts[x] = counts.get(x, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]

def run_on_video(video_path: str, out_root: str = "outputs", n_frames: int = 8, model: str = MM_MODEL) -> Dict[str, Any]:
    video_path = str(video_path)
    vid = Path(video_path).stem

    out_root_p = Path(out_root)
    frames_dir = out_root_p / "frames" / vid
    results_dir = out_root_p / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) 抽帧（带 timestamp）
    frame_evidence = sample_frames(video_path, str(frames_dir), n=n_frames)

    # 2) 逐帧多模态推理
    frame_results = []
    for fr in frame_evidence:
        fp = fr["path"]
        caption = ask_image(PROMPT_CAPTION, fp, model=model)
        count_raw = ask_image(PROMPT_COUNT, fp, model=model)
        persons_desc = ask_image(PROMPT_DESCR, fp, model=model)

        frame_results.append(
            {
                "frame_path": fp,
                "frame_index": fr["frame_index"],
                "time_sec": fr["time_sec"],
                "timecode": fr["timecode"],
                "caption": caption,
                "count_raw": count_raw,
                "count": parse_int(count_raw),
                "persons_desc": persons_desc,
            }
        )

    # 3) 视频级聚合（阶段A：简单版）
    counts = [x["count"] for x in frame_results if x["count"] is not None]
    person_count_majority = majority_vote(counts)

    # 4) 给视频级 Humor prompt 准备“证据文本”（关键：带时间戳）
    evidence_lines = []
    for x in frame_results:
        evidence_lines.append(
            f"[{x['timecode']}] caption={x['caption']}\n"
            f"[{x['timecode']}] person_count_raw={x['count_raw']}\n"
            f"[{x['timecode']}] persons_desc=\n{x['persons_desc']}\n"
        )
    evidence_text = "\n".join(evidence_lines)

    summary_text = (
        f"Video: {vid}\n"
        f"Sampled frame timestamps: {[x['timecode'] for x in frame_results]}\n"
        f"Majority person count estimate: {person_count_majority}\n\n"
        f"Frame evidence:\n{evidence_text}\n"
    )

    humor_explanation = ask_text(summary_text + "\n" + PROMPT_HUMOR, model=model)

    out = {
        "video_path": video_path,
        "video_id": vid,
        "model": model,
        "n_frames": n_frames,
        "frames": frame_evidence,
        "frame_results": frame_results,
        "summary": {
            "person_count_majority": person_count_majority,
            "humor_result": humor_explanation,
        },
    }

    out_path = results_dir / f"{vid}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out

def iter_videos(data_path: str) -> List[str]:
    p = Path(data_path)
    if p.is_file():
        return [str(p)]
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    return [str(x) for x in sorted(p.iterdir()) if x.suffix.lower() in exts]

if __name__ == "__main__":
    videos = iter_videos("data")
    if not videos:
        raise RuntimeError("No videos found in data/. Put your .avi/.mp4 in the data folder.")

    for vp in videos:
        print(f"Processing: {vp}")
        res = run_on_video(vp, out_root="outputs", n_frames=8, model=MM_MODEL)
        print("Done:", res["video_id"])