import os, json, re
from tqdm import tqdm
from frame_sampling import sample_frames
from qwen_api import ask_image, ask_text

PROMPT_CAPTION = "用一句话描述这张画面里正在发生的事情。"
PROMPT_COUNT   = "这张画面里有几个人？只给出阿拉伯数字。"
PROMPT_DESCR   = "请简短描述画面中每个人的显著特征或动作，每行一个人。"
PROMPT_HUMOR   = "根据以下信息，这段视频好笑的原因是什么？请简要中文说明。"

def parse_int(s):
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None

def run_minimal(video_path, workdir="outputs", n_frames=6):
    vid = os.path.splitext(os.path.basename(video_path))[0]
    fdir = os.path.join(workdir, "frames", vid)
    rdir = os.path.join(workdir, "results")
    os.makedirs(rdir, exist_ok=True)

    frames = sample_frames(video_path, fdir, n=n_frames)
    frame_res = []
    for fp in tqdm(frames, desc=f"[{vid}] frames"):
        cap  = ask_image(PROMPT_CAPTION, fp)
        cnt  = ask_image(PROMPT_COUNT, fp)
        desc = ask_image(PROMPT_DESCR, fp)
        frame_res.append({"frame": fp, "caption": cap, "count_raw": cnt, "desc": desc})

    # 简单聚合（阶段 A 先这样）
    counts = [parse_int(fr["count_raw"]) for fr in frame_res if parse_int(fr["count_raw"]) is not None]
    maj_count = max(set(counts), key=counts.count) if counts else None
    merged_caption = max([fr["caption"] for fr in frame_res], key=len, default="")
    merged_desc = "\n".join(fr["desc"] for fr in frame_res)

    context = f"人数投票：{maj_count}\n场景描述：{merged_caption}\n人物细节：\n{merged_desc}\n"
    humor = ask_text(context + "\n" + PROMPT_HUMOR)

    out = {
        "video": video_path,
        "frames": frames,
        "frame_results": frame_res,
        "summary": {
            "person_count_majority": maj_count,
            "caption_merged": merged_caption,
            "desc_merged": merged_desc,
            "humor_explanation": humor
        }
    }
    with open(os.path.join(rdir, f"{vid}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    res = run_minimal("data/WITHOUTOOPS_slow.avi")
    print(res["summary"])