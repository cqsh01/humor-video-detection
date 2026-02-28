import os
from typing import List, Dict, Any, Optional

import cv2
import numpy as np


def _sec_to_mmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


def sample_frames(
    video_path: str,
    out_dir: str,
    n: int = 8,
    *,
    jpeg_quality: int = 95,
) -> List[Dict[str, Any]]:
    """
    Uniformly sample n frames from a video and save them as JPGs.

    Returns a list of dict evidence:
    [
      {
        "path": ".../frame_00.jpg",
        "frame_index": 123,
        "time_sec": 4.10,
        "timecode": "00:04"
      },
      ...
    ]
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if total <= 0:
            raise RuntimeError(
                f"Cannot read total frame count for video: {video_path}. "
                "Try re-encoding the video or check codec support."
            )

        # pick indices uniformly in [0, total-1]
        idxs = np.linspace(0, max(0, total - 1), n).astype(int)

        results: List[Dict[str, Any]] = []
        for k, frame_idx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # Estimate time
            time_sec: Optional[float]
            if fps > 0:
                time_sec = float(frame_idx) / float(fps)
            else:
                # fps unknown: fallback to CAP_PROP_POS_MSEC after read
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                time_sec = float(pos_msec) / 1000.0 if pos_msec else None

            timecode = _sec_to_mmss(time_sec or 0.0)

            fp = os.path.join(out_dir, f"frame_{k:02d}_{timecode.replace(':','-')}.jpg")

            # Save JPG with quality
            cv2.imwrite(fp, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])

            results.append(
                {
                    "path": fp,
                    "frame_index": int(frame_idx),
                    "time_sec": float(time_sec) if time_sec is not None else None,
                    "timecode": timecode,
                }
            )

        return results
    finally:
        cap.release()