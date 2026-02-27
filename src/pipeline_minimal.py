from qwen_api import ask_image, ask_text

MM_MODEL = "qwen3.5-plus"  # 先用你官方示例里的；若不支持图片再换 VL 型号

cap  = ask_image(PROMPT_CAPTION, fp, model=MM_MODEL)
cnt  = ask_image(PROMPT_COUNT, fp, model=MM_MODEL)
desc = ask_image(PROMPT_DESCR, fp, model=MM_MODEL)

humor = ask_text(context + "\n" + PROMPT_HUMOR, model=MM_MODEL)