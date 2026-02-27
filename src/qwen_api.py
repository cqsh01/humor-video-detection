import os
import base64
import mimetypes
from http import HTTPStatus

import dashscope
from dashscope import MultiModalConversation
from dotenv import load_dotenv

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def _img_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def ask_image(prompt: str, image_path: str, model: str, temperature: float = 0.2) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"image": _img_data_url(image_path)},
                {"text": prompt},
            ],
        }
    ]
    resp = MultiModalConversation.call(
        model=model,
        messages=messages,
        temperature=temperature,
        api_key=os.getenv("DASHSCOPE_API_KEY"),  # 可选：不写也行（已设置 dashscope.api_key）
    )

    if resp.status_code == HTTPStatus.OK:
        # 一般是 content 里包含 {"text": "..."} 的结构
        content = resp.output.choices[0].message.content
        # 兼容处理：找第一个 text
        for item in content:
            if isinstance(item, dict) and "text" in item:
                return item["text"]
        # 如果没找到 text，就直接把 content 转字符串返回，方便你调试
        return str(content)

    raise RuntimeError(f"DashScope error: status={resp.status_code}, code={resp.code}, message={resp.message}")

def ask_text(prompt: str, model: str, temperature: float = 0.2) -> str:
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    resp = MultiModalConversation.call(model=model, messages=messages, temperature=temperature)

    if resp.status_code == HTTPStatus.OK:
        content = resp.output.choices[0].message.content
        for item in content:
            if isinstance(item, dict) and "text" in item:
                return item["text"]
        return str(content)

    raise RuntimeError(f"DashScope error: status={resp.status_code}, code={resp.code}, message={resp.message}")