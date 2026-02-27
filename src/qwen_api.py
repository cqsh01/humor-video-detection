import os, base64
from http import HTTPStatus
from dashscope import MultiModalConversation, Message, Image, Text
from dotenv import load_dotenv
load_dotenv()
import dashscope
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def _img_b64(path):
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode("utf-8")

def ask_image(prompt: str, image_path: str, model="qwen-vl-plus", temperature=0.2):
    messages = [Message(role="user", content=[Text(prompt), Image(_img_b64(image_path))])]
    resp = MultiModalConversation.call(model=model, messages=messages, temperature=temperature)
    if resp.status_code == HTTPStatus.OK:
        # 返回文本
        return resp.output.choices[0].message.content[0]["text"]
    raise RuntimeError(f"Qwen error: {resp.code} {resp.message}")

def ask_text(prompt: str, model="qwen-vl-plus", temperature=0.2):
    # 仍用 MultiModalConversation，只传 Text
    messages = [Message(role="user", content=[Text(prompt)])]
    resp = MultiModalConversation.call(model=model, messages=messages, temperature=temperature)
    if resp.status_code == HTTPStatus.OK:
        return resp.output.choices[0].message.content[0]["text"]
    raise RuntimeError(f"Qwen error: {resp.code} {resp.message}")