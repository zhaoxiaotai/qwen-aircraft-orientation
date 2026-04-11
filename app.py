#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image

# ================== 请修改为你的实际路径 ==================
BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct"
LORA_PATH = "/root/autodl-tmp/loras/Qwen/Qwen3-VL-8B-Instruct/v5-20260410-110528/checkpoint-99"
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用设备: {DEVICE}")

print("正在加载处理器...")
processor = AutoProcessor.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    min_pixels=256 * 28 * 28,
    max_pixels=1024 * 28 * 28
)

print("正在加载基础模型...")
model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print("正在加载 LoRA 适配器...")
model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.merge_and_unload()
model.eval()
print("模型加载完成！")

def predict(image: Image.Image) -> str:
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "判断相机相对于飞机的方位"}
            ]
        }
    ]
    # 应用聊天模板得到文本
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 使用关键字参数 text=... 和 images=...
    inputs = processor(
        text=text,
        images=[image],
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # 提取助手回复
    if "assistant\n" in output_text:
        response = output_text.split("assistant\n")[-1].strip()
    else:
        response = output_text.strip()
    return response

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="上传飞机照片"),
    outputs=gr.Textbox(label="判断结果", lines=2),
    title="飞机方位判断系统",
    description="上传一张包含飞机的照片，模型将自动判断相机相对于飞机的方位。"
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )