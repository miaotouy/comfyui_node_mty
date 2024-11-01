import os
import json
import base64
import openai
from PIL import Image
import io
import numpy as np
import torch

class OpenAINode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_env_vars": ("BOOLEAN", {"default": False}),
                "base_url": ("STRING", {"default": "https://api.openai.com/v1"}),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "gpt-4-vision-preview"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 2, "step": 0.1}),
                "max_tokens": ("INT", {"default": 300, "min": 1, "max": 16384}),
                "system_prompt": ("STRING", {"multiline": True}),
                "user_input": ("STRING", {"multiline": True}),
            },
            "optional": {
                "history": ("HISTORY",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "mty🦉node/🦉openai_api"

    def run(self, use_env_vars, base_url, api_key, model, temperature, max_tokens, user_input, system_prompt=None, history=None, image=None):
        if use_env_vars:
            base_url = os.getenv('OPENAI_API_BASE', base_url)
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return ("Error: OPENAI_API_KEY not found in environment variables.",)

        # 确保 base_url 以 '/v1' 结尾
        if not base_url.endswith('/v1'):
            base_url = base_url.rstrip('/') + '/v1'

        client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            try:
                history_list = json.loads(history)
                for msg in history_list:
                    if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                        raise ValueError("Invalid history format")
                messages.extend(history_list)
            except json.JSONDecodeError:
                return ("Error: Invalid JSON in history",)
            except ValueError as e:
                return (f"Error in history format: {str(e)}",)

        if image is not None:
            try:
                image_content = self.encode_image(image)
                messages.append({
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_input},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_content}"
                            }
                        }
                    ]
                })
            except Exception as e:
                return (f"Error encoding image: {str(e)}",)

        else:
            messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            assistant_response = response.choices[0].message.content
            return (assistant_response,)
        except Exception as e:
            return (f"Error: {str(e)}",)

    @staticmethod
    def encode_image(image):
        try:
            # 如果是张量,转换为numpy数组
            if torch.is_tensor(image):
                image = image.cpu().numpy()

            # 处理4维张量 (可能是批处理图像或gif帧)
            if len(image.shape) == 4:
                # 如果是批处理图像,只取第一张
                image = image[0]
            
            # 确保图像是3维的 (高度, 宽度, 通道)
            if len(image.shape) == 2:
                # 灰度图像,增加通道维度
                image = np.expand_dims(image, axis=-1)
            
            # 处理通道数
            if image.shape[-1] == 1:  # 单通道
                image = np.repeat(image, 3, axis=-1)
            elif image.shape[-1] == 4:  # RGBA
                image = image[..., :3]  # 只保留RGB通道
            elif image.shape[-1] != 3:
                raise ValueError(f"Unsupported number of channels: {image.shape[-1]}")

            # 确保数值范围在0-255之间
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            # 创建PIL图像
            img = Image.fromarray(image)

            # 保存为JPEG
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return img_str
        except Exception as e:
            raise ValueError(f"Image encoding failed: {str(e)}")



class HistoryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "role1": (["user", "assistant", "system"],),
                "content1": ("STRING", {"multiline": True}),
                "role2": (["user", "assistant", "system"],),
                "content2": ("STRING", {"multiline": True}),
                "role3": (["user", "assistant", "system"],),
                "content3": ("STRING", {"multiline": True}),
                "role4": (["user", "assistant", "system"],),
                "content4": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("HISTORY",)
    RETURN_NAMES = ("history",)
    FUNCTION = "create_history"
    CATEGORY = "mty🦉node/🦉openai_api"

    def create_history(self, role1=None, content1=None, role2=None, content2=None, role3=None, content3=None, role4=None, content4=None):
        history = []
        for role, content in [(role1, content1), (role2, content2), (role3, content3), (role4, content4)]:
            if role and content and content.strip():
                history.append({"role": role, "content": content.strip()})
        return (json.dumps(history),)

class MergeHistoryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "history1": ("HISTORY",),
                "history2": ("HISTORY",),
                "history3": ("HISTORY",),
                "history4": ("HISTORY",),
            }
        }

    RETURN_TYPES = ("HISTORY",)
    RETURN_NAMES = ("history",)
    FUNCTION = "merge_history"
    CATEGORY = "mty🦉node/🦉openai_api"

    def merge_history(self, history1=None, history2=None, history3=None, history4=None):
        merged = []
        for history in [history1, history2, history3, history4]:
            if history:
                try:
                    merged.extend(json.loads(history))
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON in history: {history}")
        return (json.dumps(merged),)

NODE_CLASS_MAPPINGS = {
    "OpenAINode": OpenAINode,
    "HistoryNode": HistoryNode,
    "MergeHistoryNode": MergeHistoryNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAINode": "🦉OpenAI API",
    "HistoryNode": "🦉历史消息",
    "MergeHistoryNode": "🦉合并历史消息"
}
