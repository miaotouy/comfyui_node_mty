import os
import json
import base64
import openai
from PIL import Image
import io
import numpy as np

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
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "user_input": ("STRING", {"multiline": True}),
                "history": ("HISTORY",),
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "mty🦉node/🦉openai_api"

    def run(self, use_env_vars, base_url, api_key, model, temperature, system_prompt, user_input, history, image):
        if use_env_vars:
            base_url = os.getenv('OPENAI_API_BASE', base_url)
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return ("Error: OPENAI_API_KEY not found in environment variables.",)

        openai.api_base = base_url
        openai.api_key = api_key

        try:
            history_list = json.loads(history)
            for msg in history_list:
                if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                    raise ValueError("Invalid history format")
        except json.JSONDecodeError:
            history_list = []
        except ValueError as e:
            return (f"Error in history format: {str(e)}",)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_list)

        if image is not None:
            image_content = self.encode_image(image)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_content}"}}
                ]
            })
        else:
            messages.append({"role": "user", "content": user_input})

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            assistant_response = response['choices'][0]['message']['content']
            return (assistant_response,)
        except Exception as e:
            return (f"Error: {str(e)}",)

    @staticmethod
    def encode_image(image):
        if isinstance(image, np.ndarray):
            img_array = image
        else:
            img_array = image.squeeze().permute(1, 2, 0).byte().numpy()

        img = Image.fromarray(img_array)

        if img.mode == 'RGBA':
            img = img.convert('RGB')

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

class HistoryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "role1": (["user", "assistant", "system"], {"default": "user"}),
                "content1": ("STRING", {"multiline": True}),
                "role2": (["user", "assistant", "system"], {"default": "assistant"}),
                "content2": ("STRING", {"multiline": True}),
                "role3": (["user", "assistant", "system"], {"default": "user"}),
                "content3": ("STRING", {"multiline": True}),
                "role4": (["user", "assistant", "system"], {"default": "assistant"}),
                "content4": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("HISTORY",)
    RETURN_NAMES = ("history",)
    FUNCTION = "create_history"
    CATEGORY = "mty🦉node/🦉openai_api"

    def create_history(self, role1, content1, role2, content2, role3, content3, role4, content4):
        history = []
        for role, content in [(role1, content1), (role2, content2), (role3, content3), (role4, content4)]:
            if content.strip():
                history.append({"role": role, "content": content.strip()})
        return (json.dumps(history),)

class MergeHistoryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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

    def merge_history(self, history1, history2, history3, history4):
        merged = []
        for history in [history1, history2, history3, history4]:
            if history.strip():
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
