import os
import json
import base64
import openai
from PIL import Image
import io
import numpy as np
from comfy.model_base import BaseNode

def load_localization():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    locale_dir = os.path.join(current_dir, 'locales')
    
    # 获取系统语言，如果无法获取则默认使用英语
    lang = os.environ.get('LANG', 'en').split('.')[0]
    lang_file = os.path.join(locale_dir, f"{lang}.json")
    
    # 如果找不到对应的语言文件，就使用英语
    if not os.path.exists(lang_file):
        lang_file = os.path.join(locale_dir, "en.json")
    
    with open(lang_file, 'r', encoding='utf-8') as f:
        return json.load(f)

translations = load_localization()

def _(text):
    return translations.get(text, text)

class OpenAINode(BaseNode):
    def __init__(self):
        super().__init__()
        self.output_type = {"response": "STRING"}
        self.required = {
            "use_env_vars": ("BOOLEAN", {"default": False}),
            "base_url": ("STRING", {"default": "https://api.openai.com/v1"}),
            "api_key": ("STRING", {"default": ""}),
            "model": ("STRING", {"default": "gpt-4-vision-preview"}),
            "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 2, "step": 0.1}),
            "system_prompt": ("STRING", {"multiline": True, "default": _("You are a helpful assistant.")}),
            "user_input": ("STRING", {"multiline": True}),
            "history": ("STRING", {"multiline": True, "default": "[]"}),
            "image": ("IMAGE", {"default": None})
        }

    def run(self, use_env_vars, base_url, api_key, model, temperature, system_prompt, user_input, history, image):
        if use_env_vars:
            base_url = os.getenv('OPENAI_API_BASE', base_url)
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                error_msg = _("Error: OPENAI_API_KEY not found in environment variables.")
                print(error_msg)
                return {"response": error_msg}

        openai.api_base = base_url
        openai.api_key = api_key

        try:
            history_list = json.loads(history)
            for msg in history_list:
                if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                    raise ValueError(_("Invalid history format"))
        except json.JSONDecodeError:
            history_list = []
        except ValueError as e:
            error_msg = _("Error in history format: {}").format(str(e))
            print(error_msg)
            return {"response": error_msg}

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
            return {"response": assistant_response}
        except Exception as e:
            error_msg = _("Error: {}").format(str(e))
            print(error_msg)
            return {"response": error_msg}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_env_vars": ("BOOLEAN", {"default": False}),
                "base_url": ("STRING", {"default": "https://api.openai.com/v1"}),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "gpt-4-vision-preview"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 2, "step": 0.1}),
                "system_prompt": ("STRING", {"multiline": True, "default": _("You are a helpful assistant.")}),
                "user_input": ("STRING", {"multiline": True}),
                "history": ("STRING", {
                    "multiline": True, 
                    "default": "[]",
                    "description": _("JSON array of previous messages. Format: [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi there!'}]")
                }),
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "OpenAI"

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

NODE_CLASS_MAPPINGS = {
    "OpenAINode": OpenAINode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAINode": _("OpenAI Chat")
}
