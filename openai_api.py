import json
import base64
import openai
from PIL import Image
import io
import numpy as np
import torch
import os
import uuid

class APISettingsNode:
    """
    API 设置节点，用于配置 OpenAI API 或其他兼容 API 的连接信息。

    Attributes:
        use_env_vars (bool): 是否使用环境变量获取 API 密钥和基础 URL。
        base_url (str): API 基础 URL，默认为 OpenAI API 的地址。
            可以设置为其他兼容 API 的地址，例如 OneAPI 的 API 地址，
            用于支持聚合模型接口，例如 GPT-4o、Claude-3-5-sonnet-20241022、gemini-1.5-pro-exp-0827 等。
        api_key (str): API 密钥。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_env_vars": ("BOOLEAN", {"default": False}),
                "base_url": ("STRING", {"default": "https://api.openai.com/v1"}),
                "api_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("API_SETTINGS",)
    RETURN_NAMES = ("api_settings",)
    FUNCTION = "get_api_settings"
    CATEGORY = "mty🦉node/🦉openai_api"

    def get_api_settings(self, use_env_vars, base_url, api_key):
        if use_env_vars:
            import os
            base_url = os.getenv('OPENAI_API_BASE', base_url)
            api_key = os.getenv('OPENAI_API_KEY', api_key)

        if not base_url.endswith('/v1'):
            base_url = base_url.rstrip('/') + '/v1'

        return ({
            "use_env_vars": use_env_vars,
            "base_url": base_url,
            "api_key": api_key,
        },)

class OpenAINode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_settings": ("API_SETTINGS",),
                "model": ("STRING", {
                    "default": "gpt-4o", 
                    "multiline": False,
                    "placeholder": "输入模型名称，如 gpt-4o, gemini-1.5-flash-exp-0827 等"
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 2, "step": 0.1}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "system_prompt": ("STRING", {"multiline": True}),
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "history": ("HISTORY",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "mty🦉node/🦉openai_api"

    def generate(self, api_settings, model, prompt, temperature, max_tokens, system_prompt=None, history=None, image=None):
        client = openai.OpenAI(
            base_url=api_settings["base_url"],
            api_key=api_settings["api_key"]
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            try:
                messages.extend(json.loads(history))
            except json.JSONDecodeError:
                pass

        if image is not None:
            image_content = ImageProcessor.encode_image(image)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_content}"}}
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return (response.choices[0].message.content,)
        except Exception as e:
            return (f"Error: {str(e)}",)

class ImageProcessor:
    """
    图像处理工具类，用于处理和编码图像数据。
    支持处理 Tensor、Numpy array 和 PIL Image 格式的输入。
    """
    
    @staticmethod
    def encode_image(image):
        """
        将图像编码为 base64 字符串。
        
        Args:
            image: 支持 Tensor、Numpy array 或 PIL Image 格式的图像数据
            
        Returns:
            str: base64 编码的图像字符串
            
        Raises:
            ValueError: 当图像处理失败时抛出异常
        """
        try:
            # 转换 Tensor 为 Numpy array
            if torch.is_tensor(image):
                image = image.cpu().numpy()

            # 处理 batch 维度
            if len(image.shape) == 4:
                image = image[0]
            
            # 处理单通道图像
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            
            # 标准化通道数
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            elif image.shape[-1] == 4:
                image = image[..., :3]
            elif image.shape[-1] != 3:
                raise ValueError(f"不支持的通道数: {image.shape[-1]}")

            # 标准化数据类型和范围
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            # 转换为 PIL Image 并编码
            img = Image.fromarray(image)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            raise ValueError(f"图像编码失败: {str(e)}")

class OpenAIChatNode:
    def __init__(self):
        self.client = None
        self.node_id = str(uuid.uuid4())
        self.cache_dir = "chat_cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.cache_file = os.path.join(self.cache_dir, f"chat_history_{self.node_id}.json")
        self.system_prompt = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_settings": ("API_SETTINGS",),
                "model": ("STRING", {
                    "default": "gpt-4o", 
                    "multiline": False,
                    "placeholder": "输入模型名称，如 gpt-4o, gemini-1.5-flash-exp-0827 等"
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 2, "step": 0.1}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "system_prompt": ("STRING", {"multiline": True}),
                "clear_history": ("BOOLEAN", {"default": False}),
                "user_input": ("STRING", {"multiline": True}),
            },
            "optional": {
                "external_history": ("HISTORY",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "HISTORY")
    RETURN_NAMES = ("full_conversation", "current_output", "updated_history")
    FUNCTION = "chat"
    CATEGORY = "mty🦉node/🦉openai_api"

    def chat(self, api_settings, model, user_input, temperature, max_tokens, clear_history, system_prompt=None, external_history=None, image=None):
        if self.client is None:
            self.client = openai.OpenAI(
                base_url=api_settings["base_url"],
                api_key=api_settings["api_key"]
            )

        # 处理外部历史
        if external_history:
            try:
                external_messages = json.loads(external_history)
            except json.JSONDecodeError:
                return ("Error: Invalid external history format.", "Error: Invalid external history format.", json.dumps([]))
        else:
            external_messages = []

        # 读取或清除缓存的内部历史
        if clear_history:
            internal_history = []
            self.save_cache([])
        else:
            internal_history = self.load_cache()

        # 更新系统提示
        if system_prompt is not None:
            self.system_prompt = system_prompt

        # 构建消息列表
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(external_messages)
        messages.extend(internal_history)

        # 添加新的用户输入
        if image is not None:
            try:
                image_content = ImageProcessor.encode_image(image)
                new_message = {
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
                }
            except Exception as e:
                error_message = f"Error encoding image: {str(e)}"
                return (self.format_conversation(messages), error_message, json.dumps(messages))
        else:
            new_message = {"role": "user", "content": user_input}

        messages.append(new_message)
        internal_history.append(new_message)

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            assistant_response = response.choices[0].message.content.strip()

            # 检查响应是否为空
            if not assistant_response:
                return (self.format_conversation(messages), "API返回了空回复，未写入历史记录。", json.dumps(external_messages + internal_history))
            
            # 只有在回复非空时才更新内部历史
            internal_history.append({"role": "assistant", "content": assistant_response})
            self.save_cache(internal_history)

            full_conversation = self.format_conversation(messages + [{"role": "assistant", "content": assistant_response}])
            updated_history = json.dumps(external_messages + internal_history)
            return (full_conversation, assistant_response, updated_history)
        except Exception as e:
            error_message = f"API Error: {str(e)}"
            return (self.format_conversation(messages), error_message, json.dumps(external_messages + internal_history))

    def format_conversation(self, messages):
        formatted = f"=== 完整对话历史 (Node ID: {self.node_id}) ===\n\n"
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if isinstance(content, list):
                content = content[0]['text'] + " [包含图像]"
            formatted += f"{role.capitalize()}: {content}\n"
            formatted += "-" * 40 + "\n"
        return formatted

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return []

    def save_cache(self, history):
        with open(self.cache_file, 'w') as f:
            json.dump(history, f)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return kwargs.get("clear_history", False) or kwargs.get("system_prompt") is not None or kwargs.get("user_input") != "" or kwargs.get("model") is not None

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



# comfyui必须在这里也要注册节点
NODE_CLASS_MAPPINGS = {
    "APISettingsNode": APISettingsNode,
    "OpenAINode": OpenAINode,
    "OpenAIChatNode": OpenAIChatNode,
    "HistoryNode": HistoryNode,
    "MergeHistoryNode": MergeHistoryNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APISettingsNode": "🦉API设置",
    "OpenAINode": "🦉OpenAI API",
    "OpenAIChatNode": "🦉OpenAI 连续对话",
    "HistoryNode": "🦉历史消息",
    "MergeHistoryNode": "🦉合并历史消息"
}

