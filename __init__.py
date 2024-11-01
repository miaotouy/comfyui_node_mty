from .openai_api import OpenAINode, HistoryNode, MergeHistoryNode,OpenAIChatNode, APISettingsNode


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


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
