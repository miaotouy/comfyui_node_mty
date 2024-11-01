from .openai_api import OpenAINode, HistoryNode, MergeHistoryNode

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

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
