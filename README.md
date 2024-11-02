
# comfyui_node_mty

## 我自己用的节点


![GitHub license](https://img.shields.io/github/license/miaotouy/comfyui_node_mty)
![GitHub last commit](https://img.shields.io/github/last-commit/miaotouy/comfyui_node_mty)

ComfyUI Node MTY 是一个为 ComfyUI 设计的自定义节点集合，主要用于扩展 ComfyUI 的功能，满足特定的工作流需求。这个项目包含了一系列在其他节点中找不到的实用功能。

## 特性

目前，ComfyUI Node MTY 提供以下节点：

1. **OpenAI API 节点**：与 OpenAI 的 API 进行交互，支持文本和图像处理。
2. **OpenAI 连续对话节点**: 支持在同一个节点内进行多轮对话，并保存历史记录。
3. **历史消息节点**：创建包含最多 4 轮对话的历史消息。
4. **合并历史消息节点**：合并来自多个历史消息节点的信息。

## 安装

1. 确保您已经安装了 ComfyUI。
2. 克隆此仓库到 ComfyUI 的 `custom_nodes` 目录：

   ```
   cd path/to/ComfyUI/custom_nodes
   git clone https://github.com/miaotouy/comfyui_node_mty.git
   ```

3. 安装所需的依赖：

   ```
   pip install openai Pillow
   ```

4. 重启 ComfyUI。

5. 可选：中文翻译

    如果您使用 AIGODLIKE-ComfyUI-Translation 插件进行翻译，可以将 `zh-CN\Nodes\comfyui_node_mty.json` 拷贝到 `.\ComfyUI\custom_nodes\AIGODLIKE-ComfyUI-Translation\zh-CN\Nodes\` 目录下，以启用中文翻译。
    
    (目前节点本身暂未实现自带翻译功能) 



## 使用方法

### OpenAI API 节点

- 用于与 OpenAI 格式的 API 进行交互，支持oneapi/newapi这类中转 API。
- 支持文本和图像输入。
- 可以使用环境变量`OPENAI_API_KEY`和`OPENAI_API_BASE`来设置 API 密钥和 base url。
  
#### 注意：请不要在分享工作流时在节点中包含 API 密钥和base url。

### OpenAI 连续对话节点

-  允许在同一个节点内进行多轮对话，并在节点内部维护对话历史记录。
-  每次输入新的用户消息后，节点会将新的消息添加到历史记录中，并将其与之前的对话一起发送到 OpenAI API。
-  可以使用 "清除历史记录" 选项来重置对话历史。
-  支持设置系统提示，以便在每次对话开始时为模型提供上下文。
-  也支持文本和图像输入。

### 历史消息节点

- 创建包含最多 4 轮对话的历史消息。
- 每轮对话可以选择角色（用户、助手、系统）和输入内容。
- 空的内容不会被包含在输出中。

### 合并历史消息节点

- 合并来自最多 4 个历史消息节点的输出。
- 支持嵌套，允许创建更长的对话链，比如把酒馆（SillyTavern）的预设塞进去。

## 贡献

欢迎贡献！如果您有任何建议或改进，请提交 issue 或 pull request。

## 许可

本项目采用 MIT 许可证。详情请见 [LICENSE](LICENSE) 文件。

## 联系

如有任何问题或建议，请通过 GitHub issues 与我联系。

---

注意：这个项目仍在开发中，功能可能会随时间而扩展或更改。
