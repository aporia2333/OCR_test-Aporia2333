# OCR / LLM Workbench (Streamlit Prototype)

这是一个可运行的 Streamlit 原型，支持：

- 批量上传文件
- 三种模式切换：纯 OCR / 纯大模型 / OCR + 大模型
- 默认提示词模板 + 自定义提示词编辑
- 结果预览
- 单文件下载与 ZIP 批量下载
- 服务端环境变量方式接入 Google / OpenAI，不在网页中直接填写 API Key

## 1. 安装

```bash
pip install -r requirements.txt
```

## 2. 配置 API Key

### OpenAI

```bash
export OPENAI_API_KEY="your_api_key_here"
```

### Google OCR

二选一：

```bash
export GOOGLE_API_KEY="your_google_api_key"
```

或：

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

Windows PowerShell 示例：

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
$env:GOOGLE_API_KEY="your_google_api_key"
```

## 3. 运行

```bash
streamlit run app.py
```

## 4. 当前实现说明

### 已实现
- 完整 Streamlit 页面
- 多文件上传
- 模式切换
- Prompt 模板与自定义 prompt
- 服务连接状态区（Google / OpenAI）
- 启用 / 禁用开关
- Mock OCR
- Mock LLM
- OpenAI 文本后处理
- txt / markdown / json 导出

### 当前保留扩展接口
在 `services/ocr.py` 中接入真实 OCR：
- Google Document AI
- Azure Document Intelligence
- AWS Textract

在 `services/llm.py` 中可继续扩展：
- 直接把图片/PDF 作为多模态输入交给模型
- 更复杂的 JSON Schema
- 重试与限流

## 5. 推荐迭代

- 增加 Google OCR 的真实 SDK 接入
- 增加任务历史记录
- 增加 ZIP 原文件 + 结果统一打包
- 增加数据库与对象存储
- 增加后台任务队列
