# OCR / LLM Workbench (DeepSeek + PaddleOCR 版)

这是一个简化后的 Streamlit 原型，只保留你当前真正需要的两条外部能力：

- PaddleOCR Layout Parsing
- DeepSeek

并支持：

- 批量上传文件
- 三种模式切换：纯 OCR / 纯大模型 / OCR + 大模型
- 默认提示词模板 + 自定义提示词编辑
- 结果预览
- 单文件下载与 ZIP 批量下载
- 服务端环境变量 / Streamlit secrets 方式接入 API

## 1. 安装

```bash
pip install -r requirements.txt
```

## 2. 本地环境变量

```bash
export PADDLEOCR_API_URL="your_paddleocr_api_url"
export PADDLEOCR_TOKEN="your_paddleocr_token"
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

如需自定义 DeepSeek OpenAI-compatible 地址：

```bash
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

## 3. Streamlit Community Cloud secrets

```toml
PADDLEOCR_API_URL="your_paddleocr_api_url"
PADDLEOCR_TOKEN="your_paddleocr_token"
DEEPSEEK_API_KEY="your_deepseek_api_key"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

## 4. 运行

```bash
streamlit run app.py
```

## 5. 当前实现说明

### 已实现
- PaddleOCR Layout Parsing 主流程
- DeepSeek 文本后处理
- Mock OCR / Mock LLM 兜底
- txt / markdown / json 导出
- 批量文件 ZIP 打包下载

### 说明
- DeepSeek 这版改成了 `chat.completions` 调用，更适合当前 DeepSeek 的 OpenAI-compatible 接口
- 纯大模型模式下，本地原型默认还是基于文本输入，不直接上传图片给模型
- PaddleOCR 返回的图片资源目前未自动下载打包
