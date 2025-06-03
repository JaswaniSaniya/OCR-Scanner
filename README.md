
# 📄 OCR Scanner Showcase

A comparative project demonstrating the capabilities of various OCR (Optical Character Recognition) tools, with a focus on multilingual and high-accuracy text extraction. This project explores and benchmarks the performance of modern OCR models across different document types, languages, and formatting challenges.

---

## 🚀 Featured OCR Models

### 🔍 1. EasyOCR
- **Description**: Open-source OCR library built on PyTorch.
- **Strengths**: 
  - Supports over 80 languages.
  - Lightweight and easy to use.
  - Decent accuracy for Latin-script languages.
- **Use case**: Ideal for quick deployments, smaller projects, and low-resource environments.

### 🧠 2. Mistral OCR (via instruction-tuned models)
- **Description**: OCR capabilities powered by LLMs such as Mistral-7B, fine-tuned on instruction-based tasks.
- **Strengths**:
  - Can process unstructured images with instruction prompts.
  - Learns OCR behavior in-context, making it flexible for novel layouts.
- **Use case**: Suitable for custom workflows, low-resource multilingual OCR with fine-tuning support.

### 🌐 3. Qwen-VL / Qwen OCR
- **Description**: Alibaba's vision-language model series with OCR capability.
- **Strengths**:
  - Multimodal input handling.
  - Competitive accuracy on complex layouts and East Asian scripts.
- **Use case**: Research and advanced document understanding tasks.

---

## 🌍 Best Models for Multilingual, High-Accuracy OCR

These models stand out when working with documents in multiple languages and varying layouts:

| Model              | Multilingual Support | Accuracy | Notes |
|--------------------|----------------------|----------|-------|
| **EasyOCR**        | ✅ (80+ languages)   | ⭐⭐       | Fast, lightweight |
| **Mistral**        | ✅ (context-based)   | ⭐⭐⭐⭐     | Requires prompting or fine-tuning |
| **OpenAI GPT-4o**  | ✅ (extensive)       | ⭐⭐⭐⭐⭐    | Industry-leading OCR + reasoning |
| **Claude Sonnet**  | ✅                   | ⭐⭐⭐⭐     | Robust vision-language understanding |

> ⚠️ **Note**: GPT-4o and Claude Sonnet are multimodal LLMs — they don't do OCR in the traditional sense but excel at extracting and interpreting text from images, especially in complex, low-quality, or multilingual contexts.

---

## 📁 Structure

