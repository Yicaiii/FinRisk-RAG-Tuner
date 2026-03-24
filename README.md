# FinRisk-RAG-Tuner: Enhancing LLM Financial Insight via RAG & QLoRA

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://python.langchain.com/)
[![Model](https://img.shields.io/badge/Model-Llama--3--8B-orange.svg)](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

## 📌 Project Overview
This project addresses the limitations of Large Language Models (LLMs) in processing highly specialized and time-sensitive financial data (e.g., SEC Filings, 10-K/10-Q reports). By implementing a **Retriever-Augmented Generation (RAG)** pipeline and performing **QLoRA fine-tuning**, I achieved a significant improvement in factual accuracy and domain-specific relevance.

**Key Results:**
* **+30% Retrieval Relevance** compared to vanilla semantic search.
* **+20% Accuracy** on financial Q&A benchmarks.
* **Reduced Hallucinations** in quantitative data extraction (revenue, EBITDA, etc.).

---

## 📂 Dataset
I used the **[Financial Reports QA Dataset](https://www.kaggle.com/datasets/ahmedsta/data-retreiver)** from Kaggle, which includes:
- Complex financial statements from public companies.
- Context-Answer pairs for evaluation.
- Domain-specific terminology and multi-table structures.

---

## 🛠️ Technical Architecture

### 1. RAG Pipeline (The "Knowledge" Boost)
Built using **LangChain** and **FAISS**:
- **Advanced Chunking**: Implemented `RecursiveCharacterTextSplitter` with overlapping to preserve context in long financial clauses.
- **Embedding**: Utilized `BAAI/bge-small-en-v1.5` for high-efficiency vectorization.
- **Reranking**: Added a Cross-Encoder reranking step to prioritize the top-k most relevant financial contexts.

### 2. QLoRA Fine-Tuning (The "Expertise" Boost)
Fine-tuned **Llama-3-8B** on a T4 GPU (Kaggle/Colab) to align the model with financial reporting styles:
- **Optimization**: Used 4-bit quantization (bitsandbytes) to reduce VRAM usage.
- **PEFT (LoRA)**: Targeted `q_proj` and `v_proj` modules to adapt the model with minimal trainable parameters.

---

## 📊 Benchmark Results

| Methodology | Accuracy (Top-1) | Perplexity | Hallucination Rate |
| :--- | :--- | :--- | :--- |
| **Base Llama-3** | 42.5% | 14.2 | High |
| **RAG Only** | 68.2% | N/A | Low |
| **RAG + Fine-tuned (Final)** | **74.8%** | **8.5** | **Minimal** |

> *Visualizations of Loss Curves and Retrieval Hit Rates are available in the `/plots` directory.*

---

## 🚀 How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
