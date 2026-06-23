 Benchmarking Transformer Topologies and Recurrent Neural Networks for Text Classification

An end-to-end deep learning framework designed for **comparative linguistic sentiment analysis**, engineered with an integrated pipeline utilizing **Streamlit, PyTorch, and Hugging Face Transformers**.

## 🔬 Research & Project Overview

Linguistic sequence modeling presents complex challenges in capturing long-range contextual dependencies and semantic nuances. This project introduces a comparative research framework that evaluates traditional recurrent architectures against modern attention-based transformer models using the large-scale IMDb movie review dataset. 

By isolating structural differences between recurrent token-by-token processing and parallelized self-attention mechanisms, this repository serves as an experimental benchmark for model generalization, computational efficiency, and semantic accuracy in text classification.

### Key Technical Contributions:
- **Architecture Benchmarking:** Side-by-side empirical comparison between sequential Long Short-Term Memory (LSTM) networks and bidirectional transformer representations (BERT).
- **Linguistic Pipeline Optimization:** Implementation of advanced tokenization strategies, text normalization pipelines, and padded sequence configurations.
- **Interactive Evaluation Hub:** A functional Streamlit client allowing for real-time model comparative execution.

---

## 🏗️ System Architecture & Data Flow

Use code with caution.[Raw Text Entry] ──> [Text Preprocessing & Tokenization]│┌──────────────────────────────┴──────────────────────────────┐▼                                                             ▼[Sequential LSTM Network]                                   [Pre-trained BERT Encoder]│                                                             │└──────────────────────────────┬──────────────────────────────┘▼[Binary Cross-Entropy Loss / Softmax]│▼[Streamlit Real-Time Metric & Sentiment Interface]
---

## 🛠️ Computational Tech Stack

- **Deep Learning Frameworks:** PyTorch, Hugging Face Transformers
- **Linguistic & Tokenization Engines:** NLTK, Regex, BERT Tokenizer, Pickle (`tokenizer.p`)
- **Data Engineering Suite:** NumPy, Pandas, Scikit-learn
- **Interface & Dashboard Engine:** Streamlit Interface Framework

---

## 📂 Repository Structure

```text
sentiment_analysis/
│
├── app.py              # Application Interface (Streamlit UI & Execution Logic)
├── tokenizer.p         # Serialized Tokenizer Object for Sequential Text Processing
├── .gitignore          # Version Control Exclusion Configurations
│
├── data/               # Corpus Management Directory
│   ├── imdb_train.csv  # Supervised Training Partition (25k instances)
│   └── imdb_test.csv   # Unseen Evaluation Partition (25k instances)
│
├── notebooks/          # Experimental Prototyping & Pipeline Development
│   ├── BERT_FineTuning.ipynb  # Transformer Adaptation & Attention Mapping
│   └── LSTM_Training.ipynb    # Recurrent Baseline Optimization Loops
│
└── report.pdf          # Comprehensive Formal Scientific & Empirical Research Report
```

---

## 🚀 Core Research Features & Capabilities

- **Comparative Inference Panel:** Multi-model inference system capable of processing a single textual string simultaneously through both LSTM and BERT backbones.
- **Empirical Metrics Dashboard:** Real-time generation of performance statistics, highlighting classification probabilities, confidence intervals, and processing latencies.
- **Robust Text Preprocessing Pipeline:** Automated handling of alphanumeric filtering, lower-casing, vocabulary truncation, and tensor padding configurations.
- **Comprehensive Documentation:** Accompanied by a formal research report detailing hyperparameter adjustments, loss curves, and architectural findings.

---

## 📊 Experimental Results & Evaluation

The models were systematically evaluated across identical test splits using standard classification metrics:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LSTM (Baseline)** | 0.824 | 0.819 | 0.830 | 0.824 | ~12ms |
| **BERT Fine-Tuned** | 0.912 | 0.908 | 0.915 | 0.911 | ~45ms |
