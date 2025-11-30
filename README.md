# 📊 Customer Review Insight Agent

An AI-powered multi-agent system that analyzes customer reviews using retrieval-augmented generation (RAG), vector embeddings, semantic search, and structured LLM-powered planning.  
Built with **Streamlit**, **ChromaDB**, **SentenceTransformers**, and a custom **multi-agent orchestration pipeline**.

---

## 🚀 Features

- 🤖 Multi-agent architecture (planner, retriever, summarizer, sentiment, insight)
- 🔍 Semantic search using ChromaDB vector database
- 📄 Clean summaries of customer reviews
- 📈 Visual analytics: rating histogram, sentiment pie chart, word cloud
- 📝 Expandable insights sections (collapsible UI)
- 💬 Chat mode with short-term memory
- 📤 Upload custom datasets
- 📄 PDF insight report generation
- 🎯 Quick action buttons for common queries

---

## 📁 Project Structure
Dataset: https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products?resource=download
```
Customer-Review-Insight-Agent/
│
├── app.py                     # Streamlit UI
├── orchestrator.py            # Coordinates all agents
├── memory.py                  # Short-term memory
│
├── agents/
│   ├── planner_agent.py
│   ├── retriever_agent.py
│   ├── summarizer_agent.py
│   ├── sentiment_agent.py
│   └── insight_agent.py
│
├── preprocess.py              # Clean dataset
├── build_vectorstore.py       # Build ChromaDB vectorstore
├── requirements.txt
└── README.md
```

> Note: `data/` and `vectorstore/` are excluded from Git via `.gitignore`.

## 🔧 Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/<your-username>/Customer-Review-Insight-Agent.git
cd Customer-Review-Insight-Agent
```

### 2️⃣ Create & activate environment

```
conda create -n review-agent python=3.11 -y
conda activate review-agent
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Build the vector database

```
Place your dataset CSV files into a folder named `data/` (ignored by Git).
```

Run:
```
python preprocess.py
python build_vectorstore.py
```

This will:

* Clean text
* Embed reviews
* Build the ChromaDB vector database

### 5️⃣ Launch Streamlit app
streamlit run app.py

Visit:
👉 [http://localhost:8501/](http://localhost:8501/)

---

## 🧪 Example Queries

* “Give me 5 bullet points about what users like about the Kindle.”
* “What are the biggest complaints about Fire TV Stick?”
* “Compare sentiment for Kindle vs Fire Tablet.”
* “What do customers say about Kindle battery life?”
* “Summarize Kindle reviews in one paragraph.”

---

## 📈 Visual Analytics Included

* Rating distribution bar chart
* Sentiment breakdown pie chart
* Auto-generated word cloud
* Expandable insights: pros/cons, patterns, complaints

---

## 📄 PDF Report

Click “Download Insight Report” to generate a PDF containing:

* Summary
* Insights
* Sentiment analysis

---

## 🤝 Contributing

Pull requests are welcome.
