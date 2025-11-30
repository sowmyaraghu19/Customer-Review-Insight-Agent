import io
import textwrap
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from wordcloud import WordCloud
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from orchestrator import ReviewInsightOrchestrator
from memory import ShortTermMemory


# ---------- Helpers ----------

def compute_rating_stats(metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
    ratings = []
    for m in metadatas:
        try:
            r = float(str(m.get("reviews.rating", m.get("rating", 0))).strip())
            if 1 <= r <= 5:
                ratings.append(r)
        except:
            continue

    if not ratings:
        return {"ratings": [], "counts": {}, "avg": None}

    counts = {i: 0 for i in range(1, 6)}
    for r in ratings:
        counts[int(round(r))] += 1

    avg = sum(ratings) / len(ratings)
    return {"ratings": ratings, "counts": counts, "avg": avg}


def plot_rating_histogram(stats: Dict[str, Any]):
    if not stats["ratings"]:
        st.info("No rating data available.")
        return
    fig, ax = plt.subplots()
    xs = list(stats["counts"].keys())
    ys = list(stats["counts"].values())
    ax.bar(xs, ys)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Rating Distribution")
    st.pyplot(fig)


def plot_sentiment_pie(stats: Dict[str, Any]):
    if not stats["ratings"]:
        st.info("No sentiment data available.")
        return

    pos = sum(c for r, c in stats["counts"].items() if r >= 4)
    neu = stats["counts"][3]
    neg = sum(c for r, c in stats["counts"].items() if r <= 2)

    labels = ["Positive (4–5★)", "Neutral (3★)", "Negative (1–2★)"]
    sizes = [pos, neu, neg]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%")
    ax.set_title("Sentiment Breakdown")
    st.pyplot(fig)


def plot_wordcloud(docs: List[str]):
    docs = [d for d in docs if isinstance(d, str) and len(d.split()) > 3]
    if not docs:
        st.info("No meaningful text available for word cloud.")
        return
    text = " ".join(docs)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


def generate_pdf(summary: str, analysis: str) -> io.BytesIO:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    def draw_block(title: str, body: str, y_start: int):
        text_obj = c.beginText(40, y_start)
        text_obj.setFont("Helvetica-Bold", 12)
        text_obj.textLine(title)
        text_obj.setFont("Helvetica", 10)
        wrapped = textwrap.wrap(body, width=90)
        for line in wrapped:
            text_obj.textLine(line)
        c.drawText(text_obj)

    draw_block("Summary", summary or "N/A", height - 60)
    draw_block("Insights & Sentiment", analysis or "N/A", height - 260)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# ---------- Streamlit Setup ----------

st.set_page_config(page_title="Customer Review Insight Agent", layout="wide")

if "short_memory" not in st.session_state:
    st.session_state.short_memory = ShortTermMemory()

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = ReviewInsightOrchestrator()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_query" not in st.session_state:
    st.session_state.user_query = ""

if "last_result" not in st.session_state:
    st.session_state.last_result = None

orchestrator = st.session_state.orchestrator
short_memory = st.session_state.short_memory


# ---------- Sidebar Controls ----------

st.sidebar.title("⚙️ Controls")

mode = st.sidebar.radio("Mode", ["Single query", "Chat"])

product_focus = st.sidebar.selectbox(
    "Product focus (optional)",
    ["Auto-detect", "Kindle", "Fire Tablet", "Fire TV Stick", "Echo", "Echo Dot", "Other"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("✨ Quick Prompts")

quick_prompts = {
    "Overview": "Summarize customer opinions about the Kindle in 5 bullet points.",
    "Pros & Cons": "List the top pros and cons customers mention about the Kindle.",
    "Battery life": "What do customers think about the battery life of the Kindle?",
    "Complaints": "What are the biggest customer complaints about the Kindle?",
    "Compare Kindle vs Fire Tablet": "Compare customer sentiment between the Kindle and the Fire Tablet."
}

for label, prompt in quick_prompts.items():
    if st.sidebar.button(label):
        st.session_state.user_query = prompt

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Short-term Memory")
st.sidebar.json(short_memory.get_all())


# ---------- Main App ----------

st.title("📊 Customer Review Insight Agent")
st.write("Ask anything about customer reviews, aspects, pros/cons, or sentiment trends.")


def run_query(query: str):
    if product_focus != "Auto-detect":
        query = f"For product {product_focus}, {query}"

    with st.spinner("Analyzing reviews..."):
        return orchestrator.run(user_query=query, short_memory=short_memory)


# ========== Mode: SINGLE QUERY ==========

if mode == "Single query":
    user_query = st.text_input(
        "Enter your query:",
        value=st.session_state.user_query,
        key="single_query_input",
    )

    if st.button("Analyze") and user_query:
        st.session_state.user_query = user_query
        result = run_query(user_query)
        st.session_state.last_result = result

# ========== Mode: CHAT ==========

else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    chat_query = st.chat_input("Ask about customer reviews…")
    if chat_query:
        st.session_state.chat_history.append({"role": "user", "content": chat_query})
        with st.chat_message("user"):
            st.markdown(chat_query)

        result = run_query(chat_query)

        answer_text = (
            f"### Summary\n{result['summary']}\n\n"
            f"### Insights\n{result['analysis']}"
        )
        st.session_state.chat_history.append({"role": "assistant", "content": answer_text})
        with st.chat_message("assistant"):
            st.markdown(answer_text)

        st.session_state.last_result = result


# ---------- RESULTS (Collapsible Sections) ----------

result = st.session_state.last_result
if result:
    summary = result["summary"]
    analysis = result["analysis"]
    plan = result["plan"]
    docs = result["docs"]
    metadatas = result["metadatas"]

    # Always visible summary
    st.subheader("📝 Summary")
    st.markdown(summary)

    # PDF Report
    with st.expander("📄 Download Insight Report (PDF)", expanded=False):
        pdf_buffer = generate_pdf(summary, analysis)
        st.download_button(
            "Download PDF",
            data=pdf_buffer,
            file_name="review_insight_report.pdf",
            mime="application/pdf",
        )

    # Planner
    with st.expander("🧭 Planner Understanding", expanded=False):
        st.json(plan)

    # Insights
    with st.expander("📈 Insights & Sentiment", expanded=False):
        st.markdown(analysis)

    # Sample Reviews
    with st.expander("🔍 Sample Retrieved Reviews", expanded=False):
        for doc, meta in list(zip(docs, metadatas))[:5]:
            st.markdown(
                f"**Product:** {meta.get('name', 'N/A')}  \n"
                f"**Rating:** {meta.get('reviews.rating', 'N/A')} ⭐"
            )
            st.markdown(doc)
            st.markdown("---")

    # Visual Analytics
    with st.expander("📊 Visual Analytics (Charts)", expanded=False):
        stats = compute_rating_stats(metadatas)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if stats["avg"]:
                st.metric("Average Rating", f"{stats['avg']:.2f} ★")
            else:
                st.write("No rating data")

        with col_b:
            st.markdown("**Rating Distribution**")
            plot_rating_histogram(stats)

        with col_c:
            st.markdown("**Sentiment Breakdown**")
            plot_sentiment_pie(stats)

    # Word Cloud
    with st.expander("☁️ Word Cloud (Top Terms)", expanded=False):
        plot_wordcloud(docs)
