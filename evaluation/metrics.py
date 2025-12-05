import json
from rouge_score import rouge_scorer
from bert_score import score as bertscore
from orchestrator import ReviewInsightOrchestrator
from memory import ShortTermMemory


# ===============================
# Evaluation Set
# ===============================
EVAL_SET = [
    {
        "question": "What do customers like most about the Kindle?",
        "reference": "Customers appreciate the readability, long battery life, lightweight design, and ease of use."
    },
    {
        "question": "What are the main complaints about the Fire Tablet?",
        "reference": "Customers complain about slow performance, low-quality screen, and limited app functionality."
    },
    {
        "question": "Summarize common opinions about Kindle battery life.",
        "reference": "The Kindle's battery life is praised for lasting weeks on a single charge."
    },
    {
        "question": "What improvements do users want for the Kindle Paperwhite?",
        "reference": "Users want faster performance, better navigation controls, and improved screen responsiveness."
    }
]


# ===============================
# Metrics
# ===============================

def compute_rouge(pred: str, ref: str):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def compute_bertscore(pred: str, ref: str):
    P, R, F1 = bertscore(
        [pred],
        [ref],
        lang="en",
        verbose=False
    )
    return float(P[0]), float(R[0]), float(F1[0])


# ===============================
# Main Evaluation
# ===============================

def run_evaluation():
    orchestrator = ReviewInsightOrchestrator()
    memory = ShortTermMemory()

    results = []

    print("\n🚀 Running Evaluation: ROUGE + BERTScore\n")

    for item in EVAL_SET:
        q = item["question"]
        ref = item["reference"]

        print(f"📌 Question: {q}")

        # Run your full agent pipeline
        output = orchestrator.run(q, memory)
        pred = output.get("summary", "").strip()

        if not pred:
            print("⚠ No prediction generated. Skipping.\n")
            continue

        # ---- Compute Metrics ----
        rouge_scores = compute_rouge(pred, ref)
        bert_precision, bert_recall, bert_f1 = compute_bertscore(pred, ref)

        print(f"   ✔ ROUGE-1:  {rouge_scores['rouge1']:.4f}")
        print(f"   ✔ ROUGE-L:  {rouge_scores['rougeL']:.4f}")
        print(f"   ✔ BERT F1:  {bert_f1:.4f}\n")

        results.append({
            "question": q,
            "reference": ref,
            "prediction": pred,
            "rouge1": rouge_scores["rouge1"],
            "rougeL": rouge_scores["rougeL"],
            "bertscore_precision": bert_precision,
            "bertscore_recall": bert_recall,
            "bertscore_f1": bert_f1,
        })

    # Save JSON
    with open("evaluation_results_combined.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Save Markdown
    with open("evaluation_report_combined.md", "w", encoding="utf-8") as f:
        f.write("# 📊 Evaluation Report (ROUGE + BERTScore)\n\n")
        f.write("This report compares ROUGE (lexical) and BERTScore (semantic) metrics.\n\n")

        for r in results:
            f.write(f"## ❓ {r['question']}\n")
            f.write(f"**Reference:** {r['reference']}\n\n")
            f.write(f"**Prediction:** {r['prediction']}\n\n")
            f.write("### Scores\n")
            f.write(f"- ROUGE-1: `{r['rouge1']:.4f}`\n")
            f.write(f"- ROUGE-L: `{r['rougeL']:.4f}`\n")
            f.write(f"- BERTScore Precision: `{r['bertscore_precision']:.4f}`\n")
            f.write(f"- BERTScore Recall: `{r['bertscore_recall']:.4f}`\n")
            f.write(f"- BERTScore F1: `{r['bertscore_f1']:.4f}`\n")
            f.write("\n---\n")

    print("✅ Evaluation complete.")
    print("📄 Saved: evaluation_results_combined.json")
    print("📝 Saved: evaluation_report_combined.md")


if __name__ == "__main__":
    run_evaluation()
