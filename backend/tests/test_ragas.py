"""
RAGAS Evaluation Script

Run with: python -m tests.test_ragas
Requires: OPENAI_API_KEY set in environment

This script evaluates the RAG pipeline using RAGAS metrics:
- Context Precision
- Context Recall
- Faithfulness
- Answer Relevancy
"""

import os
import json
import logging
from typing import List, Dict
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_eval_dataset(test_cases: List[Dict]) -> Dataset:
    return Dataset.from_dict({
        "question": [tc["question"] for tc in test_cases],
        "answer": [tc.get("answer", "") for tc in test_cases],
        "contexts": [tc["contexts"] for tc in test_cases],
        "ground_truth": [tc.get("ground_truth", "") for tc in test_cases],
    })


def evaluate_with_ragas(dataset: Dataset) -> dict:
    try:
        from ragas import evaluate
        from ragas.metrics import (
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_correctness,
        )

        result = evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
                answer_correctness,
            ],
        )
        return result
    except ImportError:
        logger.error("RAGAS not installed. Install with: pip install ragas")
        return {}


def run_evaluation(test_cases_path: str = None):
    if test_cases_path and os.path.exists(test_cases_path):
        with open(test_cases_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
    else:
        test_cases = [
            {
                "question": "Tài liệu này nói về chủ đề gì?",
                "contexts": ["Tài liệu hướng dẫn sử dụng phần mềm quản lý dự án."],
                "answer": "Tài liệu này nói về hướng dẫn sử dụng phần mềm quản lý dự án.",
                "ground_truth": "Tài liệu hướng dẫn sử dụng phần mềm quản lý dự án.",
            },
        ]

    dataset = create_eval_dataset(test_cases)
    logger.info(f"Evaluating {len(test_cases)} test cases...")

    results = evaluate_with_ragas(dataset)

    if results:
        logger.info("Evaluation Results:")
        for metric_name, score in results.items():
            logger.info(f"  {metric_name}: {score:.4f}")

    return results


if __name__ == "__main__":
    run_evaluation()
