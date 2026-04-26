"""
离线评测模块
"""

from app.evaluation.dataset_schema import EvalSample
from app.evaluation.evaluator import RAGEvaluator

__all__ = ["EvalSample", "RAGEvaluator"]
