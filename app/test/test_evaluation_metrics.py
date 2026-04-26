import unittest

from app.evaluation.metrics import hit_at_k, mean_reciprocal_rank, ndcg_at_k, percentile, recall_at_k


class EvaluationMetricsTestCase(unittest.TestCase):
    def test_hit_at_k(self):
        self.assertEqual(hit_at_k(["a", "b", "c"], ["x", "b"], 2), 1.0)
        self.assertEqual(hit_at_k(["a", "b", "c"], ["x", "y"], 2), 0.0)
        self.assertIsNone(hit_at_k(["a"], [], 1))

    def test_recall_at_k(self):
        self.assertAlmostEqual(recall_at_k(["a", "b", "c"], ["b", "c", "d"], 3), 2 / 3)
        self.assertEqual(recall_at_k(["a", "b"], ["x"], 2), 0.0)
        self.assertIsNone(recall_at_k(["a"], [], 1))

    def test_mrr(self):
        self.assertEqual(mean_reciprocal_rank(["x", "b", "c"], ["b", "c"]), 0.5)
        self.assertEqual(mean_reciprocal_rank(["x", "y"], ["b", "c"]), 0.0)
        self.assertIsNone(mean_reciprocal_rank(["x"], []))

    def test_ndcg_at_k(self):
        self.assertAlmostEqual(ndcg_at_k(["a", "b", "c"], ["a", "c"], 3), 0.9197207891, places=6)
        self.assertEqual(ndcg_at_k(["x", "y", "z"], ["a", "b"], 3), 0.0)
        self.assertIsNone(ndcg_at_k(["x"], [], 1))

    def test_percentile(self):
        values = [10.0, 20.0, 30.0, 40.0]
        self.assertEqual(percentile(values, 0), 10.0)
        self.assertEqual(percentile(values, 100), 40.0)
        self.assertAlmostEqual(percentile(values, 50), 25.0)


if __name__ == "__main__":
    unittest.main()
