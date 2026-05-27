import unittest

from app.query_process.retrieval.query_profile import build_query_profile


class TestQueryProfile(unittest.TestCase):
    def test_exact_model_or_param(self):
        profile = build_query_profile("B730 参数是多少", "B730 参数是多少", ["华为擎云B730"])
        self.assertEqual(profile.query_type, "exact_model_or_param")
        self.assertFalse(profile.retrieval_config.use_hyde)
        self.assertGreater(profile.retrieval_config.sparse_weight, profile.retrieval_config.dense_weight)

    def test_semantic_howto(self):
        profile = build_query_profile("如何安装设备", "如何安装设备", ["HAK180"])
        self.assertEqual(profile.query_type, "semantic_howto")
        self.assertFalse(profile.retrieval_config.use_hyde)
        self.assertGreater(profile.retrieval_config.dense_weight, profile.retrieval_config.sparse_weight)

    def test_troubleshooting(self):
        profile = build_query_profile("设备报错怎么排查", "设备报错怎么排查", ["HAK180"])
        self.assertEqual(profile.query_type, "troubleshooting")
        self.assertTrue(profile.retrieval_config.use_hyde)
        self.assertTrue(profile.retrieval_config.use_rerank)

    def test_ambiguous(self):
        profile = build_query_profile("它怎么用", "它怎么用", [])
        self.assertEqual(profile.query_type, "ambiguous")
        self.assertFalse(profile.retrieval_config.use_hyde)
        self.assertFalse(profile.retrieval_config.use_rerank)


if __name__ == "__main__":
    unittest.main()
