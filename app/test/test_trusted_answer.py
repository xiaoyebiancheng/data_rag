import unittest

from app.query_process.answering.trusted_answer import (
    assess_answer_gate,
    build_sources_from_reranked_docs,
    extract_image_evidence,
    verify_answer_support,
)


class TestTrustedAnswer(unittest.TestCase):
    def test_build_sources_from_local_docs(self):
        sources = build_sources_from_reranked_docs(
            [
                {
                    "source": "local",
                    "chunk_id": 101,
                    "doc_id": "doc_1",
                    "file_title": "手册A",
                    "title": "参数说明",
                    "parent_title": "第一章",
                    "item_name": "HAK180",
                    "version": 2,
                    "score": 0.91,
                    "rerank_score": 0.91,
                    "text": "HAK180 支持 220V 电源输入",
                },
                {"source": "web", "chunk_id": "", "text": "忽略"},
            ]
        )
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].doc_id, "doc_1")

    def test_gate_refuse_when_no_sources(self):
        confidence, need_clarification, refusal_reason = assess_answer_gate({}, [], 0.6)
        self.assertTrue(refusal_reason)
        self.assertLess(confidence, 0.2)
        self.assertTrue(need_clarification)

    def test_verify_answer_support(self):
        sources = build_sources_from_reranked_docs(
            [
                {
                    "source": "local",
                    "chunk_id": 101,
                    "doc_id": "doc_1",
                    "file_title": "手册A",
                    "title": "参数说明",
                    "parent_title": "第一章",
                    "item_name": "HAK180",
                    "version": 2,
                    "score": 0.91,
                    "rerank_score": 0.91,
                    "text": "HAK180 支持 220V 电源输入，额定功率 500W。",
                }
            ]
        )
        unsupported = verify_answer_support("HAK180 支持 220V 电源输入。", sources)
        self.assertEqual(unsupported, [])

    def test_source_keeps_image_urls_and_warnings(self):
        sources = build_sources_from_reranked_docs(
            [
                {
                    "source": "local",
                    "chunk_id": 102,
                    "doc_id": "doc_2",
                    "score": 0.8,
                    "rerank_score": 0.8,
                    "text": "![低置信图片摘要：无法可靠识别：画面模糊](http://minio/img.png)",
                }
            ]
        )
        self.assertEqual(sources[0].image_urls, ["http://minio/img.png"])
        self.assertTrue(sources[0].image_summary_warnings)

    def test_extract_image_evidence(self):
        image_urls, warnings = extract_image_evidence(
            "![设备接口图](http://minio/a.jpg)\n![图片摘要失败：图片摘要生成失败，请以原图为准](http://minio/b.jpg)"
        )
        self.assertEqual(image_urls, ["http://minio/a.jpg", "http://minio/b.jpg"])
        self.assertEqual(len(warnings), 1)


if __name__ == "__main__":
    unittest.main()
