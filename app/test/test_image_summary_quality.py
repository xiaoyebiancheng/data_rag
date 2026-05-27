import unittest

from app.import_process.agent.nodes.node_md_img import (
    SUMMARY_STATUS_FAILED,
    SUMMARY_STATUS_LOW_CONFIDENCE,
    SUMMARY_STATUS_OK,
    _build_markdown_image_replacement,
    normalize_image_summary,
)


class TestImageSummaryQuality(unittest.TestCase):
    def test_normal_summary_is_ok(self):
        result = normalize_image_summary("设备背面接口示意图，标注 HDMI、VGA 和电源接口位置")
        self.assertEqual(result["status"], SUMMARY_STATUS_OK)
        self.assertGreater(result["confidence"], 0.8)

    def test_generic_summary_is_low_confidence(self):
        result = normalize_image_summary("图片描述")
        self.assertEqual(result["status"], SUMMARY_STATUS_LOW_CONFIDENCE)
        self.assertIn("generic_or_too_short", result["reason"])

    def test_empty_summary_is_failed(self):
        result = normalize_image_summary("")
        self.assertEqual(result["status"], SUMMARY_STATUS_FAILED)
        self.assertEqual(result["confidence"], 0.0)

    def test_low_confidence_markdown_contains_quality_marker(self):
        result = normalize_image_summary("无法可靠识别：图片过于模糊")
        markdown = _build_markdown_image_replacement(result, "http://minio/image.png")
        self.assertIn("低置信图片摘要", markdown)
        self.assertIn("image_summary_quality", markdown)
        self.assertIn("source_url: http://minio/image.png", markdown)


if __name__ == "__main__":
    unittest.main()
