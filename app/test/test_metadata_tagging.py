import unittest

from app.import_process.metadata_tagging import enrich_document_and_chunks, infer_doc_type, infer_section_type


class TestMetadataTagging(unittest.TestCase):
    def test_infer_doc_type(self):
        doc_type = infer_doc_type(
            "hak180产品安全手册",
            [{"title": "## 注意事项", "content": "请断电后维护设备"}],
        )
        self.assertEqual(doc_type, "manual")

    def test_infer_section_type(self):
        section_type = infer_section_type("## 故障排查", "设备异常时请先检查电源")
        self.assertEqual(section_type, "troubleshooting")

    def test_enrich_document_and_chunks(self):
        doc_metadata, chunks = enrich_document_and_chunks(
            "B730用户指南",
            "华为擎云B730",
            [{"title": "## 安装步骤", "content": "先连接电源再开机", "file_title": "B730用户指南"}],
        )
        self.assertEqual(doc_metadata["doc_type"], "guide")
        self.assertEqual(chunks[0]["section_type"], "installation")
        self.assertEqual(chunks[0]["visibility"], "tenant")


if __name__ == "__main__":
    unittest.main()
