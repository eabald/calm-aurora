import unittest

from cidoc_rag.prompting.modes import detect_mode


class ModeDetectionTests(unittest.TestCase):
    def test_detects_mapping_by_keywords(self):
        self.assertEqual(detect_mode("Map this table fields to CIDOC"), "mapping")

    def test_detects_mapping_by_json(self):
        query = '{"table": "authors", "fields": ["name"]}'
        self.assertEqual(detect_mode(query), "mapping")

    def test_detects_qa_default(self):
        self.assertEqual(detect_mode("What is E21?"), "qa")


if __name__ == "__main__":
    unittest.main()
