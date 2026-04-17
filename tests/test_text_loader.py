import tempfile
import unittest
from pathlib import Path

from cidoc_rag.ingestion.loader import load_raw_data
from cidoc_rag.ingestion.normalize import normalize_entry


class TextLoaderTests(unittest.TestCase):
    def test_markdown_is_split_into_multiple_entries(self):
        content = """
# Intro
This is an intro paragraph about CIDOC CRM usage in institutions.

## Modeling
Use explicit events to preserve provenance and temporal semantics.

## Mapping
Map local schema entities to CIDOC classes and properties.
""".strip()

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "docs.md"
            file_path.write_text(content, encoding="utf-8")

            entries = load_raw_data(str(file_path))
            self.assertGreaterEqual(len(entries), 3)

    def test_prose_chunk_not_misclassified_by_embedded_e_id(self):
        raw_entry = {
            "raw_text": "This documentation references E21 as an example within prose.",
            "_source_file": "data/raw/docs/docs.md",
        }

        normalized = normalize_entry(raw_entry)
        self.assertEqual(normalized.get("type"), "documentation")
        self.assertEqual(normalized.get("id"), "")


if __name__ == "__main__":
    unittest.main()
