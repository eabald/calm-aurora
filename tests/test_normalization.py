import unittest

from cidoc_rag.ingestion.normalize import normalize_entry


class NormalizationTests(unittest.TestCase):
    def test_class_defaults(self):
        entry = normalize_entry({"id": "E21", "type": "class", "label": "Person"})
        self.assertEqual(entry["id"], "E21")
        self.assertEqual(entry["type"], "class")
        self.assertEqual(entry["definition"], "")
        self.assertEqual(entry["examples"], "")
        self.assertEqual(entry["related_properties"], [])

    def test_property_defaults(self):
        entry = normalize_entry({"id": "P14", "type": "property", "label": "carried out by"})
        self.assertEqual(entry["id"], "P14")
        self.assertEqual(entry["type"], "property")
        self.assertEqual(entry["domain"], "")
        self.assertEqual(entry["range"], "")
        self.assertEqual(entry["definition"], "")


if __name__ == "__main__":
    unittest.main()
