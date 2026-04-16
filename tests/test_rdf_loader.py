import tempfile
import unittest
from pathlib import Path

from cidoc_rag.ingestion.loader import _extract_cidoc_id_label, _local_name, load_raw_data


class RdfLoaderHelperTests(unittest.TestCase):
    def test_extract_cidoc_id_label_class(self):
        cidoc_id, label = _extract_cidoc_id_label("E21 Person")
        self.assertEqual(cidoc_id, "E21")
        self.assertEqual(label, "Person")

    def test_extract_cidoc_id_label_property(self):
        cidoc_id, label = _extract_cidoc_id_label("P14 carried out by")
        self.assertEqual(cidoc_id, "P14")
        self.assertEqual(label, "carried out by")

    def test_local_name(self):
        self.assertEqual(_local_name("http://example.org/E21"), "E21")
        self.assertEqual(_local_name("http://example.org/ns#P14"), "P14")


class RdfLoaderIntegrationTests(unittest.TestCase):
    def test_load_ttl_entries(self):
        try:
            import rdflib  # noqa: F401
        except Exception:
            self.skipTest("rdflib is not installed in this environment")

        ttl = """
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

<http://example.org/E21> a rdfs:Class ;
    rdfs:label "E21 Person" ;
    rdfs:comment "A real person." .

<http://example.org/P14> a rdf:Property ;
    rdfs:label "P14 carried out by" ;
    rdfs:domain <http://example.org/E7> ;
    rdfs:range <http://example.org/E39> ;
    rdfs:comment "Links activity to actor." .

<http://example.org/E7> a rdfs:Class ; rdfs:label "E7 Activity" .
<http://example.org/E39> a rdfs:Class ; rdfs:label "E39 Actor" .
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "cidoc.ttl"
            file_path.write_text(ttl, encoding="utf-8")

            entries = load_raw_data(str(file_path))
            by_id = {entry.get("id"): entry for entry in entries}

            self.assertIn("E21", by_id)
            self.assertIn("P14", by_id)
            self.assertEqual(by_id["E21"].get("type"), "class")
            self.assertEqual(by_id["P14"].get("type"), "property")
            self.assertEqual(by_id["P14"].get("domain"), "E7 Activity")
            self.assertEqual(by_id["P14"].get("range"), "E39 Actor")


if __name__ == "__main__":
    unittest.main()
