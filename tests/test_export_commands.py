import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cidoc_rag.exporters.service import export_session


class ExportCommandsTests(unittest.TestCase):
    def setUp(self):
        self.turns = [
            {
                "turn": 1,
                "mode": "qa",
                "decision": "retrieve_and_answer",
                "reason": "domain-cues-detected",
                "user": "What is E21?",
                "assistant": "E21 Person.",
                "retrieved_ids": ["E21", "P14"],
                "timestamp": "2026-04-16T10:00:00+00:00",
            }
        ]
        self.session = {"k": 5, "history_turns": 4, "debug": False}

    def test_export_json_writes_turns(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "chat.json"
            export_session("json", str(out), turns=self.turns, session=self.session)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["session"]["k"], 5)
            self.assertEqual(payload["turns"][0]["retrieved_ids"], ["E21", "P14"])

    def test_export_markdown_contains_sections(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "chat.md"
            export_session("markdown", str(out), turns=self.turns, session=self.session)
            text = out.read_text(encoding="utf-8")
            self.assertIn("# CIDOC Chat Export", text)
            self.assertIn("Retrieved IDs: E21, P14", text)

    def test_export_rdf_contains_citation_triples(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "chat.ttl"
            export_session("rdf", str(out), turns=self.turns, session=self.session)
            text = out.read_text(encoding="utf-8")
            self.assertIn("chatp:usesCitation <http://www.cidoc-crm.org/cidoc-crm/E21>", text)
            self.assertIn("chatp:usesCitation <http://www.cidoc-crm.org/cidoc-crm/P14>", text)

    def test_export_ttl_alias_contains_citation_triples(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "chat.ttl"
            export_session("ttl", str(out), turns=self.turns, session=self.session)
            text = out.read_text(encoding="utf-8")
            self.assertIn("chatp:usesCitation <http://www.cidoc-crm.org/cidoc-crm/E21>", text)
            self.assertIn("chatp:usesCitation <http://www.cidoc-crm.org/cidoc-crm/P14>", text)

    def test_export_rejects_unsupported_format(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "chat.bin"
            with self.assertRaises(ValueError):
                export_session("bin", str(out), turns=self.turns, session=self.session)


if __name__ == "__main__":
    unittest.main()
