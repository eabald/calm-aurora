import unittest

from cidoc_rag.prompting.builders import build_context


class ContextBuilderTests(unittest.TestCase):
    def test_builds_class_and_property_context_with_ids(self):
        results = [
            {
                "id": "E21",
                "type": "class",
                "label": "Person",
                "definition": "A person.",
                "examples": "Leonardo da Vinci",
                "related_properties": ["P14 carried out by"],
            },
            {
                "id": "P14",
                "type": "property",
                "label": "carried out by",
                "domain": "E7 Activity",
                "range": "E39 Actor",
                "definition": "Links activities to actors.",
            },
        ]

        context = build_context(results)
        self.assertIn("E21 Person (Class)", context)
        self.assertIn("P14 carried out by (Property)", context)
        self.assertIn("Domain: E7 Activity", context)
        self.assertIn("Range: E39 Actor", context)

    def test_builds_documentation_context(self):
        results = [
            {
                "id": "",
                "type": "documentation",
                "label": "CRM Primer",
                "definition": "Narrative guidance for applying CIDOC CRM in museum workflows.",
                "source_file": "data/docs/primer.md",
            }
        ]

        context = build_context(results)
        self.assertIn("CRM Primer (Documentation)", context)
        self.assertIn("Content: Narrative guidance", context)
        self.assertIn("Source: data/docs/primer.md", context)


if __name__ == "__main__":
    unittest.main()
