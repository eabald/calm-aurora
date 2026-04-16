import unittest

from cidoc_rag.prompting.builders import build_prompt


class PromptTemplateTests(unittest.TestCase):
    def test_qa_prompt_contains_grounding_rule(self):
        prompt = build_prompt(query="What is E21?", context="E21 Person (Class)", mode="qa")
        self.assertIn("Use ONLY the provided context", prompt)
        self.assertIn("I don't know", prompt)
        self.assertIn("Current question", prompt)

    def test_mapping_prompt_contains_json_schema_and_rules(self):
        prompt = build_prompt(query='{"table": "authors"}', context="P14 carried out by", mode="mapping")
        self.assertIn('"class": "..."', prompt)
        self.assertIn("Only use valid CIDOC CRM IDs (E*, P*)", prompt)

    def test_prompt_includes_history_when_provided(self):
        history = [
            {"role": "user", "content": "What is E21?"},
            {"role": "assistant", "content": "E21 is Person."},
        ]
        prompt = build_prompt(
            query="Map author name",
            context="E21 Person (Class)",
            mode="mapping",
            history=history,
        )
        self.assertIn("Conversation history:", prompt)
        self.assertIn("User: What is E21?", prompt)
        self.assertIn("Assistant: E21 is Person.", prompt)
        self.assertIn("Current question:", prompt)


if __name__ == "__main__":
    unittest.main()
