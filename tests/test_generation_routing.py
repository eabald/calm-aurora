import unittest
from unittest.mock import Mock, patch

from cidoc_rag.generation.service import generate_answer


class GenerationRoutingTests(unittest.TestCase):
    @patch("cidoc_rag.generation.service.get_ollama_client")
    @patch("cidoc_rag.generation.service.get_llm_model")
    @patch("cidoc_rag.generation.service.build_prompt")
    @patch("cidoc_rag.generation.service.build_context")
    @patch("cidoc_rag.generation.service.retrieve")
    @patch("cidoc_rag.generation.service.detect_mode")
    def test_generate_answer_uses_mapping_mode_in_prompt(
        self,
        mock_detect_mode,
        mock_retrieve,
        mock_build_context,
        mock_build_prompt,
        mock_get_llm_model,
        mock_get_ollama_client,
    ):
        mock_detect_mode.return_value = "mapping"
        mock_retrieve.return_value = [{"id": "P14"}]
        mock_build_context.return_value = "P14 carried out by (Property)"
        mock_build_prompt.return_value = "PROMPT"
        mock_get_llm_model.return_value = "llama3.1:8b"

        fake_client = Mock()
        fake_client.chat_completion.return_value = '{"class":"E7"}'
        mock_get_ollama_client.return_value = fake_client

        answer = generate_answer("{\"table\":\"authors\"}", k=4, index_path="idx", metadata_path="meta")

        self.assertEqual(answer, '{"class":"E7"}')
        mock_detect_mode.assert_called_once()
        mock_retrieve.assert_called_once_with(query='{"table":"authors"}', k=4, index_path="idx", metadata_path="meta")
        mock_build_context.assert_called_once_with([{"id": "P14"}])
        mock_build_prompt.assert_called_once_with(
            query='{"table":"authors"}',
            context="P14 carried out by (Property)",
            mode="mapping",
            history=None,
        )
        fake_client.chat_completion.assert_called_once_with(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "PROMPT"}],
            temperature=0.2,
        )


if __name__ == "__main__":
    unittest.main()
