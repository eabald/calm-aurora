import argparse
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from cidoc_rag.cli import app_cli


class AppCliTests(unittest.TestCase):
    @patch("cidoc_rag.cli.app_cli.call_llm")
    @patch("cidoc_rag.cli.app_cli.build_prompt")
    @patch("cidoc_rag.cli.app_cli.build_context")
    @patch("cidoc_rag.cli.app_cli.retrieve")
    @patch("cidoc_rag.cli.app_cli.detect_mode")
    @patch("cidoc_rag.cli.app_cli.load_dotenv")
    @patch("cidoc_rag.cli.app_cli.parse_args")
    def test_main_qa_mode_no_context_preview(
        self,
        mock_parse_args,
        mock_load_dotenv,
        mock_detect_mode,
        mock_retrieve,
        mock_build_context,
        mock_build_prompt,
        mock_call_llm,
    ):
        mock_parse_args.return_value = argparse.Namespace(
            query="What is E21?",
            k=5,
            index_path="data/vectorstore/cidoc.index",
            metadata_path="data/vectorstore/cidoc_metadata.json",
            print_context=False,
            chat=False,
            history_turns=4,
            debug=False,
        )
        mock_detect_mode.return_value = "qa"
        mock_retrieve.return_value = [{"id": "E21"}]
        mock_build_context.return_value = "E21 Person (Class)"
        mock_build_prompt.return_value = "PROMPT"
        mock_call_llm.return_value = "E21 is Person."

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = app_cli.main()

        output = buffer.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("[info] Mode: qa", output)
        self.assertIn("[answer]", output)
        self.assertIn("E21 is Person.", output)
        mock_load_dotenv.assert_called_once()
        mock_retrieve.assert_called_once()
        mock_build_context.assert_called_once_with([{"id": "E21"}])
        mock_build_prompt.assert_called_once_with(
            query="What is E21?",
            context="E21 Person (Class)",
            mode="qa",
            history=None,
        )
        mock_call_llm.assert_called_once_with("PROMPT")

    @patch("cidoc_rag.cli.app_cli.call_llm")
    @patch("cidoc_rag.cli.app_cli.build_prompt")
    @patch("cidoc_rag.cli.app_cli.build_context")
    @patch("cidoc_rag.cli.app_cli.retrieve")
    @patch("cidoc_rag.cli.app_cli.detect_mode")
    @patch("cidoc_rag.cli.app_cli.load_dotenv")
    @patch("cidoc_rag.cli.app_cli.parse_args")
    def test_main_mapping_with_context_preview(
        self,
        mock_parse_args,
        mock_load_dotenv,
        mock_detect_mode,
        mock_retrieve,
        mock_build_context,
        mock_build_prompt,
        mock_call_llm,
    ):
        mock_parse_args.return_value = argparse.Namespace(
            query='{"table":"authors","fields":["name"]}',
            k=3,
            index_path="idx",
            metadata_path="meta",
            print_context=True,
            chat=False,
            history_turns=4,
            debug=False,
        )
        mock_detect_mode.return_value = "mapping"
        mock_retrieve.return_value = [{"id": "E21"}]
        mock_build_context.return_value = "E21 Person (Class)"
        mock_build_prompt.return_value = "PROMPT"
        mock_call_llm.return_value = '{"class":"E21"}'

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = app_cli.main()

        output = buffer.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("[info] Mode: mapping", output)
        self.assertIn("[info] Retrieved context preview:", output)
        self.assertIn("E21 Person (Class)", output)
        self.assertIn('"class": "E21"', output)
        self.assertEqual(mock_retrieve.call_count, 2)
        self.assertEqual(mock_build_context.call_count, 2)
        mock_build_prompt.assert_called_once()
        mock_call_llm.assert_called_once_with("PROMPT")

    @patch("cidoc_rag.cli.app_cli.chat_loop")
    @patch("cidoc_rag.cli.app_cli.load_dotenv")
    @patch("cidoc_rag.cli.app_cli.parse_args")
    def test_main_chat_mode_delegates_to_chat_loop(self, mock_parse_args, mock_load_dotenv, mock_chat_loop):
        mock_parse_args.return_value = argparse.Namespace(
            query=None,
            k=6,
            index_path="idx",
            metadata_path="meta",
            print_context=False,
            chat=True,
            history_turns=5,
            debug=True,
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = app_cli.main()

        self.assertEqual(rc, 0)
        mock_chat_loop.assert_called_once_with(
            k=6,
            index_path="idx",
            metadata_path="meta",
            history_turns=5,
            debug=True,
        )

    @patch("cidoc_rag.cli.app_cli.chat_loop")
    @patch("cidoc_rag.cli.app_cli.load_dotenv")
    @patch("cidoc_rag.cli.app_cli.parse_args")
    def test_main_chat_mode_handles_missing_faiss(self, mock_parse_args, mock_load_dotenv, mock_chat_loop):
        mock_parse_args.return_value = argparse.Namespace(
            query=None,
            k=5,
            index_path="idx",
            metadata_path="meta",
            print_context=False,
            chat=True,
            history_turns=4,
            debug=False,
        )
        mock_chat_loop.side_effect = ModuleNotFoundError(
            "faiss is not installed. Install faiss-cpu to use vector store operations."
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = app_cli.main()

        output = buffer.getvalue()
        self.assertEqual(rc, 1)
        self.assertIn("faiss is not installed", output.lower())
        self.assertIn("pip install -r requirements.txt", output)


if __name__ == "__main__":
    unittest.main()
