import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import app


class AppChatLoopTests(unittest.TestCase):
    def test_apply_runtime_command_debug_toggle(self):
        handled, debug, k, message = app.apply_runtime_command("debug on", debug=False, k=5)
        self.assertTrue(handled)
        self.assertTrue(debug)
        self.assertEqual(k, 5)
        self.assertIn("enabled", message)

        handled, debug, k, message = app.apply_runtime_command("debug off", debug=True, k=5)
        self.assertTrue(handled)
        self.assertFalse(debug)
        self.assertEqual(k, 5)
        self.assertIn("disabled", message)

    def test_apply_runtime_command_k_validation(self):
        handled, debug, k, message = app.apply_runtime_command("k=10", debug=False, k=5)
        self.assertTrue(handled)
        self.assertEqual(k, 10)
        self.assertIn("k=10", message)

        handled, debug, k, message = app.apply_runtime_command("k=0", debug=False, k=5)
        self.assertTrue(handled)
        self.assertEqual(k, 5)
        self.assertIn("greater than 0", message)

        handled, debug, k, message = app.apply_runtime_command("k=abc", debug=False, k=5)
        self.assertTrue(handled)
        self.assertEqual(k, 5)
        self.assertIn("Invalid k", message)

    def test_mapping_answer_pretty_print_fallback(self):
        formatted = app.format_answer_for_mode("mapping", '{"class":"E21"}')
        self.assertIn("\n", formatted)
        self.assertIn('"class": "E21"', formatted)

        raw = "not-json"
        self.assertEqual(app.format_answer_for_mode("mapping", raw), raw)

    @patch("cidoc_rag.cli.chat_runtime.call_llm")
    @patch("cidoc_rag.cli.chat_runtime.build_prompt")
    @patch("cidoc_rag.cli.chat_runtime.build_context")
    @patch("cidoc_rag.cli.chat_runtime.retrieve")
    @patch("cidoc_rag.cli.chat_runtime.detect_mode")
    def test_chat_loop_truncates_history_and_retrieves_current_query_only(
        self,
        mock_detect_mode,
        mock_retrieve,
        mock_build_context,
        mock_build_prompt,
        mock_call_llm,
    ):
        mock_detect_mode.return_value = "qa"
        mock_retrieve.return_value = [{"id": "E21"}]
        mock_build_context.return_value = "E21 Person (Class)"
        mock_build_prompt.return_value = "PROMPT"
        mock_call_llm.side_effect = ["a1", "a2", "a3"]

        inputs = ["q1", "q2", "q3", "quit"]

        with patch("builtins.input", side_effect=inputs):
            with redirect_stdout(io.StringIO()):
                history = app.chat_loop(
                    k=5,
                    index_path="idx",
                    metadata_path="meta",
                    history_turns=2,
                    debug=False,
                )

        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "q2")
        self.assertEqual(history[2]["content"], "q3")

        retrieval_queries = [call.kwargs["query"] for call in mock_retrieve.call_args_list]
        self.assertEqual(retrieval_queries, ["q1", "q2", "q3"])


if __name__ == "__main__":
    unittest.main()
