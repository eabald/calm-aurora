import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import app
from cidoc_rag.agent.policy import Decision
from cidoc_rag.cli import chat_runtime


class AppChatLoopTests(unittest.TestCase):
    def test_extract_rdf_payload_from_ttl_code_block(self):
        response = (
            "Here is the updated ontology.\n\n"
            "```ttl\n"
            "@prefix ex: <http://example.org/> .\n"
            "ex:E21 ex:label \"Person\" .\n"
            "```\n"
        )

        payload = chat_runtime._extract_rdf_payload(response)
        self.assertIn("@prefix ex:", payload)
        self.assertIn("ex:E21 ex:label", payload)

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

    @patch("cidoc_rag.cli.chat_runtime.call_llm")
    @patch("cidoc_rag.cli.chat_runtime.build_prompt")
    @patch("cidoc_rag.cli.chat_runtime.build_context")
    @patch("cidoc_rag.cli.chat_runtime.retrieve")
    @patch("cidoc_rag.cli.chat_runtime.decide_next_action")
    @patch("cidoc_rag.cli.chat_runtime.detect_mode")
    def test_chat_loop_skips_retrieval_when_decision_says_so(
        self,
        mock_detect_mode,
        mock_decide_next_action,
        mock_retrieve,
        mock_build_context,
        mock_build_prompt,
        mock_call_llm,
    ):
        mock_detect_mode.return_value = "qa"
        mock_decide_next_action.return_value = Decision(
            action="answer_without_retrieval",
            reason="smalltalk-or-meta",
        )
        mock_build_context.return_value = "No CIDOC context retrieved."
        mock_build_prompt.return_value = "PROMPT"
        mock_call_llm.return_value = "Hi there."

        inputs = ["hello", "quit"]
        with patch("builtins.input", side_effect=inputs):
            with redirect_stdout(io.StringIO()) as output:
                history = app.chat_loop(
                    k=5,
                    index_path="idx",
                    metadata_path="meta",
                    history_turns=2,
                    debug=False,
                )

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["content"], "hello")
        self.assertEqual(history[1]["content"], "Hi there.")
        mock_retrieve.assert_not_called()
        mock_build_prompt.assert_called_once()
        self.assertIn("Retrieval skipped", output.getvalue())

    @patch("cidoc_rag.cli.chat_runtime.call_llm")
    @patch("cidoc_rag.cli.chat_runtime.build_prompt")
    @patch("cidoc_rag.cli.chat_runtime.build_context")
    @patch("cidoc_rag.cli.chat_runtime.retrieve")
    @patch("cidoc_rag.cli.chat_runtime.decide_next_action")
    @patch("cidoc_rag.cli.chat_runtime.detect_mode")
    def test_chat_loop_asks_clarification_then_retrieves_after_reply(
        self,
        mock_detect_mode,
        mock_decide_next_action,
        mock_retrieve,
        mock_build_context,
        mock_build_prompt,
        mock_call_llm,
    ):
        mock_detect_mode.return_value = "qa"
        mock_decide_next_action.side_effect = [
            Decision(
                action="ask_clarifying",
                reason="context-dependent-query-without-grounding",
                clarification_question="Could you clarify which CIDOC class or property you mean?",
            ),
            Decision(
                action="retrieve_and_answer",
                reason="domain-cues-detected",
            ),
        ]
        mock_retrieve.return_value = [{"id": "E21"}]
        mock_build_context.return_value = "E21 Person (Class)"
        mock_build_prompt.return_value = "PROMPT"
        mock_call_llm.return_value = "You likely mean E21 Person."

        inputs = ["which one?", "E21", "quit"]
        with patch("builtins.input", side_effect=inputs):
            with redirect_stdout(io.StringIO()) as output:
                history = app.chat_loop(
                    k=5,
                    index_path="idx",
                    metadata_path="meta",
                    history_turns=3,
                    debug=False,
                )

        self.assertEqual(len(history), 4)
        self.assertIn("Could you clarify", history[1]["content"])
        self.assertIn("Clarification: E21", mock_retrieve.call_args.kwargs["query"])
        self.assertIn("I need one clarification", output.getvalue())

    @patch("cidoc_rag.cli.chat_runtime.call_llm")
    @patch("cidoc_rag.cli.chat_runtime.build_prompt")
    @patch("cidoc_rag.cli.chat_runtime.build_context")
    @patch("cidoc_rag.cli.chat_runtime.retrieve")
    @patch("cidoc_rag.cli.chat_runtime.decide_next_action")
    @patch("cidoc_rag.cli.chat_runtime.detect_mode")
    def test_chat_loop_import_apply_and_save_rdf(
        self,
        mock_detect_mode,
        mock_decide_next_action,
        mock_retrieve,
        mock_build_context,
        mock_build_prompt,
        mock_call_llm,
    ):
        mock_detect_mode.return_value = "qa"
        mock_decide_next_action.return_value = Decision(
            action="answer_without_retrieval",
            reason="user-editing-imported-rdf",
        )
        mock_build_context.return_value = "No CIDOC context retrieved."
        mock_build_prompt.return_value = "PROMPT"
        mock_call_llm.return_value = (
            "```ttl\n"
            "@prefix ex: <http://example.org/> .\n"
            "ex:E21 ex:label \"Person\" .\n"
            "```"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "source.ttl"
            source.write_text(
                "@prefix ex: <http://example.org/> .\n"
                "ex:E21 ex:label \"Old\" .\n",
                encoding="utf-8",
            )
            target = Path(tmp_dir) / "edited.ttl"

            inputs = [
                f"import-rdf {source}",
                "Please update imported RDF and return turtle.",
                "apply-rdf",
                f"save-rdf {target}",
                "quit",
            ]

            with patch("builtins.input", side_effect=inputs):
                with redirect_stdout(io.StringIO()) as output:
                    history = app.chat_loop(
                        k=5,
                        index_path="idx",
                        metadata_path="meta",
                        history_turns=2,
                        debug=False,
                    )

            saved = target.read_text(encoding="utf-8")
            self.assertIn('ex:E21 ex:label "Person" .', saved)
            self.assertEqual(len(history), 2)
            self.assertIn("Imported RDF file", output.getvalue())
            self.assertIn("Applied RDF update", output.getvalue())
            self.assertIn("Saved RDF file", output.getvalue())
            mock_retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
