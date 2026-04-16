import unittest

from cidoc_rag.agent.policy import decide_next_action


class AgentPolicyTests(unittest.TestCase):
    def test_smalltalk_skips_retrieval(self):
        decision = decide_next_action(
            query="hello",
            history=[],
            last_retrieval=[],
            mode="qa",
        )
        self.assertEqual(decision.action, "answer_without_retrieval")

    def test_ambiguous_followup_requests_clarification(self):
        decision = decide_next_action(
            query="which one?",
            history=[],
            last_retrieval=[],
            mode="qa",
        )
        self.assertEqual(decision.action, "ask_clarifying")
        self.assertTrue(decision.clarification_question)

    def test_domain_question_runs_retrieval(self):
        decision = decide_next_action(
            query="What is E21 in CIDOC CRM?",
            history=[],
            last_retrieval=[],
            mode="qa",
        )
        self.assertEqual(decision.action, "retrieve_and_answer")


if __name__ == "__main__":
    unittest.main()
