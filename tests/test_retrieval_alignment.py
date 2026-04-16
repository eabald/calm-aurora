import unittest
from unittest.mock import patch

import numpy as np

from cidoc_rag.retrieval.service import retrieve


class FakeIndex:
    def __init__(self, ntotal):
        self.ntotal = ntotal


class RetrievalAlignmentTests(unittest.TestCase):
    @patch("cidoc_rag.retrieval.service.search_index")
    @patch("cidoc_rag.retrieval.service.embed_text")
    @patch("cidoc_rag.retrieval.service.load_metadata")
    @patch("cidoc_rag.retrieval.service.load_index")
    def test_skips_out_of_bounds_indices(self, mock_load_index, mock_load_metadata, mock_embed_text, mock_search_index):
        mock_load_index.return_value = FakeIndex(ntotal=3)
        mock_load_metadata.return_value = [{"id": "E21"}, {"id": "P14"}]
        mock_embed_text.return_value = [0.1, 0.2, 0.3]
        mock_search_index.return_value = (
            np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )

        results = retrieve("test query", k=3)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "E21")
        self.assertEqual(results[1]["id"], "P14")


if __name__ == "__main__":
    unittest.main()
