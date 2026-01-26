
import unittest
import numpy as np
from typing import Dict, List, Tuple
from flops_infra_drift.diws import DIWS
from flops_infra_drift import consts
from unittest.mock import MagicMock

class TestCosineSimilaritySelection(unittest.TestCase):
    def setUp(self):
        self.strategy = DIWS()
        self.strategy.label_distribution = {}
        self.strategy.previous_round_updates = {}
        # Mock consts
        self.original_top_k = consts.TOP_K_CLIENTS
        consts.TOP_K_CLIENTS = 2

    def tearDown(self):
        consts.TOP_K_CLIENTS = self.original_top_k

    def test_fallback_when_no_history(self):
        # No history for "dropped"
        # Falls back to label relevance
        self.strategy.label_distribution = {
            "dropped": {"A": 100},
            "c1": {"A": 100},
            "c2": {"B": 100}
        }
        
        selected = self.strategy.get_top_k_similarity_clients("dropped", ["c1", "c2"], k=1)
        self.assertEqual(selected, ["c1"])

    def test_cosine_logic(self):
        # History exists
        # Update vectors (mocked as simple lists/arrays)
        # Dropped: [1, 0]
        # c1: [1, 0] (Sim = 1.0)
        # c2: [0, 1] (Sim = 0.0)
        # c3: [-1, 0] (Sim = -1.0)
        
        self.strategy.previous_round_updates = {
            "dropped": np.array([1, 0]),
            "c1": np.array([1, 0]),
            "c2": np.array([0, 1]),
            "c3": np.array([-1, 0])
        }
        # Fake label dist just to bypass empty check if needed
        self.strategy.label_distribution = {"dropped": {"A": 1}}
        
        selected = self.strategy.get_top_k_similarity_clients("dropped", ["c1", "c2", "c3"], k=3)
        
        self.assertEqual(selected[0], "c1") # Best
        self.assertEqual(selected[1], "c2") # Middle
        self.assertEqual(selected[2], "c3") # Worst

if __name__ == '__main__':
    unittest.main()
