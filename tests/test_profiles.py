import os
import unittest
from unittest.mock import patch

from spark import use_spark, use_resident_fp8


class ProfileTest(unittest.TestCase):
    def test_discrete_resident_does_not_use_shared_memory_preflight(self):
        with patch.dict(os.environ, {"MINIMAX_H3_PROFILE": "resident-fp8"}):
            self.assertTrue(use_resident_fp8())
            self.assertFalse(use_spark())

    def test_automatic_selection_preserves_discrete_default(self):
        with patch.dict(os.environ, {"MINIMAX_H3_PROFILE": "auto"}), patch('spark.is_spark', return_value=False):
            self.assertFalse(use_resident_fp8())
        with patch.dict(os.environ, {"MINIMAX_H3_PROFILE": "auto"}), patch('spark.is_spark', return_value=True):
            self.assertTrue(use_resident_fp8())
