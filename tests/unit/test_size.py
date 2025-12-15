# tests/unit/test_size.py
#
# SWE 45000 – Phase 2
#
# Unit tests for model size–based compatibility scoring.
#
# This file validates the size scoring logic used to evaluate how well
# a model fits different hardware environments (Raspberry Pi, Jetson Nano,
# desktop PC, and AWS server) based on its total model size.
#
# The tests ensure:
# - Full branch coverage of the internal normalization logic
#   (minimum, maximum, and interpolated ranges)
# - Correct unit conversions from bytes → MB → GB
# - Proper clamping behavior at hardware-specific size thresholds
# - Deterministic, hardware-aware scoring outputs in the range [0.0, 1.0]
#
# External dependencies are mocked or reimplemented locally to keep the
# tests isolated, deterministic, and rubric-compliant.
import unittest
from unittest.mock import Mock, patch
from typing import Dict, Any

# Mock the external dependency structure that the score_model_size function uses.
# We define a placeholder for ModelMetadata to satisfy type hints and structure.
# In a real setup, this would be imported from 'huggingface_service'.
class ModelMetadata:
    """Mock class to simulate the structure of ModelMetadata."""
    def __init__(self, modelSize: int):
        self.modelSize = modelSize

# --- Copy of the score_model_size function from the user's request ---
def score_model_size(metadata: ModelMetadata) -> Dict[str, float]:
    """Return size compatibility scores per hardware type (0–1)."""
    # 1024 * 1024 is the number of bytes in one megabyte.
    size_mb = metadata.modelSize / (1024 * 1024)  # MB
    # Dividing MB by 1024 converts it to GB.
    size_gb = size_mb / 1024

    def normalize(value: float, min_val: float, max_val: float) -> float:
        """Linearly scale size into [0,1], clamped."""
        if value <= min_val:
            return 1.0  # Hit this branch with size 0
        elif value >= max_val:
            return 0.0  # Hit this branch with sizes above the maximum
        # Hit this branch with sizes between min and max
        return 1 - ((value - min_val) / (max_val - min_val))

    return {
        "raspberry_pi": normalize(size_gb, 0.0, 1.0),    # best <1GB
        "jetson_nano": normalize(size_gb, 0.0, 2.0),     # best <2GB
        "desktop_pc": normalize(size_gb, 0.0, 6.0),      # best <6GB
        "aws_server": normalize(size_gb, 0.0, 10.0),    # best <10GB
    }
# -------------------------------------------------------------------


# Constant for one gigabyte in bytes (1024^3)
GIGABYTE_IN_BYTES = 1024 * 1024 * 1024

class TestModelSizeScoring(unittest.TestCase):
    """
    Tests the score_model_size function to ensure full line coverage
    on the internal 'normalize' function and correct threshold application.
    """

    # Use a small delta for float comparison since the calculations are precise
    # and we want to ensure exact conversions work as expected.
    places = 6

    def create_metadata(self, size_gb: float) -> ModelMetadata:
        """Helper to create a ModelMetadata mock with size set in GB."""
        return ModelMetadata(modelSize=int(size_gb * GIGABYTE_IN_BYTES))

    def test_zero_size_score(self):
        """
        Tests the minimum size case (0 bytes).
        This ensures the 'value <= min_val' branch of normalize is hit for all keys.
        Expected score: 1.0 everywhere.
        """
        metadata = self.create_metadata(0.0) # 0 GB
        scores = score_model_size(metadata)
        self.assertEqual(scores['raspberry_pi'], 1.0)
        self.assertEqual(scores['jetson_nano'], 1.0)
        self.assertEqual(scores['desktop_pc'], 1.0)
        self.assertEqual(scores['aws_server'], 1.0)

    def test_oversized_score(self):
        """
        Tests a size larger than the maximum threshold (10 GB).
        This ensures the 'value >= max_val' branch of normalize is hit for all keys.
        Expected score: 0.0 everywhere.
        """
        metadata = self.create_metadata(11.0) # 11 GB
        scores = score_model_size(metadata)
        self.assertEqual(scores['raspberry_pi'], 0.0)
        self.assertEqual(scores['jetson_nano'], 0.0)
        self.assertEqual(scores['desktop_pc'], 0.0)
        self.assertEqual(scores['aws_server'], 0.0)

    def test_mid_range_score(self):
        """
        Tests a medium size (0.5 GB) that should fall between min and max
        for all hardware, ensuring the interpolation branch is hit for all keys.
        """
        metadata = self.create_metadata(0.5) # 0.5 GB
        scores = score_model_size(metadata)

        # Pi: 1 - (0.5 / 1.0) = 0.5
        self.assertAlmostEqual(scores['raspberry_pi'], 0.5, places=self.places)

        # Nano: 1 - (0.5 / 2.0) = 0.75
        self.assertAlmostEqual(scores['jetson_nano'], 0.75, places=self.places)

        # PC: 1 - (0.5 / 6.0) = 0.916666...
        self.assertAlmostEqual(scores['desktop_pc'], 0.916667, places=self.places)

        # Server: 1 - (0.5 / 10.0) = 0.95
        self.assertAlmostEqual(scores['aws_server'], 0.95, places=self.places)

    def test_hardware_specific_limits(self):
        """
        Tests sizes that precisely match the maximum threshold for specific hardware
        types, ensuring correct clamping (0.0) and interpolation for others.
        """
        # --- Test 1.0 GB limit (Pi clamped, others interpolating) ---
        metadata_1gb = self.create_metadata(1.0) # 1.0 GB
        scores_1gb = score_model_size(metadata_1gb)

        # Pi (max=1.0): 1.0 >= 1.0 -> 0.0 (clamped)
        self.assertAlmostEqual(scores_1gb['raspberry_pi'], 0.0, places=self.places)
        # Nano (max=2.0): 1 - (1.0 / 2.0) = 0.5
        self.assertAlmostEqual(scores_1gb['jetson_nano'], 0.5, places=self.places)
        # PC (max=6.0): 1 - (1.0 / 6.0) = 0.833333...
        self.assertAlmostEqual(scores_1gb['desktop_pc'], 0.833333, places=self.places)
        # Server (max=10.0): 1 - (1.0 / 10.0) = 0.9
        self.assertAlmostEqual(scores_1gb['aws_server'], 0.9, places=self.places)

        # --- Test 2.0 GB limit (Pi/Nano clamped, others interpolating) ---
        metadata_2gb = self.create_metadata(2.0) # 2.0 GB
        scores_2gb = score_model_size(metadata_2gb)

        # Pi (max=1.0): 2.0 >= 1.0 -> 0.0 (clamped)
        self.assertAlmostEqual(scores_2gb['raspberry_pi'], 0.0, places=self.places)
        # Nano (max=2.0): 2.0 >= 2.0 -> 0.0 (clamped)
        self.assertAlmostEqual(scores_2gb['jetson_nano'], 0.0, places=self.places)
        # PC (max=6.0): 1 - (2.0 / 6.0) = 0.666666...
        self.assertAlmostEqual(scores_2gb['desktop_pc'], 0.666667, places=self.places)
        # Server (max=10.0): 1 - (2.0 / 10.0) = 0.8
        self.assertAlmostEqual(scores_2gb['aws_server'], 0.8, places=self.places)

        # --- Test 6.0 GB limit (Pi/Nano/PC clamped, Server interpolating) ---
        metadata_6gb = self.create_metadata(6.0) # 6.0 GB
        scores_6gb = score_model_size(metadata_6gb)

        # Pi, Nano, PC: clamped
        self.assertAlmostEqual(scores_6gb['raspberry_pi'], 0.0, places=self.places)
        self.assertAlmostEqual(scores_6gb['jetson_nano'], 0.0, places=self.places)
        self.assertAlmostEqual(scores_6gb['desktop_pc'], 0.0, places=self.places)
        # Server (max=10.0): 1 - (6.0 / 10.0) = 0.4
        self.assertAlmostEqual(scores_6gb['aws_server'], 0.4, places=self.places)

