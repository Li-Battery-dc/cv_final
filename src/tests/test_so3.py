"""Tests for SO(3) rotation utilities."""

import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ba.utils import rodrigues_rotate, matrix_to_rotvec


def test_matrix_to_rotvec_roundtrip():
    """Rotation matrix -> rotvec -> matrix should be identity."""
    rng = np.random.RandomState(42)
    for _ in range(100):
        # Generate random rotation via random rotation vector
        rv = rng.randn(3)
        R = rodrigues_rotate(rv)
        # Round-trip
        rv2 = matrix_to_rotvec(R)
        R2 = rodrigues_rotate(rv2)
        np.testing.assert_allclose(R, R2, atol=1e-10,
                                   err_msg="Matrix->rotvec->matrix round-trip failed")


def test_rotvec_to_matrix_roundtrip():
    """Rotation vector -> matrix -> rotvec should produce equivalent rotation."""
    rng = np.random.RandomState(42)
    for _ in range(100):
        rv = rng.randn(3)
        R = rodrigues_rotate(rv)
        rv2 = matrix_to_rotvec(R)
        # Reconstruct matrix from recovered rotvec to verify equivalence
        R2 = rodrigues_rotate(rv2)
        np.testing.assert_allclose(R, R2, atol=1e-10,
                                   err_msg="Rotvec->matrix->rotvec equivalence failed")


def test_identity_rotation():
    """Zero rotation vector should give identity matrix."""
    rv = np.zeros(3)
    R = rodrigues_rotate(rv)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    # Identity matrix round-trip
    rv2 = matrix_to_rotvec(np.eye(3))
    np.testing.assert_allclose(rv2, np.zeros(3), atol=1e-10)


def test_rotation_degree():
    """Check that rotation angle is preserved through round-trip."""
    rng = np.random.RandomState(42)
    for _ in range(20):
        angle = rng.uniform(0.1, np.pi - 0.1)
        axis = rng.randn(3)
        axis = axis / np.linalg.norm(axis)
        rv = axis * angle
        R = rodrigues_rotate(rv)
        rv2 = matrix_to_rotvec(R)
        # Angle should be preserved
        np.testing.assert_allclose(np.linalg.norm(rv2), angle, rtol=1e-5)

    print("All SO(3) tests passed!")


if __name__ == "__main__":
    test_matrix_to_rotvec_roundtrip()
    test_rotvec_to_matrix_roundtrip()
    test_identity_rotation()
    test_rotation_degree()
