# tests/test_discrete_mi.py

import unittest
import numpy as np
from src.mi.discrete_mi import compute_mi

class TestDiscreteMI(unittest.TestCase):
    def test_compute_mutual_information(self):
        # Define a simple joint PMF
        p_xy = np.array([[0.1, 0.2], [0.3, 0.4]])

        # Compute the expected mutual information manually
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)
        p_xy_normalized = p_xy / np.sum(p_xy)
        expected_mi = np.sum(p_xy_normalized * np.log(p_xy_normalized / (p_x[:, None] * p_y[None, :])))

        # Compute MI using the function
        computed_mi = compute_mi(p_xy)

        # Assert that the computed MI is close to the expected MI
        self.assertAlmostEqual(computed_mi, expected_mi, places=5)

if __name__ == '__main__':
    unittest.main()