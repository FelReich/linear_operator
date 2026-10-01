#!/usr/bin/env python3

import unittest

import torch

from linear_operator.utils.cg_lanczos import recover_lanczos_cache_from_cg_directions


class TestCGLanczosRecovery(unittest.TestCase):
    def test_recovers_projected_spd_matrix(self):
        matrix = torch.diag(torch.tensor([1.0, 2.0, 4.0, 7.0], dtype=torch.float64))
        directions = torch.tensor(
            [[1.0, 1.0], [1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
        )

        q_mat, t_mat = recover_lanczos_cache_from_cg_directions(directions, matrix @ directions)

        self.assertEqual(q_mat.shape, (4, 2))
        self.assertEqual(t_mat.shape, (2, 2))
        self.assertTrue(torch.allclose(q_mat.mT @ q_mat, torch.eye(2, dtype=q_mat.dtype), atol=1e-12))
        self.assertTrue(torch.allclose(t_mat, q_mat.mT @ matrix @ q_mat, atol=1e-12))
        self.assertTrue(torch.all(torch.linalg.eigvalsh(t_mat) > 0))

    def test_rank_deficient_directions_request_fallback(self):
        directions = torch.tensor(
            [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64
        )

        self.assertEqual(
            recover_lanczos_cache_from_cg_directions(directions, directions.clone()),
            (None, None),
        )


if __name__ == "__main__":
    unittest.main()
