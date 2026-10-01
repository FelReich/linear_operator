#!/usr/bin/env python3

import unittest

import torch
from scipy.stats import ortho_group

from linear_operator.test.utils import approx_equal
from linear_operator.utils.lanczos import (
    extend_lanczos_basis,
    extend_lanczos_basis_to_root_inv_decomposition,
    lanczos_tridiag,
)


class TestLanczos(unittest.TestCase):
    def assert_valid_sizes(self, size, t_mat, q_mat):
        rank = t_mat.shape[0]
        self.assertTrue(0 < rank <= size)
        self.assertEqual(rank, t_mat.shape[1])
        self.assertEqual(rank, q_mat.shape[1])
        self.assertEqual(size, q_mat.shape[0])

    def assert_tridiagonally_positive(self, t_mat):
        for i in range(t_mat.shape[0]):
            for elem in t_mat.data[i, i - 1 : i + 2]:
                self.assertGreater(elem, 0)

    def lanczos_tridiag_test(self, matrix):
        size = matrix.shape[0]
        q_mat, t_mat = lanczos_tridiag(
            matrix.matmul,
            max_iter=size,
            dtype=matrix.dtype,
            device=matrix.device,
            matrix_shape=matrix.shape,
        )

        self.assert_valid_sizes(size, t_mat, q_mat)
        self.assert_tridiagonally_positive(t_mat)
        approx = q_mat.matmul(t_mat).matmul(q_mat.mT)
        self.assertTrue(approx_equal(approx, matrix))

    # this type of matrix has eigenvalues of similar scale, so our approximation will likely create a tridiaganal
    # matrix of the same size
    def test_lanczos_tridiag_near_exact(self):
        size = 100
        matrix = torch.randn(size, size)
        matrix = matrix.matmul(matrix.mT)
        matrix.div_(matrix.norm())
        matrix.add_(torch.diag_embed(torch.ones(matrix.size(-1)).mul(1e-6)))
        self.lanczos_tridiag_test(matrix)

    # this kind of matrix has eigenvalues on very different scales, so our approximation will likely create a
    # tridiagonal matrix of smaller size
    def test_lanczos_tridiag_approx(self):
        size = 30
        orthogonal = torch.from_numpy(ortho_group.rvs(size)).float()
        diag = torch.diag_embed(torch.FloatTensor([10**-i for i in range(size)]))
        matrix = torch.matmul(orthogonal, torch.matmul(diag, orthogonal.transpose(0, 1)))
        self.lanczos_tridiag_test(matrix)

    def test_extend_batched_basis(self):
        matrices = torch.tensor(
            [
                [[2.0, 0.25, 0.0], [0.25, 3.0, 0.3], [0.0, 0.3, 4.0]],
                [[1.5, 0.4, 0.0], [0.4, 2.5, 0.2], [0.0, 0.2, 3.5]],
            ],
            dtype=torch.float64,
        )
        q_initial = torch.zeros(2, 3, 1, dtype=torch.float64)
        q_initial[:, 0, 0] = 1
        t_initial = matrices[:, :1, :1]

        q_mat, t_mat = extend_lanczos_basis(
            matrices.matmul, 3, matrices.dtype, matrices.device, matrices.shape[-2:], q_initial, t_initial
        )

        self.assertEqual(q_mat.shape, (2, 3, 3))
        self.assertEqual(t_mat.shape, (2, 3, 3))
        torch.testing.assert_close(q_mat.transpose(-1, -2) @ q_mat, torch.eye(3, dtype=matrices.dtype).expand(2, 3, 3))
        torch.testing.assert_close(t_mat, q_mat.transpose(-1, -2) @ matrices @ q_mat)

        inv_root = extend_lanczos_basis_to_root_inv_decomposition(
            matrices.matmul, 3, matrices.dtype, matrices.device, matrices.shape[-2:], q_initial, t_initial
        )
        self.assertEqual(inv_root.shape, (2, 3, 3))
        torch.testing.assert_close(
            inv_root @ inv_root.transpose(-1, -2), torch.linalg.inv(matrices), atol=1e-5, rtol=1e-5
        )

        q_two = torch.eye(3, dtype=matrices.dtype)[:, :2].expand(2, 3, 2)
        t_two = matrices[:, :2, :2]
        q_mat, t_mat = extend_lanczos_basis(
            matrices.matmul, 3, matrices.dtype, matrices.device, matrices.shape[-2:], q_two, t_two
        )
        self.assertEqual(q_mat.shape, (2, 3, 3))
        torch.testing.assert_close(t_mat, q_mat.transpose(-1, -2) @ matrices @ q_mat)

        multi_matrices = matrices.unsqueeze(0).expand(2, 2, 3, 3)
        multi_q = q_initial.unsqueeze(0).expand(2, 2, 3, 1)
        multi_t = t_initial.unsqueeze(0).expand(2, 2, 1, 1)
        q_mat, t_mat = extend_lanczos_basis(
            multi_matrices.matmul,
            3,
            matrices.dtype,
            matrices.device,
            matrices.shape[-2:],
            multi_q,
            multi_t,
        )
        self.assertEqual(q_mat.shape, (2, 2, 3, 3))
        torch.testing.assert_close(t_mat, q_mat.transpose(-1, -2) @ multi_matrices @ q_mat)
        multi_inv_root = extend_lanczos_basis_to_root_inv_decomposition(
            multi_matrices.matmul,
            3,
            matrices.dtype,
            matrices.device,
            matrices.shape[-2:],
            multi_q,
            multi_t,
        )
        self.assertEqual(multi_inv_root.shape, (2, 2, 3, 3))
        torch.testing.assert_close(
            multi_inv_root @ multi_inv_root.transpose(-1, -2),
            torch.linalg.inv(multi_matrices),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_extend_batched_basis_with_mixed_breakdown(self):
        matrices = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
                [[1.0, 0.2, 0.0], [0.2, 2.0, 0.0], [0.0, 0.0, 3.0]],
            ],
            dtype=torch.float64,
        )
        q_initial = torch.zeros(2, 3, 1, dtype=torch.float64)
        q_initial[:, 0, 0] = 1
        t_initial = matrices[:, :1, :1]

        q_mat, t_mat = extend_lanczos_basis(
            matrices.matmul, 2, matrices.dtype, matrices.device, matrices.shape[-2:], q_initial, t_initial
        )

        self.assertEqual(q_mat.shape, (2, 3, 1))
        self.assertTrue(torch.isfinite(q_mat).all())
        self.assertTrue(torch.isfinite(t_mat).all())
        torch.testing.assert_close(q_mat, q_initial)
        torch.testing.assert_close(t_mat, t_initial)
