#!/usr/bin/env python3
from __future__ import annotations

import torch


def recover_lanczos_cache_from_cg_directions(
        d_mat, 
        kd_mat,
        rank_tol=None,
        eps=None,
    ):
    if rank_tol is None:
        rank_tol = 1e-10 if d_mat.dtype == torch.float64 else 1e-5
    if eps is None:
        eps = torch.finfo(d_mat.dtype).eps

    q_mat, r_mat = torch.linalg.qr(d_mat, mode="reduced")

    diag_r = torch.diagonal(r_mat, dim1=-2, dim2=-1).abs()
    rel_diag_r = diag_r / diag_r[..., :1].clamp_min(eps)
    valid = (rel_diag_r > rank_tol).to(torch.int64).cumprod(dim=-1).bool()
    num_keep = int(valid.to(torch.int64).sum(dim=-1).min().item())

    if num_keep <= 1:
        return None, None

    q_mat = q_mat[..., :, :num_keep]
    r_mat = r_mat[..., :num_keep, :num_keep]
    kd_mat = kd_mat[..., :, :num_keep]

    kq_mat_t = torch.linalg.solve(r_mat.transpose(-1, -2), kd_mat.transpose(-1, -2))

    t_mat = kq_mat_t.matmul(q_mat)
    t_mat = 0.5 * (t_mat + t_mat.transpose(-1, -2))

    return q_mat, t_mat