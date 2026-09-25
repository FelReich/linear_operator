#!/usr/bin/env python3
from __future__ import annotations

import warnings

import torch

from linear_operator import settings
from linear_operator.utils.warnings import NumericalWarning


def _default_preconditioner(x):
    return x.clone()


def linear_cg(
    matmul_closure,
    rhs,
    n_tridiag=0,
    tolerance=None,
    eps=1e-10,
    stop_updating_after=1e-10,
    max_iter=None,
    max_tridiag_iter=None,
    initial_guess=None,
    preconditioner=None,
    save_directions=False,
):
    """
    Implements the linear conjugate gradients method for (approximately) solving systems of the form

        lhs result = rhs

    for positive definite and symmetric matrices.

    Args:
      - matmul_closure - a function which performs a left matrix multiplication with lhs_mat
      - rhs - the right-hand side of the equation
      - n_tridiag - returns a tridiagonalization of the first n_tridiag columns of rhs
      - tolerance - stop the solve when the (average) norm of the residual(s) is less than this
      - eps - noise to add to prevent division by zero
      - stop_updating_after - will stop updating a vector after this residual norm is reached
      - max_iter - the maximum number of CG iterations
      - max_tridiag_iter - the maximum size of the tridiagonalization matrix
      - initial_guess - an initial guess at the solution `result`
      - precondition_closure - a functions which left-preconditions a supplied vector
      - save_directions - store CG search directions and matrix-vector products for CG-Lanczos variance estimates

    Returns:
      result - a solution to the system (if n_tridiag is 0)
      result, tridiags - a solution to the system, and corresponding tridiagonal matrices (if n_tridiag > 0)
      result, d_mat, kd_mat - a solution, stored directions, and matrix-vector products (if save_directions)
    """
    # Unsqueeze, if necesasry
    is_vector = rhs.ndimension() == 1
    if is_vector:
        rhs = rhs.unsqueeze(-1)

    # Some default arguments
    if max_iter is None:
        max_iter = settings.max_cg_iterations.value()
    if max_tridiag_iter is None:
        max_tridiag_iter = settings.max_lanczos_quadrature_iterations.value()
    if initial_guess is None:
        initial_guess = torch.zeros_like(rhs)
    else:
        # Unsqueeze, if necesasry
        is_vector = initial_guess.ndimension() == 1
        if is_vector:
            initial_guess = initial_guess.unsqueeze(-1)
    if tolerance is None:
        tolerance = settings.cg_tolerance.value()
    precond = preconditioner is not None
    if preconditioner is None:
        preconditioner = _default_preconditioner

    # If we are running m CG iterations, we obviously can't get more than m Lanczos coefficients
    if max_tridiag_iter > max_iter:
        raise RuntimeError("Getting a tridiagonalization larger than the number of CG iterations run is not possible!")

    # Check matmul_closure object
    if torch.is_tensor(matmul_closure):
        matmul_closure = matmul_closure.matmul
    elif not callable(matmul_closure):
        raise RuntimeError("matmul_closure must be a tensor, or a callable object!")

    if save_directions and n_tridiag:
        raise NotImplementedError("save_directions cannot currently be combined with n_tridiag.")
    if save_directions and precond:
        raise NotImplementedError("save_directions currently supports only the unpreconditioned case.")
    if save_directions and rhs.size(-1) != 1:
        raise NotImplementedError("save_directions currently supports only a single right-hand side.")

    # Get some constants
    num_rows = rhs.size(-2)
    n_iter = min(max_iter, num_rows) if settings.terminate_cg_by_size.on() else max_iter
    n_tridiag_iter = min(max_tridiag_iter, num_rows)
    eps = torch.tensor(eps, dtype=rhs.dtype, device=rhs.device)

    # Get the norm of the rhs - used for convergence checks
    # Here we're going to make almost-zero norms actually be 1 (so we don't get divide-by-zero issues)
    # But we'll store which norms were actually close to zero
    rhs_norm = rhs.norm(2, dim=-2, keepdim=True)
    rhs_is_zero = rhs_norm.lt(eps)
    rhs_norm = rhs_norm.masked_fill_(rhs_is_zero, 1)

    # Let's normalize. We'll un-normalize afterwards
    rhs = rhs.div(rhs_norm)
    initial_guess = initial_guess.div(rhs_norm)

    # residual: residual_{0} = b_vec - lhs x_{0}
    residual = rhs - matmul_closure(initial_guess)
    batch_shape = residual.shape[:-2]

    # result <- x_{0}
    result = initial_guess.expand_as(residual).contiguous()

    # Maybe log
    if settings.verbose_linalg.on():
        settings.verbose_linalg.logger.debug(
            f"Running CG on a {rhs.shape} RHS for {n_iter} iterations (tol={tolerance}). Output: {result.shape}."
        )

    # Check for NaNs
    if not torch.equal(residual, residual):
        raise RuntimeError("NaNs encountered when trying to perform matrix-vector multiplication")

    # Sometime we're lucky and the preconditioner solves the system right away
    # Check for convergence
    residual_norm = residual.norm(2, dim=-2, keepdim=True)
    has_converged = torch.lt(residual_norm, stop_updating_after)

    if has_converged.all() and not n_tridiag:
        n_iter = 0  # Skip the iteration!

    # Otherwise, let's define precond_residual and curr_conjugate_vec
    else:
        # precon_residual{0} = M^-1 residual_{0}
        precond_residual = preconditioner(residual)
        curr_conjugate_vec = precond_residual
        residual_inner_prod = precond_residual.mul(residual).sum(-2, keepdim=True)

        # Define storage matrices
        mul_storage = torch.empty_like(residual)
        alpha = torch.empty(*batch_shape, 1, rhs.size(-1), dtype=residual.dtype, device=residual.device)
        beta = torch.empty_like(alpha)
        is_zero = torch.empty(*batch_shape, 1, rhs.size(-1), dtype=torch.bool, device=residual.device)

    # Define tridiagonal matrices, if applicable
    if n_tridiag:
        t_mat = torch.zeros(
            n_tridiag_iter,
            n_tridiag_iter,
            *batch_shape,
            n_tridiag,
            dtype=alpha.dtype,
            device=alpha.device,
        )
        alpha_tridiag_is_zero = torch.empty(*batch_shape, n_tridiag, dtype=torch.bool, device=t_mat.device)
        alpha_reciprocal = torch.empty(*batch_shape, n_tridiag, dtype=t_mat.dtype, device=t_mat.device)
        prev_alpha_reciprocal = torch.empty_like(alpha_reciprocal)
        prev_beta = torch.empty_like(alpha_reciprocal)

    update_tridiag = True
    last_tridiag_iter = 0

    # It's conceivable we reach the tolerance on the last iteration, so can't just check iteration number.
    tolerance_reached = False

    num_stored = 0
    if save_directions:
        d_mat = rhs.new_zeros(n_iter, *batch_shape, num_rows)
        kd_mat = rhs.new_zeros(n_iter, *batch_shape, num_rows)
        save_directions_cg = True
    else:
        save_directions_cg = False

    # Start the iteration
    for k in range(n_iter):
        # Get next alpha
        # alpha_{k} = (residual_{k-1}^T precon_residual{k-1}) / (p_vec_{k-1}^T mat p_vec_{k-1})
        mvms = matmul_closure(curr_conjugate_vec)

        direction_was_reorthogonalized = False
        if save_directions_cg:
            if k > 0:
                d_prev = d_mat[:k]
                kd_prev = kd_mat[:k]
                direction_was_reorthogonalized = True

                could_reorthogonalize = False
                for _ in range(10):
                    dkd = torch.mul(d_prev, kd_prev).sum(dim=-1)
                    dkd_is_zero = torch.lt(dkd.abs(), eps)
                    dkd.masked_fill_(dkd_is_zero, 1.0)

                    coeffs = torch.mul(kd_prev, curr_conjugate_vec.squeeze(-1)).sum(dim=-1).div(dkd)
                    coeffs.masked_fill_(dkd_is_zero, 0.0)

                    curr_conjugate_vec.sub_((d_prev * coeffs.unsqueeze(-1)).sum(dim=0).unsqueeze(-1))
                    mvms.sub_((kd_prev * coeffs.unsqueeze(-1)).sum(dim=0).unsqueeze(-1))

                    inner_products = torch.mul(kd_prev, curr_conjugate_vec.squeeze(-1)).sum(dim=-1)
                    new_dkd = torch.mul(curr_conjugate_vec.squeeze(-1), mvms.squeeze(-1)).sum(dim=-1)
                    if new_dkd.dim() == 0:
                        scale = torch.sqrt(dkd.abs().mul(new_dkd.abs()))
                    else:
                        scale = torch.sqrt(torch.matmul(dkd.abs(), new_dkd.abs()))
                    rel_inner_products = inner_products.abs().div(scale.clamp_min(eps))

                    if not torch.sum(rel_inner_products.abs() > tolerance):
                        could_reorthogonalize = True
                        break

                if not could_reorthogonalize:
                    save_directions_cg = False
                    num_stored = k
                    if settings.cg_lanczos_aggressive_mean_stop.on():
                        tolerance_reached = True
                        break

            if save_directions_cg:
                d_mat[k].copy_(curr_conjugate_vec.squeeze(-1))
                kd_mat[k].copy_(mvms.squeeze(-1))
                num_stored = k + 1

        torch.mul(curr_conjugate_vec, mvms, out=mul_storage)
        torch.sum(mul_storage, -2, keepdim=True, out=alpha)

        # Do a safe division here
        torch.lt(alpha, eps, out=is_zero)
        alpha.masked_fill_(is_zero, 1)
        if direction_was_reorthogonalized:
            alpha_numerator = torch.mul(residual, curr_conjugate_vec).sum(dim=-2, keepdim=True)
        else:
            alpha_numerator = residual_inner_prod
        torch.div(alpha_numerator, alpha, out=alpha)
        alpha.masked_fill_(is_zero, 0)

        # We'll cancel out any updates by setting alpha=0 for any vector that has already converged
        alpha.masked_fill_(has_converged, 0)

        # Update residual
        # residual_{k} = residual_{k-1} - alpha_{k} mat p_vec_{k-1}
        residual = torch.addcmul(residual, alpha, mvms, value=-1, out=residual)

        # Update precond_residual
        # precon_residual{k} = M^-1 residual_{k}
        precond_residual = preconditioner(residual)

        # Update result
        # result_{k} = result_{k-1} + alpha_{k} p_vec_{k-1}
        result = torch.addcmul(result, alpha, curr_conjugate_vec, out=result)

        # beta_{k} = (precon_residual{k}^T r_vec_{k}) / (precon_residual{k-1}^T r_vec_{k-1})
        beta.resize_as_(residual_inner_prod).copy_(residual_inner_prod)
        torch.mul(residual, precond_residual, out=mul_storage)
        torch.sum(mul_storage, -2, keepdim=True, out=residual_inner_prod)

        # Do a safe division here
        torch.lt(beta, eps, out=is_zero)
        beta.masked_fill_(is_zero, 1)
        torch.div(residual_inner_prod, beta, out=beta)
        beta.masked_fill_(is_zero, 0)

        # Update curr_conjugate_vec
        # curr_conjugate_vec_{k} = precon_residual{k} + beta_{k} curr_conjugate_vec_{k-1}
        curr_conjugate_vec.mul_(beta).add_(precond_residual)

        torch.linalg.vector_norm(residual, ord=2, dim=-2, keepdim=True, out=residual_norm)
        residual_norm.masked_fill_(rhs_is_zero, 0)
        torch.lt(residual_norm, stop_updating_after, out=has_converged)

        if (
            k >= min(10, max_iter - 1)
            and bool(residual_norm.mean() < tolerance)
            and not (n_tridiag and k < min(n_tridiag_iter, max_iter - 1))
        ):
            tolerance_reached = True
            break

        # Update tridiagonal matrices, if applicable
        if n_tridiag and k < n_tridiag_iter and update_tridiag:
            alpha_tridiag = alpha.squeeze(-2).narrow(-1, 0, n_tridiag)
            beta_tridiag = beta.squeeze(-2).narrow(-1, 0, n_tridiag)
            torch.eq(alpha_tridiag, 0, out=alpha_tridiag_is_zero)
            alpha_tridiag.masked_fill_(alpha_tridiag_is_zero, 1)
            torch.reciprocal(alpha_tridiag, out=alpha_reciprocal)
            alpha_tridiag.masked_fill_(alpha_tridiag_is_zero, 0)

            if k == 0:
                t_mat[k, k].copy_(alpha_reciprocal)
            else:
                torch.addcmul(alpha_reciprocal, prev_beta, prev_alpha_reciprocal, out=t_mat[k, k])
                torch.mul(prev_beta.sqrt_(), prev_alpha_reciprocal, out=t_mat[k, k - 1])
                t_mat[k - 1, k].copy_(t_mat[k, k - 1])

                if t_mat[k - 1, k].max() < 1e-6:
                    update_tridiag = False

            last_tridiag_iter = k

            prev_alpha_reciprocal.copy_(alpha_reciprocal)
            prev_beta.copy_(beta_tridiag)

    # Un-normalize
    result = result.mul(rhs_norm)

    if not tolerance_reached and n_iter > 0:
        warnings.warn(
            "CG terminated in {} iterations with average residual norm {}"
            " which is larger than the tolerance of {} specified by"
            " linear_operator.settings.cg_tolerance."
            " If performance is affected, consider raising the maximum number of CG iterations by running code in"
            " a linear_operator.settings.max_cg_iterations(value) context.".format(
                k + 1, residual_norm.mean(), tolerance
            ),
            NumericalWarning,
        )

    if is_vector:
        result = result.squeeze(-1)

    if save_directions:
        d_mat = d_mat[:num_stored].permute(*range(1, 1 + len(batch_shape)), -1, 0).contiguous()
        kd_mat = kd_mat[:num_stored].permute(*range(1, 1 + len(batch_shape)), -1, 0).contiguous()
        return result, d_mat, kd_mat

    if n_tridiag:
        t_mat = t_mat[: last_tridiag_iter + 1, : last_tridiag_iter + 1]
        return (
            result,
            t_mat.permute(-1, *range(2, 2 + len(batch_shape)), 0, 1).contiguous(),
        )
    else:
        return result



def cg_store_lanczos_basis(
    matmul_closure,
    rhs,
    n_tridiag=0,
    tolerance=None,
    eps=1e-10,
    stop_updating_after=1e-10,
    max_iter=None,
    max_tridiag_iter=None,
    initial_guess=None,
    preconditioner=None,
):
    """Run CG and recover a Lanczos-type basis from stored search directions."""
    if preconditioner is not None:
        raise NotImplementedError("cg_store_lanczos_basis currently supports only the unpreconditioned case.")

    result, d_mat, kd_mat = linear_cg(
        matmul_closure,
        rhs,
        n_tridiag=n_tridiag,
        tolerance=tolerance,
        eps=eps,
        stop_updating_after=stop_updating_after,
        max_iter=max_iter,
        max_tridiag_iter=max_tridiag_iter,
        initial_guess=initial_guess,
        preconditioner=None,
        save_directions=True,
    )

    q_mat, r_mat = torch.linalg.qr(d_mat, mode="reduced")

    rank_tol = 1e-10 if d_mat.dtype == torch.float64 else 1e-5
    diag_r = torch.diagonal(r_mat, dim1=-2, dim2=-1).abs()
    rel_diag_r = diag_r / diag_r[..., :1].clamp_min(eps)
    valid = (rel_diag_r > rank_tol).to(torch.int64).cumprod(dim=-1).bool()
    num_keep = int(valid.to(torch.int64).sum(dim=-1).min().item()) if valid.dim() > 1 else int(valid.sum().item())

    if num_keep == 0:
        raise RuntimeError("All stored CG directions were discarded by the QR rank cutoff.")

    q_mat = q_mat[..., :, :num_keep]
    r_mat = r_mat[..., :num_keep, :num_keep]
    kd_mat = kd_mat[..., :, :num_keep]

    kq_mat_t = torch.linalg.solve(r_mat.transpose(-1, -2), kd_mat.transpose(-1, -2))
    t_mat = kq_mat_t.matmul(q_mat)
    t_mat = 0.5 * (t_mat + t_mat.transpose(-1, -2))

    return result, q_mat, t_mat
