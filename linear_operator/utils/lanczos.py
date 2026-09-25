#!/usr/bin/env python3
from __future__ import annotations

import torch

from linear_operator import settings


def lanczos_tridiag(
    matmul_closure,
    max_iter,
    dtype,
    device,
    matrix_shape,
    batch_shape=torch.Size(),
    init_vecs=None,
    num_init_vecs=1,
    tol=1e-5,
):
    """ """
    # Determine batch mode
    multiple_init_vecs = False

    if not callable(matmul_closure):
        raise RuntimeError(
            "matmul_closure should be a function callable object that multiples a (Lazy)Tensor "
            "by a vector. Got a {} instead.".format(matmul_closure.__class__.__name__)
        )

    # Get initial probe ectors - and define if not available
    if init_vecs is None:
        init_vecs = torch.randn(matrix_shape[-1], num_init_vecs, dtype=dtype, device=device)
        init_vecs = init_vecs.expand(*batch_shape, matrix_shape[-1], num_init_vecs)

    else:
        if settings.debug.on():
            if dtype != init_vecs.dtype:
                raise RuntimeError(
                    "Supplied dtype {} and init_vecs.dtype {} do not agree!".format(dtype, init_vecs.dtype)
                )
            if device != init_vecs.device:
                raise RuntimeError(
                    "Supplied device {} and init_vecs.device {} do not agree!".format(device, init_vecs.device)
                )
            if batch_shape != init_vecs.shape[:-2]:
                raise RuntimeError(
                    "batch_shape {} and init_vecs.shape {} do not agree!".format(batch_shape, init_vecs.shape)
                )
            if matrix_shape[-1] != init_vecs.size(-2):
                raise RuntimeError(
                    "matrix_shape {} and init_vecs.shape {} do not agree!".format(matrix_shape, init_vecs.shape)
                )

        num_init_vecs = init_vecs.size(-1)

    # Define some constants
    num_iter = min(max_iter, matrix_shape[-1])
    dim_dimension = -2

    if settings.verbose_linalg.on():
        settings.verbose_linalg.logger.debug(
            f"Running Lanczos on a {matrix_shape} matrix with a {init_vecs.shape} RHS for {num_iter} iterations."
        )

    # Create storage for q_mat, alpha,and beta
    # q_mat - batch version of Q - orthogonal matrix of decomp
    # alpha - batch version main diagonal of T
    # beta - batch version of off diagonal of T
    q_mat = torch.zeros(
        num_iter,
        *batch_shape,
        matrix_shape[-1],
        num_init_vecs,
        dtype=dtype,
        device=device,
    )
    t_mat = torch.zeros(num_iter, num_iter, *batch_shape, num_init_vecs, dtype=dtype, device=device)

    # Begin algorithm
    # Initial Q vector: q_0_vec
    q_0_vec = init_vecs / torch.linalg.vector_norm(init_vecs, ord=2, dim=dim_dimension).unsqueeze(dim_dimension)
    q_mat[0].copy_(q_0_vec)

    # Initial alpha value: alpha_0
    r_vec = matmul_closure(q_0_vec)
    alpha_0 = q_0_vec.mul(r_vec).sum(dim_dimension)

    # Initial beta value: beta_0
    r_vec.sub_(alpha_0.unsqueeze(dim_dimension).mul(q_0_vec))
    beta_0 = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension)

    # Copy over alpha_0 and beta_0 to t_mat
    t_mat[0, 0].copy_(alpha_0)
    t_mat[0, 1].copy_(beta_0)
    t_mat[1, 0].copy_(beta_0)

    # Compute the first new vector
    q_mat[1].copy_(r_vec.div_(beta_0.unsqueeze(dim_dimension)))

    # Now we start the iteration
    for k in range(1, num_iter):
        # Get previous values
        q_prev_vec = q_mat[k - 1]
        q_curr_vec = q_mat[k]
        beta_prev = t_mat[k, k - 1].unsqueeze(dim_dimension)

        # Compute next alpha value
        r_vec = matmul_closure(q_curr_vec) - q_prev_vec.mul(beta_prev)
        alpha_curr = q_curr_vec.mul(r_vec).sum(dim_dimension, keepdim=True)
        # Copy over to t_mat
        t_mat[k, k].copy_(alpha_curr.squeeze(dim_dimension))

        # Copy over alpha_curr, beta_curr to t_mat
        if (k + 1) < num_iter:
            # Compute next residual value
            r_vec.sub_(alpha_curr.mul(q_curr_vec))
            # Full reorthogonalization: r <- r - Q (Q^T r)
            correction = r_vec.unsqueeze(0).mul(q_mat[: k + 1]).sum(dim_dimension, keepdim=True)
            correction = q_mat[: k + 1].mul(correction).sum(0)
            r_vec.sub_(correction)
            r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)
            r_vec.div_(r_vec_norm)

            # Get next beta value
            beta_curr = r_vec_norm.squeeze_(dim_dimension)
            # Update t_mat with new beta value
            t_mat[k, k + 1].copy_(beta_curr)
            t_mat[k + 1, k].copy_(beta_curr)

            # Run more reorthoganilzation if necessary
            inner_products = q_mat[: k + 1].mul(r_vec.unsqueeze(0)).sum(dim_dimension)
            could_reorthogonalize = False
            for _ in range(10):
                if not torch.sum(inner_products > tol):
                    could_reorthogonalize = True
                    break
                correction = r_vec.unsqueeze(0).mul(q_mat[: k + 1]).sum(dim_dimension, keepdim=True)
                correction = q_mat[: k + 1].mul(correction).sum(0)
                r_vec.sub_(correction)
                r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)
                r_vec.div_(r_vec_norm)
                inner_products = q_mat[: k + 1].mul(r_vec.unsqueeze(0)).sum(dim_dimension)

            # Update q_mat with new q value
            q_mat[k + 1].copy_(r_vec)

            if torch.sum(beta_curr.abs() > 1e-6) == 0 or not could_reorthogonalize:
                break

    # Now let's transpose q_mat, t_mat intot the correct shape
    num_iter = k + 1

    # num_init_vecs x batch_shape x matrix_shape[-1] x num_iter
    q_mat = q_mat[:num_iter].permute(-1, *range(1, 1 + len(batch_shape)), -2, 0).contiguous()
    # num_init_vecs x batch_shape x num_iter x num_iter
    t_mat = t_mat[:num_iter, :num_iter].permute(-1, *range(2, 2 + len(batch_shape)), 0, 1).contiguous()

    # If we weren't in batch mode, remove batch dimension
    if not multiple_init_vecs:
        q_mat.squeeze_(0)
        t_mat.squeeze_(0)

    # We're done!
    return q_mat, t_mat


def lanczos_tridiag_to_diag(t_mat):
    """
    Given a num_init_vecs x num_batch x k x k tridiagonal matrix t_mat,
    returns a num_init_vecs x num_batch x k set of eigenvalues
    and a num_init_vecs x num_batch x k x k set of eigenvectors.

    TODO: make the eigenvalue computations done in batch mode.
    """
    orig_device = t_mat.device
    if settings.verbose_linalg.on():
        settings.verbose_linalg.logger.debug(f"Running symeig on a matrix of size {t_mat.shape}.")

    if t_mat.size(-1) < 32:
        retr = torch.linalg.eigh(t_mat.cpu())
    else:
        retr = torch.linalg.eigh(t_mat)

    evals, evecs = retr
    mask = evals.ge(0)
    evecs = evecs * mask.type_as(evecs).unsqueeze(-2)
    evals = evals.masked_fill_(~mask, 1)

    return evals.to(orig_device), evecs.to(orig_device)


def _postprocess_lanczos_root_inv_decomp(linear_op, inv_roots, initial_vectors, test_vectors):
    """
    Given linear_op and a set of inv_roots of shape num_init_vecs x num_batch x n x k,
    as well as the initial vectors of shape num_init_vecs x num_batch x n,
    determine which inverse root is best given the test_vectors of shape
    num_init_vecs x num_batch x n
    """
    num_probes = initial_vectors.size(-1)
    test_vectors = test_vectors.unsqueeze(0)

    # Compute solves
    solves = inv_roots.matmul(inv_roots.mT.matmul(test_vectors))

    # Compute linear_op * solves
    solves = (
        solves.permute(*range(1, linear_op.dim() + 1), 0)
        .contiguous()
        .view(*linear_op.batch_shape, linear_op.matrix_shape[-1], -1)
    )
    mat_times_solves = linear_op.matmul(solves)
    mat_times_solves = mat_times_solves.view(
        *linear_op.batch_shape, linear_op.matrix_shape[-1], -1, num_probes
    ).permute(-1, *range(0, linear_op.dim()))

    # Compute residuals
    residuals = (mat_times_solves - test_vectors).norm(2, dim=-2)
    residuals = residuals.view(residuals.size(0), -1).sum(-1)

    # Choose solve that best fits
    _, best_solve_index = residuals.min(0)
    inv_root = inv_roots[best_solve_index].squeeze(0)
    return inv_root

def extend_lanczos_basis(
    matmul_closure,
    max_iter,
    dtype,
    device,
    matrix_shape,
    q_mat,
    t_mat,
    tol=1e-6,
):
    """Extend an existing Lanczos-type basis.

    This prototype starts from an already available orthonormal basis ``q_mat``
    and projected matrix ``t_mat``. In the CG-based experiments, these are
    obtained from stored CG directions. The routine then continues the Lanczos
    recurrence until either ``max_iter`` vectors have been reached, numerical
    breakdown occurs, or the reorthogonalization step fails.

    The input basis is expected in the external convention

        q_mat: [batch, n, J],
        t_mat: [batch, J, J],

    where ``J`` is the current basis size. Internally, the function uses the
    same leading-iteration convention as ``linear_operator``'s Lanczos routine,

        q_ext: [J, batch, n],
        t_ext: [J, J, batch, 1],

    and converts back before returning.

    Args:
        matmul_closure: Callable implementing multiplication by the system
            matrix.
        max_iter: Maximum final basis size.
        dtype: Floating point dtype used for the basis and projected matrix.
        device: Device used for the computation.
        matrix_shape: Shape of the system matrix. Only the final dimension is
            used to cap the maximum number of iterations.
        q_mat: Existing orthonormal basis with shape ``[batch, n, J]``.
        t_mat: Projected matrix for ``q_mat`` with shape ``[batch, J, J]``.
        tol: Tolerance used for breakdown and reorthogonalization checks.

    Returns:
        A tuple ``(q_final, t_final)`` where ``q_final`` has shape
        ``[batch, n, J_final]`` and ``t_final`` has shape
        ``[batch, J_final, J_final]``.
    """

    if not callable(matmul_closure):
        raise RuntimeError(
            "matmul_closure should be a callable object that multiplies by a matrix."
        )

    q_mat = q_mat.to(dtype=dtype, device=device)
    t_mat = t_mat.to(dtype=dtype, device=device)

    squeeze_batch = False
    if q_mat.dim() == 2:
        q_mat = q_mat.unsqueeze(0)
        t_mat = t_mat.unsqueeze(0)
        squeeze_batch = True

    if q_mat.dim() != 3:
        raise ValueError("This prototype expects q_mat with shape [batch, n, J] or [n, J].")

    batch_shape = q_mat.shape[:-2]
    num_rows = q_mat.size(-2)
    current_iter = q_mat.size(-1)
    target_iter = min(max_iter, matrix_shape[-1])
    dim_dimension = -2

    if matrix_shape[-1] != num_rows:
        raise ValueError("matrix_shape and q_mat do not agree.")

    if current_iter == 0:
        raise ValueError("q_mat must contain at least one basis vector.")

    if target_iter <= 0:
        raise ValueError("max_iter must be positive.")

    # If no extension is needed, truncate and recompute the projected matrix
    if target_iter <= current_iter:
        q_final = q_mat[..., :, :target_iter].contiguous()
        t_final = torch.matmul(q_final.transpose(-1,-2), matmul_closure(q_final))
        t_final = 0.5 * (t_final + t_final.transpose(-1, -2))
        if squeeze_batch:
            q_final = q_final.squeeze(0)
            t_final = t_final.squeeze(0)
        return q_final, t_final

    # Convert q_mat from [batch, n, J] to [J, batch, n]
    q_ext = q_mat.new_zeros(target_iter, *batch_shape, num_rows)
    q_ext[:current_iter].copy_(q_mat.permute(-1, *range(len(batch_shape)), -2))

    # Convert t_mat from [batch, J, J] to [J, J, batch, 1]
    t_ext = t_mat.new_zeros(target_iter, target_iter, *batch_shape, 1)
    t_ext[:current_iter, :current_iter].copy_(t_mat.permute(-2, -1, *range(len(batch_shape))).unsqueeze(-1))

    # Start from the last available basis vector
    q = q_ext[current_iter - 1].unsqueeze(-1)

    # Initial residual for the first new Lanczos vector
    r_vec = matmul_closure(q)

    if r_vec.shape != q.shape:
        raise ValueError("matmul_closure must return a tensor with the same shape as the basis vectors.")

    # Remove the known recurrence components from the last stored vector
    alpha_last = t_ext[current_iter - 1, current_iter - 1]
    r_vec.sub_(q.mul(alpha_last))

    if current_iter > 1:
        beta_prev = t_ext[current_iter - 2, current_iter - 1]
        q_prev = q_ext[current_iter - 2].unsqueeze(-1)
        r_vec.sub_(q_prev.mul(beta_prev))

    # Reorthogonalize against all previously stored basis vectors
    q_prev_all = q_ext[: current_iter]

    correction = r_vec.unsqueeze(0).mul(q_prev_all.unsqueeze(-1)).sum(dim=dim_dimension,keepdim=True)
    correction = q_prev_all.unsqueeze(-1).mul(correction).sum(0)
    r_vec.sub_(correction)

    r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)

    inner_products = q_prev_all.unsqueeze(-1).mul(r_vec.div(r_vec_norm).unsqueeze(0)).sum(dim_dimension)

    could_reorthogonalize = False

    # Repeat reorthogonalization if necessary
    for _ in range(10):
        if not torch.sum(inner_products.abs() > tol):
            could_reorthogonalize = True
            break

        correction = r_vec.unsqueeze(0).mul(q_prev_all.unsqueeze(-1)).sum(dim=dim_dimension,keepdim=True)
        correction = q_prev_all.unsqueeze(-1).mul(correction).sum(0)
        r_vec.sub_(correction)

        r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)


        inner_products = q_prev_all.unsqueeze(-1).mul(r_vec.div(r_vec_norm).unsqueeze(0)).sum(dim_dimension)

    if not could_reorthogonalize:
        target_iter = current_iter

    num_iter = current_iter

    # Continue the Lanczos recurrence
    for k in range(current_iter, target_iter):
        beta = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)

        # Store the new off-diagonal coefficient
        t_ext[k - 1, k].copy_(beta.squeeze(-1))
        t_ext[k, k - 1].copy_(beta.squeeze(-1))

        if torch.sum(beta.abs() > tol) == 0:
            break

        # Normalize the residual to obtain the next Lanczos vector
        q_prev = q
        q = r_vec.div(beta)

        q_ext[k].copy_(q.squeeze(-1))
        num_iter += 1

        r_vec = matmul_closure(q)

        # Compute and store the diagonal coefficient
        alpha = torch.sum(q * r_vec, dim=dim_dimension, keepdim=True)
        t_ext[k, k].copy_(alpha.squeeze(-1))

        # Remove the three-term recurrence components
        r_vec.sub_(q.mul(alpha))
        r_vec.sub_(q_prev.mul(beta))

        # Full reorthogonalization against the current basis
        q_prev_all = q_ext[: k + 1]

        correction = r_vec.unsqueeze(0).mul(q_prev_all.unsqueeze(-1)).sum(dim=dim_dimension,keepdim=True)
        correction = q_prev_all.unsqueeze(-1).mul(correction).sum(0)
        r_vec.sub_(correction)

        r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)

        inner_products = q_prev_all.unsqueeze(-1).mul(r_vec.div(r_vec_norm).unsqueeze(0)).sum(dim_dimension)

        could_reorthogonalize = False

        # Repeat reorthogonalization if necessary
        for _ in range(10):
            if not torch.sum(inner_products.abs() > tol):
                could_reorthogonalize = True
                break

            correction = r_vec.unsqueeze(0).mul(q_prev_all.unsqueeze(-1)).sum(dim=dim_dimension,keepdim=True)
            correction = q_prev_all.unsqueeze(-1).mul(correction).sum(0)
            r_vec.sub_(correction)

            r_vec_norm = torch.linalg.vector_norm(r_vec, ord=2, dim=dim_dimension, keepdim=True)

            inner_products = q_prev_all.unsqueeze(-1).mul(r_vec.div(r_vec_norm).unsqueeze(0)).sum(dim_dimension)


        if not could_reorthogonalize:
            break

    q_final = q_ext[:num_iter].permute(*range(1, 1 + len(batch_shape)), -1, 0).contiguous()
    t_final = t_ext[:num_iter, :num_iter].squeeze(-1).permute(*range(2, 2 + len(batch_shape)), 0, 1).contiguous()
    t_final = 0.5 * (t_final + t_final.transpose(-1, -2))
    if squeeze_batch:
        q_final = q_final.squeeze(0)
        t_final = t_final.squeeze(0)
    return q_final, t_final
