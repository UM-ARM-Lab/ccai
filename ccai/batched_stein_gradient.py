import torch
from torch_cg import cg_batch


def compute_constrained_gradient(xuz, starts, goals, constraints, problem, alpha_J=1, alpha_C=1):
    """

    :param xuz: B x N x T(dx + du + dz)
    :param problem:
    :return:
    """
    B, N = xuz.shape[:2]
    d = problem.dx + problem.du
    xuz = xuz.detach()
    xuz.requires_grad = True
    J, grad_J, hess_J, K, grad_K, C, dC, hess_C = problem.eval(xuz, starts, goals, constraints)

    with torch.no_grad():
        # we try and invert the dC dCT, if it is singular then we use the psuedo-inverse
        eye = torch.eye(problem.dg + problem.dh).repeat(B, N, 1, 1).to(device=C.device)
        dCdCT = dC @ dC.permute(0, 1, 3, 2)
        A_bmm = lambda x: dCdCT @ x

        try:
            dCdCT_inv = torch.linalg.solve(dCdCT, eye)
            if torch.any(torch.isnan(dCdCT_inv)):
                raise ValueError('nan in inverse')
        except Exception as e:
            print(e)
            # dCdCT_inv = torch.linalg.lstsq(dC @ dC.permute(0, 2, 1), eye).solution
            dCdCT_inv = torch.linalg.pinv(dCdCT)
            #dCdCT_inv, _ = cg_batch(A_bmm, eye, verbose=False)
        # get projection operator
        projection = dCdCT_inv @ dC
        eye = torch.eye(d * problem.T + problem.dh, device=xuz.device, dtype=xuz.dtype).unsqueeze(0)
        projection = eye - dC.permute(0, 1, 3, 2) @ projection

        # compute term for repelling towards constraint
        xi_C = dCdCT_inv @ C.unsqueeze(-1)
        xi_C = (dC.permute(0, 1, 3, 2) @ xi_C).squeeze(-1)

        # compute gradient for projection
        # now the second index (1) is the
        # x with which we are differentiating
        dCdCT_inv = dCdCT_inv.unsqueeze(2)

        dCT = dC.permute(0, 1, 3, 2).unsqueeze(2)
        dC = dC.unsqueeze(2)

        hess_C = hess_C.permute(0, 1, 4, 2, 3)
        hess_CT = hess_C.permute(0, 1, 2, 4, 3)

        # compute first term
        first_term = hess_CT @ (dCdCT_inv @ dC)
        second_term = dCT @ dCdCT_inv @ (hess_C @ dCT + dC @ hess_CT) @ dCdCT_inv @ dC
        third_term = dCT @ dCdCT_inv @ hess_C

        # add terms and permute so last dimension is the x which we are differentiating w.r.t
        grad_projection = (first_term - second_term + third_term).permute(0, 1, 3, 4, 2)

        # compute total gradient of kernel
        # first term is grad of scalar
        grad_proj = torch.einsum('bmijj->bmij', grad_projection)

        # now we need to combine all the different projections together
        PP = projection.unsqueeze(1) @ projection.unsqueeze(2)  # should now be B xN x N x D x D
        PQ = projection
        first_term = torch.einsum('bnmj, bnmij->bnmi', grad_K, PP)
        matrix_K = K.reshape(B, N, N, 1, 1) * projection.unsqueeze(1) @ projection.unsqueeze(2)

        grad_matrix_K = first_term
        second_term = K.reshape(B, N, N, 1, 1) * PQ.unsqueeze(1) @ grad_proj.unsqueeze(2)
        second_term = torch.sum(second_term, dim=4)
        grad_matrix_K = grad_matrix_K + second_term

        grad_matrix_K = torch.sum(grad_matrix_K, dim=1)

        # compute kernelized score
        kernelized_score = torch.sum(matrix_K @ -grad_J.reshape(B, N, 1, -1, 1), dim=1)
        phi = kernelized_score.squeeze(-1) / N + grad_matrix_K / N  # maximize phi
        xi_J = -phi

    return (alpha_J * xi_J + alpha_C * xi_C).detach(), J, C
