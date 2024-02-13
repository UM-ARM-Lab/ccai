#include <torch/extension.h>

/**
 * LibTorch implementation of compute_update after calculating dCdCT_inv.
 */
at::Tensor after_inverse(
    at::Tensor C,
    at::Tensor dC,
    at::Tensor hess_C,
    at::Tensor dCdCT_inv,
    at::Tensor K,
    at::Tensor grad_J,
    at::Tensor grad_K) {

    int N = 8;
    int gamma = 1;
    float alpha_J = 0.1;
    int alpha_C = 1;
    int T = 12;
    int d = 16;
    int dh = 0;

    auto projection = dCdCT_inv.matmul(dC);
    auto options = torch::TensorOptions().dtype(C.dtype()).device(C.device().type(), C.device().index());
    auto eye = torch::eye(d * T + dh, options).unsqueeze(0);
    projection = eye - dC.permute({0, 2, 1}).matmul(projection);

    auto xi_C = dCdCT_inv.matmul(C.unsqueeze(-1));
    xi_C = dC.permute({0, 2, 1}).matmul(xi_C).squeeze(-1);

    dCdCT_inv = dCdCT_inv.unsqueeze(1);

    auto dCT = dC.permute({0, 2, 1}).unsqueeze(1);
    dC = dC.unsqueeze(1);

    hess_C = hess_C.permute({0, 3, 1, 2});
    auto hess_CT = hess_C.permute({0, 1, 3, 2});

    auto first_term = at::matmul(hess_CT, at::matmul(dCdCT_inv, dC));
    auto second_term = at::matmul(
        dCT,
        at::matmul(dCdCT_inv,
        at::matmul(at::matmul(hess_C, dCT) + at::matmul(dC, hess_CT),
        at::matmul(dCdCT_inv,
        dC))));
    auto third_term = at::matmul(dCT, at::matmul(dCdCT_inv, hess_C));
    auto grad_projection = (first_term - second_term + third_term).permute({0, 2, 3, 1});
    auto grad_proj = at::einsum("mijj->mij", grad_projection);

    // now we need to combine all the different projections together
    auto PP = projection.unsqueeze(0).matmul(projection.unsqueeze(1));  // should now be N x N x D x D
    auto matrix_K = K.reshape({N, N, 1, 1}) * projection.unsqueeze(0).matmul(projection.unsqueeze(1));
    auto grad_matrix_K = at::einsum("nmj, nmij->nmi", {grad_K, PP}) + (K.reshape({N, N, 1, 1}) * projection.unsqueeze(0).matmul(grad_proj.unsqueeze(1))).sum(3);
    grad_matrix_K = grad_matrix_K.sum(0);

    // compute kernelized score
    auto kernelized_score = at::sum(matrix_K.matmul(-grad_J.reshape({N, 1, -1, 1})), 0);
    auto phi = gamma * kernelized_score.squeeze(-1) / N +  grad_matrix_K / N;
    auto xi_J = -phi;
    return at::detach(alpha_J * xi_J + alpha_C * xi_C);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("after_inverse", &after_inverse);
}
