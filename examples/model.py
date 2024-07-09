from collections import namedtuple

# from controller import QPControllerLayer
from vae import VAE
import torch
import numpy as np
import matplotlib.pyplot as plt


loss_dict = namedtuple('loss_dict', ['dyn_recon_loss', 'constraint_recon_loss', 'dyn_loss', 'constraint_loss', 'constraint_reg_loss', 'connect_loss', 's', 'u'])
loss_dict_diff = namedtuple('loss_dict', ['recon_loss', 'constraint_loss'])
# loss_dict_diff = namedtuple('loss_dict', ['recon_loss'])
class ConstraintEmbedding(torch.nn.Module):
    """
    Embedding for constraints

    Takes in a constraint idx and outputs its embedding
    """

    def __init__(self, num_constraints, embedding_dim):
        super(ConstraintEmbedding, self).__init__()
        #Embedding matrix
        self.embedding = torch.nn.Embedding(num_constraints+2, embedding_dim, padding_idx=0)
        # self.embedding = torch.nn.Embedding(num_constraints+1, embedding_dim * 2, padding_idx=0)
    def forward(self, constraint_idx):
        return self.embedding(constraint_idx)

class LatentDiffusionModel(torch.nn.Module):
    def __init__(self, config, problem_for_sampler):
        super(LatentDiffusionModel, self).__init__()
        self.nx = config['nx']
        self.nu = config['nu']
        self.nc = config['nc']
        self.nzx = config['nzx']
        self.nzu = config['nzu']
        self.nzt = config['nzt']
        self.nhidden = config['nhidden']
        self.xu_dim = self.nx + self.nu

        self.vae_x = VAE(self.nx, self.nzx, self.nc, self.nhidden)
        self.vae_u = VAE(self.nu, self.nzu, self.nc, self.nhidden)
        self.vae_t = VAE((self.nzx*2+self.nzu), self.nzt, self.nc, self.nhidden)

        self.lambda_recon = config['lambda_recon']
        self.lambda_constraint = config['lambda_constraint']

        self.problem_for_sampler = problem_for_sampler
        self.problem_key_tensors = []

    def encode(self, x, u, x_t, c_type):
        z_x, mu_x, logvar_x = self.vae_x.encode(x, c_type)
        z_x_t, mu_x_t, logvar_x_t = self.vae_x.encode(x_t, c_type)
        z_u, mu_u, logvar_u = self.vae_u.encode(u, c_type)
        z_t, mu_t, logvar_t = self.vae_t.encode(torch.cat([z_x, z_u, z_x_t], dim=-1), c_type)
        # z_t, mu_t, logvar_t = self.vae_t.encode(torch.cat([mu_x, logvar_x, mu_u, logvar_u, mu_x_t, logvar_x_t], dim=-1))
        return z_x, mu_x, logvar_x, z_u, mu_u, logvar_u, z_t, mu_t, logvar_t, z_x_t, mu_x_t, logvar_x_t
    
    def make_csvto_traj(self, x_deocde, u_decode, x_t_decode):
        t_0 = torch.cat((x_deocde, u_decode), dim=-1)
        t_1 = torch.cat((x_t_decode, torch.zeros_like(u_decode)), dim=-1)
        return torch.stack((t_0, t_1), dim=1)

    def forward(self, x, u, x_t, c_type):

        # Encode x, u, x_t
        z_x, mu_x, logvar_x, z_u, mu_u, logvar_u, z_t, mu_t, logvar_t, z_x_t, mu_x_t, logvar_x_t = self.encode(x, u, x_t, c_type)

        z_x_decode_direct = self.vae_x.decode(z_x, c_type)
        z_u_decode_direct = self.vae_u.decode(z_u, c_type)
        z_x_t_decode_direct = self.vae_x.decode(z_x_t, c_type)

        z_t_decode = self.vae_t.decode(z_t, c_type)

        z_x_decode_indirect = self.vae_x.decode(z_t_decode[..., :self.nzx], c_type)
        z_u_decode_indirect = self.vae_u.decode(z_t_decode[..., self.nzx:self.nzx+self.nzu], c_type)
        z_x_t_decode_indirect = self.vae_x.decode(z_t_decode[..., self.nzx+self.nzu:], c_type)

        recon_loss_direct = self.vae_x.loss(x, z_x_decode_direct, mu_x, logvar_x) + self.vae_u.loss(u, z_u_decode_direct, mu_u, logvar_u) + self.vae_x.loss(x_t, z_x_t_decode_direct, mu_x_t, logvar_x_t)
        recon_loss_t = self.vae_t.loss(torch.cat([z_x, z_u, z_x_t], dim=-1), z_t_decode, mu_t, logvar_t)
        recon_loss_indirect = self.vae_x.loss(x, z_x_decode_indirect, mu_x, logvar_x) + self.vae_u.loss(u, z_u_decode_indirect, mu_u, logvar_u) + self.vae_x.loss(x_t, z_x_t_decode_indirect, mu_x_t, logvar_x_t)

        recon_loss = recon_loss_direct + recon_loss_t + recon_loss_indirect
        transition_direct = self.make_csvto_traj(z_x_decode_direct, z_u_decode_direct, z_x_t_decode_direct)
        transition_indirect = self.make_csvto_traj(z_x_decode_indirect, z_u_decode_indirect, z_x_t_decode_indirect)

        constraint_loss, dC = self.constraint_loss(transition_direct, transition_indirect, c_type)

        loss = self.lambda_recon * recon_loss + self.lambda_constraint * constraint_loss

        losses = {
            'recon_loss': recon_loss,
            'constraint_loss': constraint_loss
        }
        losses = loss_dict_diff(**losses)

        return loss, losses, dC
    
    def norm(self, x):
        return torch.norm(x[..., :2], dim=-1)
    
    # def violation(self, x, c_type):
    #     return (torch.relu(10 - self.norm(x)) ** 2).mean()
    
    @torch.no_grad()
    def violation(self, x, mask, key, C, dC):
        masked_x = x[mask]
        problem = self.problem_for_sampler[key]
        # z_dim = problem.dz
        # masked_x = torch.cat((masked_x, torch.zeros_like(masked_x[:, :, :z_dim])), dim=-1)
        mask_for_state = torch.ones((self.xu_dim), device=masked_x.device).bool()
        if key == (-1, -1, -1):
            mask_for_state[27:36] = False
        elif key == (-1 , 1, 1):
            mask_for_state[27:30] = False
        elif key == (1, -1, -1):
            mask_for_state[30:36] = False
        # mask_no_z = mask_for_state.clone()
        # mask_no_z[-z_dim:] = False

        problem._preprocess(masked_x[:, :, mask_for_state], projected_diffusion=True)
        g, dg, _ = problem._con_eq(masked_x[:, :, mask_for_state], compute_grads=True, compute_hess=False, projected_diffusion=True)
        h, dh, _ = problem._con_ineq(masked_x[:, :, mask_for_state], compute_grads=True, compute_hess=False, projected_diffusion=True)
        relu_mask = h < 0
        h[relu_mask] = 0
        dh[relu_mask] = 0
        C += g.sum(-1) + h.sum(-1)
        dC += dg.sum(-2) + dh.sum(-2)
        print('shapes', g.shape, dg.shape, h.shape, dh.shape)
        #TODO: Split gradients for x, u, x_t
        return C, dC
    
    def constraint_loss(self, transition_direct, transition_indirect, c_type):
        C_all, dC_all = 0, 0
        
        if self.problem_key_tensors == []:
            for key in self.problem_for_sampler:
                self.problem_key_tensors.append(torch.tensor(key, dtype=torch.long, device=c_type.device))
        for key in self.problem_key_tensors:
            C, dC = 0, 0
            mask = (c_type == key).all(-1)
            key_for_func = tuple(key.cpu().tolist())
            
            C, dC = self.violation(transition_direct, mask, key_for_func, C, dC)
            C, dC = self.violation(transition_indirect, mask, key_for_func, C, dC)

            C_all += C.mean()
            dC_all += dC.mean()
        return C_all/len(self.problem_key_tensors), dC_all/len(self.problem_key_tensors)

class TrajOptModel(torch.nn.Module):
    """
    Overall module.

    Includes encoder, decoder, dynamics, constraint embedding, constraint predictor, and controller
    """

    def __init__(self, config, Q, R, ground_truth_dynamics):
        super(TrajOptModel, self).__init__()
        self.nx = config['nx']
        self.nu = config['nu']
        self.nz = config['nz']
        self.nhidden = config['nhidden']
        self.ncz = config['ncz']
        self.num_constraints = config['num_constraints']

        self.local = config['local']

        self.vae_dynamics = VAE(self.nx, self.nz, self.nhidden)
        self.vae_constraint = VAE(self.nx, self.nz, self.nhidden)

        self.constraint_embedding = ConstraintEmbedding(self.num_constraints, self.ncz)
        self.constraint_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.nz + self.ncz if self.local else self.ncz, self.nhidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.nhidden, self.nhidden),
            torch.nn.ReLU(),
            # torch.nn.Linear(self.nhidden, self.nhidden),
            # torch.nn.ReLU(),
            # torch.nn.Linear(self.nhidden, self.nhidden),
            # torch.nn.ReLU(),
            torch.nn.Linear(self.nhidden, self.nz+self.nu+1),
            # torch.nn.Softplus()
        )

        self.constraint_predictor_constraint = torch.nn.Sequential(
            # torch.nn.Linear(self.nz + self.ncz if self.local else self.ncz, self.nhidden),
            torch.nn.Linear(self.ncz, self.nhidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.nhidden, self.nhidden),
            torch.nn.ReLU(),
            # torch.nn.Linear(self.nhidden, self.nhidden),
            # torch.nn.ReLU(),
            # torch.nn.Linear(self.nhidden, self.nhidden),
            # torch.nn.ReLU(),
            torch.nn.Linear(self.nhidden, self.nz+self.nu+1),
            # torch.nn.Softplus()
        )

        # if self.local:        
        #     self.dynamics_predictor = torch.nn.Sequential(
        #         torch.nn.Linear(self.nz, self.nhidden),
        #         torch.nn.ReLU(),
        #         torch.nn.Linear(self.nhidden, self.nhidden),
        #         torch.nn.ReLU(),
        #         torch.nn.Linear(self.nhidden, 2*self.nz+self.nz*self.nu)
        #     )
        # else:
        self.v = torch.nn.Parameter(torch.randn(self.nz, 1) * .01)
        self.r = torch.nn.Parameter(torch.randn(self.nz, 1) * .01)
        # self.register_parameter('v', self.v)
        # self.register_parameter('r', self.r)

        # self.dynamics_predictor = torch.nn.Sequential(
        #     torch.nn.Linear(1, self.nhidden),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.nhidden, self.nhidden),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.nhidden, self.nz*self.nz+self.nz*self.nu)
        # )

        # self.dynamics_predictor_constraint = torch.nn.Sequential(
        #     torch.nn.Linear(1, self.nhidden),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.nhidden, self.nhidden),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.nhidden, self.nz*self.nz+self.nz*self.nu)
        # )
        # self.A = torch.nn.Parameter(torch.randn(self.nz*2, self.nz*2).cuda().double() * .01)
        # self.B = torch.nn.Parameter(torch.randn(self.nz*2, self.nu).cuda().double() * .01)

        self.A = torch.nn.Parameter(torch.randn(self.nz, self.nz) *.01)
        self.register_parameter('A', self.A)
        self.A_constraint = torch.nn.Parameter(torch.randn(self.nz, self.nz) *.01)
        self.register_parameter('A_constraint', self.A_constraint)

        self.B = torch.nn.Parameter(torch.randn(self.nz, self.nu) *.01)
        self.register_parameter('B', self.B)
        self.B_constraint = torch.nn.Parameter(torch.randn(self.nz, self.nu) *.01)
        self.register_parameter('B_constraint', self.B_constraint)

        # Turn off gradients for A and B
        # self.A.requires_grad = False
        # self.B.requires_grad = False

        self.Q = Q
        self.R = R
        self.Q_tensor = torch.eye(self.nx, dtype=torch.float64).cuda()
        self.R_tensor = torch.eye(self.nu, dtype=torch.float64).cuda()
        self.N = config['N']

        self.controller = QPControllerLayer(self.nz, self.nu, self.Q, self.R, self.N, self.num_constraints, 2)

        self.ground_truth_dynamics = ground_truth_dynamics

        self.lambda_recon = config['lambda_recon']
        self.lambda_dyn = config['lambda_dyn']
        self.lambda_constraint = config['lambda_constraint']
        self.lambda_reg = config['lambda_reg']
        self.lambda_connect = config['lambda_connect']

        self.dyn_bool = 1
        self.constraint_bool = 1

        self.num_dyn_samples = config['num_dyn_samples']

    def get_A_B(self, z):
        # if self.local:
        #     dyn_pred = self.dynamics_predictor(z)
        #     v, r = torch.chunk(dyn_pred[..., :2*self.nz], 2, dim=1)
        #     v = v.reshape(-1, self.nz, 1)
        #     r = r.reshape(-1, self.nz, 1)
        #     A = torch.eye(self.nz, device=z.device) + torch.bmm(v, r.transpose(1, 2))
        #     B = dyn_pred[..., 2*self.nz:].reshape(-1, self.nz, self.nu)
        # else:
        v = self.v
        r = self.r
        # print(torch.round(self.v @ self.r.transpose(0, 1), decimals=2))
        # A = torch.eye(self.nz, device=z.device) + self.v @ self.r.transpose(0, 1)
        # v = 0
        # r = 0
        # dyn_pred_out = self.dynamics_predictor(torch.zeros(1, 1, device=z.device, dtype=z.dtype) + .01)
        # A = dyn_pred_out[..., :self.nz*self.nz].reshape(self.nz, self.nz)
        # B = dyn_pred_out[..., self.nz*self.nz:].reshape(self.nz, self.nu)

        # dyn_pred_out_constraint = self.dynamics_predictor_constraint(torch.zeros(1, 1, device=z.device, dtype=z.dtype) + .01)
        # A_constraint = dyn_pred_out_constraint[..., :self.nz*self.nz].reshape(self.nz, self.nz)
        # B_constraint = dyn_pred_out_constraint[..., self.nz*self.nz:].reshape(self.nz, self.nu)

        A = torch.block_diag(self.A, self.A_constraint) + torch.eye(self.nz * 2, device=z.device)
        B = torch.cat([self.B, self.B_constraint], dim=0)

        # A = torch.block_diag(self.A + torch.eye(self.nz, device=z.device), torch.zeros_like(self.A))
        # B = torch.cat([self.B, torch.zeros_like(self.B)], dim=0)

        A = A.unsqueeze(0).repeat(z.shape[0], 1, 1)
        B = B.unsqueeze(0).repeat(z.shape[0], 1, 1)
        return v, r, A, B

    def index_E_F_b(self, model, constraint_pred_input, z, mask):
        constraint_pred = model(constraint_pred_input)
        E_i = constraint_pred[..., :self.nz].reshape(z.shape[0], self.num_constraints+1, -1)
        F_i = constraint_pred[..., self.nz:self.nz+self.nu].reshape(z.shape[0], self.num_constraints+1, -1)
        b_i = constraint_pred[..., -1].reshape(z.shape[0], self.num_constraints+1)

        E_i[mask] = 0
        F_i[mask] = 0
        b_i[mask] = 0

        return E_i, F_i, b_i

    def get_E_F_b(self, z, constraint_idx=None, cz=None):
        if constraint_idx is not None and cz is not None:
            raise ValueError('Cannot provide both constraint_idx and cz')
        elif constraint_idx is not None:
            cz = self.constraint_embedding(constraint_idx)
            cz_mu = cz
            cz_logvar = 0
            # cz_mu, cz_logvar = torch.chunk(self.constraint_embedding(constraint_idx), 2, dim=-1)
            # if self.training:
            #     cz = cz_mu + torch.randn_like(cz_mu) * torch.exp(0.5 * cz_logvar)
            # else:
            #     cz = cz_mu
            mask = constraint_idx == 0
        elif cz is None:
            raise ValueError('Must provide either constraint_idx or cz')

        # if self.local:
        #     constraint_pred_input = torch.cat([z.repeat_interleave(self.num_constraints+1, 0).reshape(z.shape[0], self.num_constraints+1, -1), cz], dim=-1)
        # else:
        constraint_pred_input = cz

        # E_1, F_1, b_1 = self.index_E_F_b(self.constraint_predictor, constraint_pred_input, z, mask)
        E_2, F_2, b_2 = self.index_E_F_b(self.constraint_predictor_constraint, constraint_pred_input, z, mask)
        
        # E, F, b = self.index_E_F_b(self.constraint_predictor_constraint, constraint_pred_input, z, mask)

        # E_1 = torch.cat([E_1, torch.zeros_like(E_1)], dim=-1)
        E_2 = torch.cat([torch.zeros_like(E_2), E_2], dim=-1)
        # E = torch.cat([E_1, E_2], dim=1)
        # F = torch.cat([F_1, F_2], dim=1)
        # b = torch.cat([b_1, b_2], dim=1)

        return cz_mu, cz_logvar, E_2, F_2, b_2

    def forward(self, x, goal, u_mppi, x_t, constraint_idx=None, cz=None):

        z, mu, logvar, z_goal, goal_mu, goal_logvar, z_constraint, mu_constraint, logvar_constraint, z_goal_constraint, goal_mu_constraint, goal_logvar_constraint, z_traj, z_constraint_traj, u, s, cz_mu, cz_logvar, E, F, b, v, r, A, B = self.command(x, goal, constraint_idx, cz)

        loss, losses = self.loss(x, goal, u_mppi, x_t, z, mu, s, logvar, z_goal, goal_mu, goal_logvar, z_constraint, mu_constraint, logvar_constraint, z_goal_constraint, goal_mu_constraint, goal_logvar_constraint, cz_mu, cz_logvar, z_traj, z_constraint_traj, u, E, F, b, v, r, A, B)

        return loss, losses

    def command(self, x, goal, constraint_idx=None, cz=None):
        z, mu, logvar = self.vae_dynamics.encode(x)
        z_goal, goal_mu, goal_logvar = self.vae_dynamics.encode(goal)

        z_constraint, mu_constraint, logvar_constraint = self.vae_constraint.encode(x)
        z_goal_constraint, goal_mu_constraint, goal_logvar_constraint = self.vae_constraint.encode(goal)

        v, r, A, B = self.get_A_B(z)

        cz_mu, cz_logvar, E, F, b = self.get_E_F_b(mu_constraint, constraint_idx, cz)
        z_traj, z_constraint_traj, u, s = self.controller.solve(mu, mu_constraint, goal_mu, goal_mu_constraint, A, B, E, F, b)
        return z, mu, logvar, z_goal, goal_mu, goal_logvar, z_constraint, mu_constraint, logvar_constraint, z_goal_constraint, goal_mu_constraint, goal_logvar_constraint, z_traj, z_constraint_traj, u, s, cz_mu, cz_logvar, E, F, b, v, r, A, B
    
    def dynamics_loss(self, vae, gt_rollout_no_grad, z0, z0_constraint, u, A, B):
        z0 = torch.cat([z0, z0_constraint], dim=1)
        z_rollout = [z0.reshape(z0.shape[0], -1, self.nz * 2)]
        for i in range(u.shape[1]):
            z_rollout.append(torch.bmm(z_rollout[-1], A.transpose(1, 2)) + torch.bmm(u[:, i].unsqueeze(1), B.transpose(1, 2)))
        z_rollout = torch.concat(z_rollout, dim=1)
        z_rollout = z_rollout.reshape(z0.shape[0], -1, self.nz * 2)
        z_rollout, z_constraint_rollout = torch.chunk(z_rollout, 2, dim=-1)

        _, z_gt_rollout, _ = vae.encode(gt_rollout_no_grad)
        # z_gt_rollout = z_gt_rollout.reshape(z_traj.shape[0], -1, self.nz)
        # gt_rollout = gt_rollout.reshape(z_traj.shape[0], -1, self.nx) 
        # Check how well the encoded rollout matches the encoded predicted trajectory
        rollout_loss_enc = torch.nn.functional.mse_loss(z_rollout, z_gt_rollout)
        x_traj = vae.decode(z_rollout)
        # dyn_loss = torch.nn.functional.mse_loss(x_traj[:, 1:], self.ground_truth_dynamics(x_traj[:, :-1, :], u))
        dyn_loss = torch.nn.functional.mse_loss(x_traj, gt_rollout_no_grad)
        # Dynamics loss
        # print(x_traj.shape, x_traj[:, :-1, :].shape, u.shape)
        dyn_loss = self.lambda_dyn * self.dyn_bool * (dyn_loss + rollout_loss_enc)
        return dyn_loss, x_traj, z_constraint_rollout
    
    def loss(self, x, goal, u_mppi, x_t, z, mu, s, logvar, z_goal, goal_mu, goal_logvar, z_constraint, mu_constraint, logvar_constraint, z_goal_constraint, goal_mu_constraint, goal_logvar_constraint, cz_mu, cz_logvar, z_traj, z_constraint_traj, u, E, F, b, v, r, A, B):
        # Reconstruction loss
        x_hat = self.vae_dynamics.decode(z)
        goal_hat = self.vae_dynamics.decode(z_goal)
        # Include full x_t in the reconstruction loss
        x_constraint_hat = self.vae_constraint.decode(z_constraint)
        goal_hat_constraint = self.vae_constraint.decode(z_goal_constraint)

        full_x = torch.cat([x, goal], dim=0)
        full_x_hat = torch.cat([x_hat, goal_hat], dim=0)
        full_mu = torch.cat([mu, goal_mu], dim=0)
        full_logvar = torch.cat([logvar, goal_logvar], dim=0)
        dyn_recon_loss = self.lambda_recon * self.dyn_bool * self.vae_dynamics.loss(full_x, full_x_hat, full_mu, full_logvar)

        full_x_constraint_hat = torch.cat([x_constraint_hat, goal_hat_constraint], dim=0)
        full_mu_constraint = torch.cat([mu_constraint, goal_mu_constraint], dim=0)
        full_logvar_constraint = torch.cat([logvar_constraint, goal_logvar_constraint], dim=0)
        constraint_recon_loss = self.lambda_recon * self.vae_constraint.loss(full_x, full_x_constraint_hat, full_mu_constraint, full_logvar_constraint)

        recon_loss = dyn_recon_loss + constraint_recon_loss * self.constraint_bool
        # u_flat = u.transpose(1, 2).flatten(0, 1).detach().cpu().numpy()
        # plt.quiver(np.zeros(u_flat.shape[0]), np.zeros(u_flat.shape[0]), u_flat[:, 0], u_flat[:, 1])
        # plt.xlim(-4, 4)
        # plt.ylim(-4, 4)
        # plt.savefig('u.png')
        # plt.close()
        # try:
        # Calculate the rollout given x and u using self.ground_truth_dynamics

        # if torch.rand(1) < .2:
        # u += torch.randn_like(u)

        # # Recompute z_traj by rolling out the dynamics using self.A and self.B
        # z_rollout = [z.unsqueeze(-1)]
        # for i in range(u.shape[2]):
        #     z_rollout.append(torch.bmm(A, z_rollout[-1]) + torch.bmm(B, u[..., i].unsqueeze(-1)))
        # z_rollout = torch.concat(z_rollout, dim=1)
        # z_rollout = z_rollout.reshape(z_traj.shape[0], -1, self.nz)
        # z_traj = z_rollout.transpose(1, 2)
        with torch.no_grad():
            gt_rollout = [x]
            for i in range(u_mppi.shape[1]):
                gt_rollout.append(self.ground_truth_dynamics(gt_rollout[-1], u_mppi[:, i]))
            gt_rollout = torch.concat(gt_rollout, dim=1)
            gt_rollout = gt_rollout.reshape(z_traj.shape[0], -1, self.nx)
        # plt.scatter(x.cpu().numpy()[:, 0], x.cpu().numpy()[:, 1], c='b')
        # plt.scatter(goal.cpu().numpy()[:, 0], goal.cpu().numpy()[:, 1], c='g')
        # for i in range(gt_rollout.shape[0]):
        #     plt.plot(gt_rollout[i].cpu().numpy()[:, 0], gt_rollout[i].cpu().numpy()[:, 1], c='r')
        # plt.savefig('rollout.png')
        # plt.close()
        dyn_loss, x_traj, z_constraint_rollout = self.dynamics_loss(self.vae_dynamics, gt_rollout, z, z_constraint, u_mppi, A, B)

        # Constraint satisfaction loss
        z_constraint_traj = z_constraint_traj.transpose(1, 2)
        x_traj_constraint = self.vae_constraint.decode(z_constraint_traj)

        # z_traj = z_traj.transpose(1, 2)
        # x_traj = self.vae_dynamics.decode(z_traj)

        # MSE between x_traj and x_traj_constraint
        connect_loss = self.lambda_connect * torch.nn.functional.mse_loss(x_traj, x_traj_constraint)

        x_traj_norm = torch.norm(x_traj_constraint[..., :2], dim=2)
        violation = torch.nn.functional.relu(10 - x_traj_norm)
        constraint_loss = self.lambda_constraint * self.constraint_bool * violation.mean()# + torch.norm(s, dim=-1, p=1).mean())

        gt_rollout = [x]
        for i in range(u.shape[2]):
            gt_rollout.append(self.ground_truth_dynamics(gt_rollout[-1], u[..., i]))
        gt_rollout = torch.concat(gt_rollout, dim=1)
        gt_rollout = gt_rollout.reshape(z_traj.shape[0], -1, self.nx)
        gt_rollout_norm = torch.norm(gt_rollout[..., :2], dim=2)
        violation_gt_rollout = torch.nn.functional.relu(10 - gt_rollout_norm)
        constraint_loss += self.lambda_constraint * self.constraint_bool * violation_gt_rollout.mean()

        constraint_violation_in_plan = torch.nn.functional.relu(-s).sum(-1)
        # Penalize calculated constraint violation in plan if there is actually constraint violation
        constraint_violation_in_plan = constraint_violation_in_plan * torch.logical_and(violation[:, 1:] == 0, violation_gt_rollout[:, 1:] == 0).double()
        constraint_loss += self.lambda_constraint * self.constraint_bool * constraint_violation_in_plan.mean()

        # Cost loss
        cost_loss = 0
        # gt_rollout = [x]
        # for i in range(u.shape[1]):
        #     gt_rollout.append(self.ground_truth_dynamics(gt_rollout[-1], u[:, i]))
        #     # cost_loss += torch.diag((gt_rollout[-1] - goal) @ self.Q_tensor @ (gt_rollout[-1] - goal).T) + torch.diag(u[:, i] @ self.R_tensor @ u[:, i].T)
        # gt_rollout = torch.concat(gt_rollout, dim=1)
        # gt_rollout = gt_rollout.reshape(z_traj.shape[0], -1, self.nx)[:, 1:]
        # dist = torch.norm(gt_rollout - goal.unsqueeze(1), dim=-1) ** 2
        # dist = dist.sum(-1)
        # ctrl = torch.norm(u, dim=-1) ** 2
        # ctrl = ctrl.sum(-1)
        # cost = dist + ctrl
        # cost_loss = self.lambda_cost * (cost).mean()
        # except Exception as e:
        #     dyn_loss = 0
        #     constraint_loss = 0
        #     cost_loss = 0

        # Constraint regularization loss
        constraint_reg_loss = self.lambda_reg * (
            torch.square(torch.linalg.norm(cz_mu, dim=-1)).mean()# +
            # 1 * (-.5 * torch.mean(1 + cz_logvar - cz_mu.pow(2) - cz_logvar.exp())) 
            # torch.norm(E, dim=-1).mean() +
            # torch.norm(F, dim=-1).mean() +
            # b.mean() 
            # torch.square(torch.linalg.norm(v)).mean() +
            # torch.square(torch.linalg.norm(r)).mean() +
            # torch.square(torch.linalg.norm(B.flatten())).mean()
            )

        losses = {
            'dyn_recon_loss': dyn_recon_loss/self.lambda_recon,
            'constraint_recon_loss': constraint_recon_loss/self.lambda_recon,
            'dyn_loss': dyn_loss/self.lambda_dyn,
            'constraint_loss': constraint_loss/self.lambda_constraint,
            'constraint_reg_loss': constraint_reg_loss/self.lambda_reg,
            'connect_loss': connect_loss/self.lambda_connect,
            # 'cost_loss': cost_loss.item() if cost_loss != 0 else 0,
            's': torch.nn.functional.relu(-s).mean(),
            'u': torch.norm(u, dim=-1).mean(),
        }

        losses = loss_dict(**losses)

        return recon_loss + dyn_loss + constraint_loss + constraint_reg_loss + connect_loss, losses# + torch.nn.functional.relu(-s).mean(), losses




