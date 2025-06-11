import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import numpy as np

from torch.distributions import Bernoulli
from ccai.models.helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    MLPBlock,
    Residual,
    PreNorm,
    LinearAttention,
    pad_to_multiple,
    Mish
)


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon=None, kernel_size=3):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])
        if embed_dim > 0:
            self.time_mlp = nn.Sequential(
                nn.Linear(256, embed_dim),
                Mish(),
                nn.Linear(embed_dim, out_channels),
                Rearrange('batch t -> batch t 1'),
            )
        self.embed_dim = embed_dim

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        if self.embed_dim > 0:
            embed = self.time_mlp(t)
        out = self.blocks[0](x)
        # print(x.shape, out.shape)
        if self.embed_dim > 0:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class ResidualBlock(nn.Module):
    """
    Like ResidualTemporalBlock but with MLPs instead of convolutions
    """
    def __init__(self, inp_channels, out_channels, embed_dim):
        super().__init__()

        self.blocks = nn.ModuleList([
            MLPBlock(inp_channels, out_channels),
            MLPBlock(out_channels, out_channels),
        ])

        self.time_mlp = nn.Sequential(
            nn.Linear(256, embed_dim),
            Mish(),
            nn.Linear(embed_dim, out_channels),
        )

        self.residual_conv = nn.Linear(inp_channels, out_channels) \
            if inp_channels != out_channels else nn.Identity()
        
    def forward(self, x, t):
        embed = self.time_mlp(t)
        out = self.blocks[0](x)
        out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class TemporalUnet(nn.Module):

    def __init__(
            self,
            horizon,
            transition_dim,
            cond_dim,
            dim=32,
            dim_mults=(1, 2, 4),
            attention=False,
            context_dropout_p=0.25,
            trajectory_condition=False
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        self.time_embedding = SinusoidalPosEmb(32)
        self.trajectory_condition = trajectory_condition
        if not trajectory_condition:
            self.constraint_type_embed = nn.Sequential(
                nn.Linear(cond_dim, 32),
                Mish()
            )
        # self.constraint_type_embed = SinusoidalPosEmb(32)
        context_hidden_dim = 32 if not trajectory_condition else 0
        self.time_mlp = nn.Sequential(
            nn.Linear(32 + context_hidden_dim, 256),
            Mish()
            # nn.Linear(256, 256)
        )

        self.register_buffer('context_dropout_p', torch.tensor([context_dropout_p]))

        self.mask_dist = Bernoulli(probs=1 - context_dropout_p)

        '''
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, 128),
            nn.Mish(),
        )

        self.context_mlp = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.Mish(),
        )
        '''
        time_dim = 256

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        feature_dims = [horizon]
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            feature_dims.append(np.ceil(feature_dims[-1] / 2))
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
                # nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_block3 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_block4 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        feature_dims = feature_dims[::-1][1:]
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            # kernel_size = 4 if feature_dims[ind + 1] % feature_dims[ind] == 0 else 3
            kernel_size = 3
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in, kernel_size) if not is_last else nn.Identity()
                # nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=3),
            nn.Conv1d(dim, transition_dim, 1),
        )
        # self.context_dropout_p = context_dropout_p
        self.cond_dim = cond_dim

    def vmapped_fwd(self, t, x, context=None):
        return self(t.reshape(1), x.unsqueeze(0), context.unsqueeze(0)).squeeze(0)

    def forward(self, t, x, context=None, dropout=False):
        '''
            x : [ batch x horizon x transition ]
        '''

        # ensure t is a batched tensor
        B, H, d = x.shape
        t = t.reshape(-1)
        if t.shape[0] == 1:
            t = t.repeat(B)
        t = self.time_embedding(t)
        # x = einops.rearrange(x, 'b h t -> b t h')
        x = x.permute(0, 2, 1)
        x = pad_to_multiple(x, m=2 ** len(self.downs))

        if context is not None and not self.trajectory_condition:
            # constraint_type = context[:, -2:]
            # context = context[:, :-2]
            context = context.reshape(B, -1)
            c = self.constraint_type_embed(context)
            # c = context
            # c = torch.cat((context, ctype_embed), dim=-1)
            # need to do dropout on context embedding to train unconditional model alongside conditional
            if dropout:
                mask_dist = Bernoulli(probs=1 - self.context_dropout_p)
                mask = mask_dist.sample((B,))  # .to(device=context.device)
                c = mask * c
                t = torch.cat((t, c), dim=-1)
            else:
                t = torch.cat((t, c), dim=-1)
        elif not self.trajectory_condition:
            t = torch.cat((t, torch.zeros(B, 32, device=t.device)), dim=-1)

        t = self.time_mlp(t)
        #
        h = []
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            # x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        # x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        x = self.mid_block3(x, t)
        x = self.mid_block4(x, t)
        latent = x.clone()
        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            # x = attn(x)
            x = upsample(x)
        x = self.final_conv(x)

        # x = einops.rearrange(x, 'b t h -> b h t')
        x = x.permute(0, 2, 1)
        # get rid of padding
        x = x[:, :H]
        return x, latent

    # @torch.compile(mode='max-autotune')
    def compiled_conditional_test(self, t, x, context):
        return self(t, x, context, dropout=False)

    # @torch.compile(mode='max-autotune')
    def compiled_unconditional_test(self, t, x):
        return self(t, x, context=None, dropout=False)

    @torch.compile(backend='inductor')
    def compiled_conditional_train(self, t, x, context):
        return self(t, x, context, dropout=True)


class TemporalUnetDynamics(nn.Module):

    def __init__(
            self,
            horizon,
            transition_dim,
            cond_dim,
            dim=32,
            dim_mults=(1, 2, 4),
            attention=False,
            context_dropout_p=0.25
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        self.time_embedding = SinusoidalPosEmb(32)
        self.constraint_type_embed = nn.Sequential(
            nn.Linear(cond_dim, 32),
            Mish()
        )
        # self.constraint_type_embed = SinusoidalPosEmb(32)
        self.time_mlp = nn.Sequential(
            nn.Linear(32 + 32, 256),
            Mish()
            # nn.Linear(256, 256)
        )

        self.register_buffer('context_dropout_p', torch.tensor([context_dropout_p]))

        self.mask_dist = Bernoulli(probs=1 - context_dropout_p)

        '''
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, 128),
            nn.Mish(),
        )

        self.context_mlp = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.Mish(),
        )
        '''
        time_dim = 256
        u_dim = 21

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        feature_dims = [horizon]
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            feature_dims.append(np.ceil(feature_dims[-1] / 2))
            self.downs.append(nn.ModuleList([
                ResidualBlock(dim_in, dim_out, embed_dim=time_dim),
                ResidualBlock(dim_out, dim_out, embed_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                # Downsample1d(dim_out) if not is_last else nn.Identity()
                # nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, embed_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, embed_dim=time_dim)
        self.mid_block3 = ResidualBlock(mid_dim, mid_dim, embed_dim=time_dim)
        self.mid_block4 = ResidualBlock(mid_dim, mid_dim, embed_dim=time_dim)

        self.action_pred_mlp = nn.Sequential(
            nn.Linear(mid_dim+256, dim),
            Mish(),
            nn.Linear(dim, dim),
            Mish(),
            nn.Linear(dim, u_dim)
        )

        feature_dims = feature_dims[::-1][1:]
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            # kernel_size = 4 if feature_dims[ind + 1] % feature_dims[ind] == 0 else 3
            kernel_size = 3
            self.ups.append(nn.ModuleList([
                ResidualBlock(dim_out * 2+u_dim, dim_in, embed_dim=time_dim),
                ResidualBlock(dim_in, dim_in, embed_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                # nn.Linear(dim_in, dim_in) if not is_last else nn.Identity()
                # nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            MLPBlock(dim, dim),
            nn.Linear(dim, transition_dim),
        )

        # self.context_dropout_p = context_dropout_p
        self.cond_dim = cond_dim

    def vmapped_fwd(self, t, x, context=None):
        return self(t.reshape(1), x.unsqueeze(0), context.unsqueeze(0)).squeeze(0)


    def forward(self, t, x, context=None, dropout=False):
        '''
            x : [ batch x horizon x transition ]
        '''

        # ensure t is a batched tensor
        B, d = x.shape
        t = t.reshape(-1)
        if t.shape[0] == 1:
            t = t.repeat(B)
        t = self.time_embedding(t)
        # x = einops.rearrange(x, 'b h t -> b t h')
        # x = x.permute(0, 2, 1)

        if context is not None:
            # constraint_type = context[:, -2:]
            # context = context[:, :-2]
            context = context.reshape(B, -1)
            c = self.constraint_type_embed(context)
            # c = context
            # c = torch.cat((context, ctype_embed), dim=-1)
            # need to do dropout on context embedding to train unconditional model alongside conditional
            if dropout:
                mask_dist = Bernoulli(probs=1 - self.context_dropout_p)
                mask = mask_dist.sample((B,))  # .to(device=context.device)
                c = mask * c
                t = torch.cat((t, c), dim=-1)
            else:
                t = torch.cat((t, c), dim=-1)
        else:
            t = torch.cat((t, torch.zeros(B, 32, device=t.device)), dim=-1)
        t = self.time_mlp(t)
        #
        h = []
        for resnet, resnet2, attn in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            # x = attn(x)
            h.append(x)
            # x = downsample(x)
        x = self.mid_block1(x, t)
        # x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        x = self.mid_block3(x, t)
        x = self.mid_block4(x, t)

        u_hat = self.action_pred_mlp(torch.cat((x, t), dim=-1))

        latent = x.clone()
        for resnet, resnet2, attn in self.ups:
            x = torch.cat((x, h.pop(), u_hat), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            # x = attn(x)
            # x = upsample(x)
        x = self.final_conv(x)

        # x = einops.rearrange(x, 'b t h -> b h t')
        # x = x.permute(0, 2, 1)
        # get rid of padding
        # x = x[:, :H]
        return x, latent, u_hat

    @torch.compile(mode='max-autotune')
    def compiled_conditional_test(self, t, x, context):
        return self(t, x, context, dropout=False)

    @torch.compile(mode='max-autotune')
    def compiled_unconditional_test(self, t, x):
        return self(t, x, context=None, dropout=False)

    @torch.compile(mode='max-autotune')
    def compiled_conditional_train(self, t, x, context):
        return self(t, x, context, dropout=True)

class TemporalUnetStateAction(nn.Module):

    def __init__(
            self,
            horizon,
            transition_dim,
            cond_dim,
            dim=32,
            dim_mults=(1, 2, 4),
            attention=False,
            context_dropout_p=0.25,
            problem_dict=None
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        self.time_embedding = SinusoidalPosEmb(32)
        self.constraint_type_embed = nn.Sequential(
            nn.Linear(cond_dim, 32),
            Mish()
        )
        # self.constraint_type_embed = SinusoidalPosEmb(32)
        self.time_mlp = nn.Sequential(
            nn.Linear(32 + 32, 256),
            Mish()
            # nn.Linear(256, 256)
        )

        self.register_buffer('context_dropout_p', torch.tensor([context_dropout_p]))

        self.mask_dist = Bernoulli(probs=1 - context_dropout_p)

        '''
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, 128),
            nn.Mish(),
        )

        self.context_mlp = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.Mish(),
        )
        '''
        time_dim = 256

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        feature_dims = [horizon]
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            feature_dims.append(np.ceil(feature_dims[-1] / 2))
            self.downs.append(nn.ModuleList([
                ResidualBlock(dim_in, dim_out, embed_dim=time_dim),
                ResidualBlock(dim_out, dim_out, embed_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                # Downsample1d(dim_out) if not is_last else nn.Identity()
                # nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, embed_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, embed_dim=time_dim)
        self.mid_block3 = ResidualBlock(mid_dim, mid_dim, embed_dim=time_dim)
        self.mid_block4 = ResidualBlock(mid_dim, mid_dim, embed_dim=time_dim)

        # self.action_pred_mlp = nn.Sequential(
        #     nn.Linear(mid_dim+256, 128),
        #     Mish(),
        #     nn.Linear(128, 128),
        #     Mish(),
        #     nn.Linear(128, u_dim)
        # )

        feature_dims = feature_dims[::-1][1:]
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            # kernel_size = 4 if feature_dims[ind + 1] % feature_dims[ind] == 0 else 3
            kernel_size = 3
            self.ups.append(nn.ModuleList([
                ResidualBlock(dim_out * 2, dim_in, embed_dim=time_dim),
                ResidualBlock(dim_in, dim_in, embed_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                # nn.Linear(dim_in, dim_in) if not is_last else nn.Identity()
                # nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            MLPBlock(dim, dim),
            nn.Linear(dim, transition_dim),
        )

        # self.context_dropout_p = context_dropout_p
        self.cond_dim = cond_dim

        self.problem_dict = problem_dict
        self.transition_dim = transition_dim

    def vmapped_fwd(self, t, x, context=None):
        return self(t.reshape(1), x.unsqueeze(0), context.unsqueeze(0)).squeeze(0)


    def forward(self, t, x, context=None, dropout=False):
        '''
            x : [ batch x horizon x transition ]
        '''

        # ensure t is a batched tensor
        B, d = x.shape
        t = t.reshape(-1)
        if t.shape[0] == 1:
            t = t.repeat(B)
        t = self.time_embedding(t)
        # x = einops.rearrange(x, 'b h t -> b t h')
        # x = x.permute(0, 2, 1)
        x_orig = x.clone()

        if context is not None:
            # constraint_type = context[:, -2:]
            # context = context[:, :-2]
            context = context.reshape(B, -1)
            c = self.constraint_type_embed(context)
            # c = context
            # c = torch.cat((context, ctype_embed), dim=-1)
            # need to do dropout on context embedding to train unconditional model alongside conditional
            if dropout:
                mask_dist = Bernoulli(probs=1 - self.context_dropout_p)
                mask = mask_dist.sample((B,))  # .to(device=context.device)
                c = mask * c
                t = torch.cat((t, c), dim=-1)
            else:
                t = torch.cat((t, c), dim=-1)
        else:
            t = torch.cat((t, torch.zeros(B, 32, device=t.device)), dim=-1)
        t = self.time_mlp(t)
        #
        h = []
        for resnet, resnet2, attn in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            # x = attn(x)
            h.append(x)
            # x = downsample(x)
        x = self.mid_block1(x, t)
        # x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        x = self.mid_block3(x, t)
        x = self.mid_block4(x, t)

        latent = x.clone()
        for resnet, resnet2, attn in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            # x = attn(x)
            # x = upsample(x)
        x = self.final_conv(x)

        return x_orig, x
        # x = einops.rearrange(x, 'b t h -> b h t')
        # x = x.permute(0, 2, 1)
        # get rid of padding
        # x = x[:, :H]


    def c_state_mask(self, context, x):
        c_state = context
        z_dim = self.problem_dict[c_state].dz
        mask = torch.ones((self.transition_dim + z_dim), device=x.device).bool()
        if c_state == (-1, -1, -1):
            mask[27:36] = False
        elif c_state == (-1 , 1, 1):
            mask[27:30] = False
        elif c_state == (1, -1, -1):
            mask[30:36] = False
        # Concat False to mask to match the size of x
        mask = torch.cat((mask, torch.zeros(self.transition_dim + z_dim -x.shape[-1], device=x.device).bool()))
        mask_no_z = mask.clone()
        if z_dim > 0:
            mask_no_z[-z_dim:] = False
        return c_state, mask, mask_no_z

    def project(self, x_orig, x, context):
        if self.problem_dict is not None:
            for problem_idx in self.problem_dict:
                problem = self.problem_dict[problem_idx]
                p_idx = sum(problem_idx)
                c_state, mask, mask_no_z = self.c_state_mask(problem_idx, x)
                num_dim = mask.long().sum()

                x_this_problem = x_orig[context.sum(-1) == p_idx][:, None] * self.x_std + self.x_mean
                grad_x_this_problem = x[context.sum(-1) == p_idx][:, None] * self.x_std
                b = x_this_problem.shape[0]


                
                if x_this_problem.shape[0] == 0:
                    continue

                problem._preprocess(x_this_problem[:, :, mask_no_z], projected_diffusion=True)
                z_dim = problem.dz
                C, dC, _ = problem.combined_constraints(x_this_problem[:, :, mask], compute_hess=False, projected_diffusion=True)

                dtype = torch.float64
                C = C.to(dtype=dtype).squeeze()
                dC = dC.to(dtype=dtype).squeeze()

                update_this_c = torch.cat((grad_x_this_problem[:, :, mask], torch.zeros((b, 1, z_dim), device=x.device, dtype=dtype)), dim=-1)
                eye = torch.eye(C.shape[1]).repeat(b, 1, 1).to(device=C.device, dtype=dtype)
                
                try:
                    dCdCT_inv = torch.linalg.solve(dC @ dC.permute(0, 2, 1) +
                                                1e-6 * eye
                                                , eye)
                except Exception as e:
                    dCdCT_inv = torch.linalg.pinv(dC @ dC.permute(0, 2, 1) * 1e-6 * eye)
                projection = dC.permute(0, 2, 1) @ dCdCT_inv @ dC
                eye2 = torch.eye(x_this_problem.shape[-2] * num_dim, device=x.device, dtype=dtype).unsqueeze(0)
                update_this_c = (eye2 - projection) @ update_this_c.reshape(b, -1, 1)
                update_this_c = update_this_c.squeeze(-1)

                # Update to decrease constraint violation
                xi_C = dCdCT_inv @ C.unsqueeze(-1)
                xi_C = (dC.permute(0, 2, 1) @ xi_C).squeeze(-1)

                update_this_c -= .1 * xi_C

                if problem.dz > 0:
                    update_this_c = update_this_c[:, : :-problem.dz]

                update_this_c = (update_this_c) / self.x_std[mask]

                update_this_c = update_this_c.to(dtype=x.dtype)

                c_ = (context.sum(-1) == p_idx).reshape(-1, 1).repeat(1, x.shape[-1])
                m = mask.reshape(1, -1).repeat(x.shape[0], 1)

                full_mask = c_ & m
                full_mask_inv_m = c_ & ~m
                x[full_mask] = update_this_c.flatten()
                x[full_mask_inv_m] = 0

        return x, x

    @torch.compile(mode='max-autotune')
    def compiled_conditional_test_fwd(self, t, x, context):
        x_orig, x = self(t, x, context, dropout=False)
        return x_orig, x

    def compiled_conditional_test(self, t, x, context):
        x_orig, x = self.compiled_conditional_test_fwd(t, x, context)
        x, x = self.project(x_orig, x, context)
        return x, x

    @torch.compile(mode='max-autotune')
    def compiled_unconditional_test(self, t, x):
        _, x = self(t, x, context=None, dropout=False)
        return x, x
    
    @torch.compile(mode='max-autotune')
    def compiled_conditional_train_fwd(self, t, x, context):
        x_orig, x = self(t, x, context, dropout=True)
        return x_orig, x
    
    def compiled_conditional_train(self, t, x, context):
        x_orig, x = self.compiled_conditional_train_fwd(t, x, context)
        x, x = self.project(x_orig, x, context)
        return x, x

class StateActionMLP(nn.Module):

    def __init__(
            self,
            horizon,
            transition_dim,
            cond_dim,
            dim=32,
            dim_mults=(1, 2, 4),
            attention=False,
            context_dropout_p=0.25
    ):
        super().__init__()

        self.time_embedding = SinusoidalPosEmb(32)
        self.constraint_type_embed = nn.Sequential(
            nn.Linear(cond_dim, 32),
            Mish()
        )
        # self.constraint_type_embed = SinusoidalPosEmb(32)
        self.time_mlp = nn.Sequential(
            nn.Linear(32 + 32, 256),
            Mish()
            # nn.Linear(256, 256)
        )

        self.residual_block = ResidualBlock(transition_dim, transition_dim, 256)
        

        self.register_buffer('context_dropout_p', torch.tensor([context_dropout_p]))

        self.mask_dist = Bernoulli(probs=1 - context_dropout_p)

        time_dim = 256


        # self.context_dropout_p = context_dropout_p
        self.cond_dim = cond_dim

    def vmapped_fwd(self, t, x, context=None):
        return self(t.reshape(1), x.unsqueeze(0), context.unsqueeze(0)).squeeze(0)


    def forward(self, t, x, context=None, dropout=False):
        '''
            x : [ batch x horizon x transition ]
        '''

        # ensure t is a batched tensor
        B, d = x.shape
        t = t.reshape(-1)
        if t.shape[0] == 1:
            t = t.repeat(B)
        t = self.time_embedding(t)
        # x = einops.rearrange(x, 'b h t -> b t h')
        # x = x.permute(0, 2, 1)

        if context is not None:
            # constraint_type = context[:, -2:]
            # context = context[:, :-2]
            context = context.reshape(B, -1)
            c = self.constraint_type_embed(context)
            # c = context
            # c = torch.cat((context, ctype_embed), dim=-1)
            # need to do dropout on context embedding to train unconditional model alongside conditional
            if dropout:
                mask_dist = Bernoulli(probs=1 - self.context_dropout_p)
                mask = mask_dist.sample((B,))  # .to(device=context.device)
                c = mask * c
                t = torch.cat((t, c), dim=-1)
            else:
                t = torch.cat((t, c), dim=-1)
        else:
            t = torch.cat((t, torch.zeros(B, 32, device=t.device)), dim=-1)
        t = self.time_mlp(t)

        x = self.residual_block(x, t)
        return x, x

    @torch.compile(mode='max-autotune')
    def compiled_conditional_test(self, t, x, context):
        return self(t, x, context, dropout=False)

    @torch.compile(mode='max-autotune')
    def compiled_unconditional_test(self, t, x):
        return self(t, x, context=None, dropout=False)

    @torch.compile(mode='max-autotune')
    def compiled_conditional_train(self, t, x, context):
        return self(t, x, context, dropout=True)


class TemporalUNetContext(nn.Module):
    """ does a temporal unet for a score function and also a score function for the context"""

    def __init__(
            self,
            horizon,
            state_dim,
            action_dim,
            cond_dim,
            dim=32,
            dim_mults=(1, 2, 4),
            attention=False,
            dropout_p=0.25,
            trajectory_embed_dim=4,
            trajectory_condition=True,
            true_s0=False
    ):
        transition_dim = state_dim + action_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.transition_dim = transition_dim
        super().__init__()
        self.trajectory_condition = trajectory_condition
        self.register_buffer('dropout_prob', torch.tensor([1.0 - dropout_p]))
        self.dropout_p = dropout_p
        self.mask_dist = Bernoulli(probs=1 - self.dropout_p)
        # self.register_buffer('traj_dropout_p', torch.tensor([dropout_p]))
        self.temporal_unet = TemporalUnet(horizon, transition_dim, cond_dim, dim, dim_mults, attention, dropout_p, trajectory_condition)
        self.time_embedding = SinusoidalPosEmb(32)
        self.pooling = nn.AdaptiveAvgPool1d(trajectory_embed_dim)
        init_state_embed_dim = 512
        self.true_s0 = true_s0

        if self.trajectory_condition:
            if true_s0:
                traj_hidden_dim = dim_mults[-1] * dim * trajectory_embed_dim + init_state_embed_dim
            else:
                traj_hidden_dim = dim_mults[-1] * dim * trajectory_embed_dim
        else:
            if true_s0:
                traj_hidden_dim = init_state_embed_dim
            else:
                traj_hidden_dim = init_state_embed_dim
        # if true_s0:
        #     if self.trajectory_condition:
        #         traj_hidden_dim = dim_mults[-1] * dim * trajectory_embed_dim + init_state_embed_dim
        #     else:
        #         traj_hidden_dim = init_state_embed_dim
        # else:
        #     if self.trajectory_condition:
        #         # traj_hidden_dim += dim_mults[-1] * dim * trajectory_embed_dim
        #         traj_hidden_dim = dim_mults[-1] * dim * trajectory_embed_dim
        #     else:
        #         traj_hidden_dim = init_state_embed_dim

        print(traj_hidden_dim)
        print(dim_mults[-1], dim, trajectory_embed_dim)
        self.context_net = nn.Sequential(
            nn.Linear(cond_dim + 32 + traj_hidden_dim, 128),
            Mish(),
            nn.Linear(128, 128),
            Mish(),
            nn.Linear(128, cond_dim)
        )

        if not self.trajectory_condition or self.true_s0:
            self.initial_state_context_net = nn.Sequential(
                nn.Linear(state_dim, init_state_embed_dim),
                Mish(),
                nn.Linear(init_state_embed_dim, init_state_embed_dim),
                Mish(),
                nn.Linear(init_state_embed_dim, init_state_embed_dim),
                # Mish(),
                # nn.Linear(init_state_embed_dim, init_state_embed_dim),
            )

    def preprocess_t(self, t, x, context):
        B, H, d = x.shape
        t = t.reshape(-1)
        if t.shape[0] == 1:
            t = t.repeat(B)
        return B, H, d, t
    # @torch.compile(mode='max-autotune')
    def forward(self, t, x, context, dropout=False, initial_state=None):
        '''
            x : [ batch x horizon x transition ]
        '''

        B, H, d, t = self.preprocess_t(t, x, context)

        if self.true_s0 or not self.trajectory_condition:
            initial_state_embed = self.initial_state_context_net(initial_state)

        if self.trajectory_condition:
            e_x, h_pre_pool = self.temporal_unet(t, x)
            h = self.pooling(h_pre_pool).reshape(B, -1)
            if self.true_s0:
                h = torch.cat((h, initial_state_embed), dim=-1)
        else:
            e_x, _ = self.temporal_unet(t, x, context, dropout=dropout)
            if self.true_s0:
                h = self.initial_state_context_net(initial_state) # True initial state for conditioning
            else:
                h = self.initial_state_context_net(x[:, 0, :self.state_dim]) # Noised initial state for conditioning

        t = self.time_embedding(t)
        # with some probability, dropout trajectory from context diffusion
        if self.trajectory_condition and dropout:
            mask = torch.bernoulli(self.dropout_prob.expand(B).to(device=h.device))
            mask = mask.unsqueeze(-1)
            h = mask * h
        h = torch.cat((context, h, t), dim=-1)

        e_c = self.context_net(h)
        return e_x, e_c
    
    def model_pred_for_sample(self, t, x, context):
        #Alt 2
        B, H, d, t = self.preprocess_t(t, x, context)
        if self.trajectory_condition:

            e_x, h = self.temporal_unet(t, x, dropout=False)
            h = self.pooling(h).reshape(B, -1)
            if self.true_s0:
                initial_state_embed = self.initial_state_context_net(x[:, 0, :self.state_dim])
                h = torch.cat((h, initial_state_embed), dim=-1)
            t = self.time_embedding(t)

            h_for_e_c = torch.cat((context, h, t), dim=-1)
            uncond_h_for_e_c = torch.cat((context, torch.zeros_like(h), t), dim=-1)
            unconditional_e_c = self.context_net(uncond_h_for_e_c)
            conditional_e_c = self.context_net(h_for_e_c)

            w_total = 1.2

            e_c = unconditional_e_c + w_total * (conditional_e_c - unconditional_e_c)
            return e_x, e_c
        #Alt 1
        else:
            t_embed = self.time_embedding(t)
            initial_state = x[:, 0, :self.state_dim]
            inital_state_embed = self.initial_state_context_net(initial_state)
            h_for_e_c = torch.cat((context, inital_state_embed, t_embed), dim=-1)
            uncond_h_for_e_c = torch.cat((context, torch.zeros_like(inital_state_embed), t_embed), dim=-1)
            unconditional_e_c = self.context_net(uncond_h_for_e_c)
            conditional_e_c = self.context_net(h_for_e_c)

            w_total = 1.2
            e_c = unconditional_e_c + w_total * (conditional_e_c - unconditional_e_c)

            unconditional_e_x, _ = self.temporal_unet(t, x, context=None, dropout=False)
            conditional_e_x, _ = self.temporal_unet(t, x, context=context, dropout=False)

            e_x = unconditional_e_x + w_total * (conditional_e_x - unconditional_e_x)
            return e_x, e_c

    # @torch.compile(mode='max-autotune')
    def compiled_conditional_test(self, t, x, context):
        return self(t, x, context, dropout=False)

    # @torch.compile(mode='max-autotune')
    def compiled_unconditional_test(self, t, x, initial_state=None):
        return self(t, x, context=None, dropout=False, initial_state=initial_state)

    @torch.compile(mode='max-autotune')
    def compiled_conditional_train(self, t, x, context, initial_state=None):
        return self(t, x, context, dropout=True, initial_state=initial_state)

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_size=64):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool1d(input_dim)
        self.fc1 = nn.Linear(512, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.act = nn.ReLU()
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        B = x.shape[0]
        x = self.pooling(x).reshape(B, -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return self.output_act(x)


class UnetClassifier(nn.Module):

    def __init__(
            self,
            horizon,
            transition_dim,
            cond_dim,
            dim=32,
            dim_mults=(1, 2, 4),
            attention=False,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        self.constraint_type_embed = nn.Sequential(
            nn.Linear(cond_dim, 32),
            Mish()
        )
        # self.constraint_type_embed = SinusoidalPosEmb(32)
        self.time_mlp = nn.Sequential(
            nn.Linear(32, 256),
            Mish()
        )

        time_dim = 256

        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)
        feature_dims = [horizon]
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            feature_dims.append(np.ceil(feature_dims[-1] / 2))
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
                # nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_block3 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_block4 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(mid_dim, 1)
        self.output_act_fn = nn.Sigmoid()

    def vmapped_fwd(self, t, x, context=None):
        return self(t.reshape(1), x.unsqueeze(0), context.unsqueeze(0)).squeeze(0)

    # @torch.compile(mode='max-autotune')
    def forward(self, t, x, context=None, dropout=False):
        '''
            x : [ batch x horizon x transition ]
        '''

        # ensure t is a batched tensor
        B, H, d = x.shape
        # x = einops.rearrange(x, 'b h t -> b t h')
        x = x.permute(0, 2, 1)
        x = pad_to_multiple(x, m=2 ** len(self.downs))

        if context is not None:
            # constraint_type = context[:, -2:]
            # context = context[:, :-2]
            context = context.reshape(B, -1)
            c = self.constraint_type_embed(context)
            # c = context
            # c = torch.cat((context, ctype_embed), dim=-1)
            # need to do dropout on context embedding to train unconditional model alongside conditional
            if dropout:
                mask_dist = Bernoulli(probs=1 - self.context_dropout_p)
                mask = mask_dist.sample((B,))  # .to(device=context.device)
                c = mask * c

        t = self.time_mlp(c)
        h = []
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            # x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        # x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        x = self.mid_block3(x, t)
        x = self.mid_block4(x, t)
        x = self.pooling(x).reshape(B, -1)
        
        return self.output_act_fn(self.output_layer(x))