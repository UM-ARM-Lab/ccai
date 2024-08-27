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

        self.time_mlp = nn.Sequential(
            nn.Linear(256, embed_dim),
            Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''

        embed = self.time_mlp(t)
        out = self.blocks[0](x)
        # print(x.shape, out.shape)
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

    @torch.compile(mode='max-autotune')
    def compiled_conditional_train(self, t, x, context):
        return self(t, x, context, dropout=True)


class TemporalUNetContext(nn.Module):
    """ does a temporal unet for a score function and also a score function for the context"""

    def __init__(
            self,
            horizon,
            transition_dim,
            cond_dim,
            dim=32,
            dim_mults=(1, 2, 4),
            attention=False,
            dropout_p=0.25,
            trajectory_embed_dim=4
    ):
        super().__init__()
        # self.register_buffer('traj_dropout_p', torch.tensor([dropout_p]))
        self.temporal_unet = TemporalUnet(horizon, transition_dim, cond_dim, dim, dim_mults, attention, dropout_p)
        self.time_embedding = SinusoidalPosEmb(32)
        self.pooling = nn.AdaptiveAvgPool1d(trajectory_embed_dim)

        traj_hidden_dim = dim_mults[-1] * dim * trajectory_embed_dim
        print(traj_hidden_dim)
        print(dim_mults[-1], dim, trajectory_embed_dim)
        self.context_net = nn.Sequential(
            nn.Linear(cond_dim + 32 + traj_hidden_dim, 128),
            Mish(),
            nn.Linear(128, 128),
            Mish(),
            nn.Linear(128, cond_dim)
        )

    # @torch.compile(mode='max-autotune')
    def forward(self, t, x, context, dropout=False):
        '''
            x : [ batch x horizon x transition ]
        '''

        B, H, d = x.shape
        t = t.reshape(-1)
        if t.shape[0] == 1:
            t = t.repeat(B)
        e_x, h = self.temporal_unet(t, x, context, dropout=dropout)
        h = self.pooling(h).reshape(B, -1)

        t = self.time_embedding(t)
        # with some probability, dropout trajectory from context diffusion
        # if dropout:
        #    mask_dist = Bernoulli(probs=1 - self.traj_dropout_p)
        #    mask = mask_dist.sample((B,))  # .to(device=context.device)
        #    h = mask * h
        h = torch.cat((context, h, t), dim=-1)

        e_c = self.context_net(h)
        return e_x, e_c


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
