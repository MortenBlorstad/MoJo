import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as torchd
from typing import Dict, List
import math
from world_model import utils
import numpy as np


# """
# h_t: hidden state at time t
# z_t: latent state at time t
# o_t: observation at time t
# a_t: action at time t
# q(): encoder - Representation Model z_t = q(z_t | h_t, o_t)
# f(): Recurrent Model h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
# t(): transition Predictor  \hat{z}_t ~ t(hat{z}_t  | h_t)
# p(): decoder - Observation Model o_t = p(o_t | h_t, z_t)
# """

class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=False, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(inp_size + size, 3 * size, bias=False)
        if norm:
            self._norm = nn.LayerNorm(3 * size, eps=1e-03)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):

        state = state[0]  # Keras wraps the state in a list.

        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]

class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm="LayerNorm",
        kernel_size=4,
        minres=4,
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        h, w, input_ch = input_shape
        layers = []
        for i in range(int(np.log2(h) - np.log2(minres))):
            if i == 0:
                in_dim = input_ch
            else:
                in_dim = 2 ** (i - 1) * depth
            out_dim = 2**i * depth
            layers.append(
                Conv2dSame(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            layers.append(ChLayerNorm(out_dim))
            layers.append(act())
            h, w = h // 2, w // 2

        self.outdim = out_dim * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(utils.weight_init)

    def forward(self, obs):
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class CubeMaskGenerator:
    def __init__(self, input_size, image_size, clip_size, block_size, mask_ratio):
        assert mask_ratio <= 1.0

        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)
        self.image_size = image_size
        self.upsampler = nn.Upsample((image_size, image_size))

        self.block_size = block_size
        self.num_blocks = clip_size // block_size

    
    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        for i in range(self.num_blocks):
            np.random.shuffle(mask)
            cur_mask = torch.from_numpy(mask).reshape(self.height, self.width)
            cur_mask = self.upsampler(cur_mask[None, None].float()) # (1, 1, h, w)
            cur_mask = cur_mask.expand(self.block_size, *cur_mask.size()[1:])
            cube_mask = torch.cat([cube_mask, cur_mask]) if i > 0 else cur_mask
        return cube_mask


class ChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ChLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(ch, eps=eps)  # Normalize over `channels` only

    def forward(self, x):
        # Permute to (batch, height, width, channels) for LayerNorm
        x = x.permute(0, 2, 3, 1)  
        x = self.norm(x)  # Apply LayerNorm
        x = x.permute(0, 3, 1, 2)  # Revert to (batch, channels, height, width)
        return x


class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


# region Representation model
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_shape = config['image_size']
        input_ch = input_shape[0]
        h = input_shape[1]
        w = input_shape[2]
        step_embedding_dim = config['step_embedding_dim']
        scalar_dim = config['scalar_dim']
        depth = 96
        mlp_units = 64
        mlp_layers = 2
        self.layers = []
        for i in range(int(np.log2(24) - np.log2(4))):
            if i == 0:
                in_dim = input_ch
            else:
                in_dim = 2 ** (i - 1) * depth
            out_dim = 2**i * depth
    
            self.layers.append(Conv2dSame(in_dim, out_dim, kernel_size=4, stride=2))
            self.layers.append(ChLayerNorm(out_dim))
            self.layers.append(nn.SiLU())
            h, w = h // 2, w // 2
            


        self.conv = nn.Sequential(*self.layers) 
        self.cnn_outdim = out_dim * h * w
        self.cnn_outshape = (self.cnn_outdim//h**2, h, w)
        print("cnn_outdim", self.cnn_outdim)

        self.layers =   [
                            nn.Linear(scalar_dim + step_embedding_dim, mlp_units),
                            nn.LayerNorm(mlp_units),
                            nn.SiLU()
                        ]  
        
        for i in range(mlp_layers-1):
            self.layers.append(nn.Linear(mlp_units, mlp_units))
            self.layers.append(nn.LayerNorm(mlp_units))
            self.layers.append(nn.SiLU())
        self.fc = nn.Sequential(*self.layers)

 
        self.fc_embedding = nn.Sequential(
            nn.Linear(mlp_units + self.cnn_outdim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, config['latent_dim']),
            nn.LayerNorm(config['latent_dim']),
            nn.SiLU()
        )
        
        
        

    def forward(self, obs:Dict[str, torch.Tensor]):
        image = obs["image"]
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = image.reshape((-1,) + tuple(image.shape[-3:]))
        conv_out = self.conv(x)
        
        # (batch * time, ...) -> (batch * time, -1)
        conv_out = conv_out.reshape([conv_out.shape[0], np.prod(conv_out.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        conv_out = conv_out.reshape(list(image.shape[:-3]) + [conv_out.shape[-1]])

        step_embedding = obs["step_embedding"]
        scalars = obs["scalars"]
        mlp_input = torch.cat([step_embedding, scalars], dim=-1)
        fc_out = self.fc(mlp_input)
        
        outputs = torch.cat([conv_out, fc_out], -1)

        outputs = self.fc_embedding(outputs)

        return outputs

# endregion


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="SiLU",
        norm="LayerNorm",
        dist="normal",
        std=1.0,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        self._layers = layers
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._dist = dist
        self._std = std
        self._symlog_inputs = symlog_inputs
        self._device = device

        layers = []
        for index in range(self._layers):
            layers.append(nn.Linear(inp_dim, units, bias=False))
            layers.append(norm(units, eps=1e-03))
            layers.append(act())
            if index == 0:
                inp_dim = units
        self.layers = nn.Sequential(*layers)
        self.layers.apply(utils.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(utils.uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(utils.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(utils.uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(utils.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = utils.symlog(x)
        out = self.layers(x)
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if dist == "normal":
            return utils.ContDist(
                torchd.independent.Independent(
                    torchd.normal.Normal(mean, std), len(shape)
                )
            )
        if dist == "huber":
            return utils.ContDist(
                torchd.independent.Independent(
                    utils.UnnormalizedHuber(mean, std, 1.0), len(shape)
                )
            )
        if dist == "binary":
            return utils.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        if dist == "symlog_disc":
            return utils.DiscDist(logits=mean, device=self._device)
        if dist == "symlog_mse":
            return utils.SymlogDist(mean)
        raise NotImplementedError(dist)


class ActionHead(nn.Module):
    def __init__(
        self,
        inp_dim,
        size,
        layers,
        units,
        act=nn.ELU,
        norm=nn.LayerNorm,
        dist="trunc_normal",
        init_std=0.0,
        min_std=0.1,
        max_std=1.0,
        temp=0.1,
        outscale=1.0,
        unimix_ratio=0.01,
    ):
        super(ActionHead, self).__init__()
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._min_std = min_std
        self._max_std = max_std
        self._init_std = init_std
        self._unimix_ratio = unimix_ratio
        self._temp = temp() if callable(temp) else temp

        pre_layers = []
        for index in range(self._layers):
            pre_layers.append(nn.Linear(inp_dim, self._units, bias=False))
            pre_layers.append(norm(self._units, eps=1e-03))
            pre_layers.append(act())
            if index == 0:
                inp_dim = self._units
        self._pre_layers = nn.Sequential(*pre_layers)
        self._pre_layers.apply(utils.weight_init)

        if self._dist in ["tanh_normal", "tanh_normal_5", "normal", "trunc_normal"]:
            self._dist_layer = nn.Linear(self._units, 2 * self._size)
            self._dist_layer.apply(utils.uniform_weight_init(outscale))

        elif self._dist in ["normal_1", "onehot", "onehot_gumbel"]:
            self._dist_layer = nn.Linear(self._units, self._size)
            self._dist_layer.apply(utils.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        x = self._pre_layers(x)
        if self._dist == "tanh_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = torch.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, utils.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = utils.SampleDist(dist)
        elif self._dist == "tanh_normal_5":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, utils.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = utils.SampleDist(dist)
        elif self._dist == "normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = utils.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "normal_1":
            mean = self._dist_layer(x)
            dist = torchd.normal.Normal(mean, 1)
            dist = utils.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "trunc_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = utils.SafeTruncatedNormal(mean, std, -1, 1)
            dist = utils.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "onehot":
            x = self._dist_layer(x)
            dist = utils.OneHotDist(x, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            x = self._dist_layer(x)
            temp = self._temp
            dist = utils.ContDist(torchd.gumbel.Gumbel(x, 1 / temp))
        else:
            raise NotImplementedError(self._dist)
        return dist


# region Observation Model
class Decoder(nn.Module):
    """
    Decoder module that reconstructs the observation from the latent state.
    Observation Model: o_t = p(o_t | h_t, z_t)
    """
    def __init__(self, config, cnn_outshape):
        super().__init__()
        latent_dim = config['latent_dim']
        step_embedding_dim = config['step_embedding_dim']
        self.scalar_dim = config['scalar_dim']
        depth = 96
        mlp_units = 1024
        mlp_layers = 5
        original_ch, original_h, original_w = config['image_size']
        self.cnn_outshape = cnn_outshape
        self.cnn_outdim = cnn_outshape[0] * cnn_outshape[1] * cnn_outshape[2]

        # Fully connected layer to expand latent space
        self.fc_expand = nn.Sequential(
            nn.Linear(latent_dim, 4096),
            nn.LayerNorm(4096),
            nn.SiLU(),
            nn.Linear(4096, mlp_units + self.cnn_outdim),
            nn.LayerNorm(mlp_units + self.cnn_outdim),
            nn.SiLU()
        )
        # CNN Decoder to reconstruct image
        layers = []

    
        input_ch, h, w = cnn_outshape

        in_dim = input_ch
        minres = h
        layer_num = int(np.log2(24) - np.log2(4))
        act = True
        bias = False
        norm = True
        for i in range(layer_num):
            out_dim = self.cnn_outdim // (minres**2) // (2 ** (i + 1))
            if i == layer_num - 1:
                out_dim = original_ch
                act = False
                bias = True
                norm = False
            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=4, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=4, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size=4,
                    stride=2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )

            if norm:
                layers.append(ChLayerNorm(out_dim))
            if act:
                layers.append(nn.SiLU())

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            h, w = h * 2, w * 2

        self.deconv = nn.Sequential(*layers)

        fc_mlp = []
        
        for _ in range(mlp_layers - 1):
            fc_mlp.append(nn.Linear(mlp_units, mlp_units))
            fc_mlp.append(nn.LayerNorm(mlp_units))
            fc_mlp.append(nn.SiLU())
        
        fc_mlp = [nn.Linear(mlp_units, self.scalar_dim + step_embedding_dim),
                            nn.LayerNorm(self.scalar_dim + step_embedding_dim),
                        ]
    
        self.fc_mlp = nn.Sequential(*fc_mlp)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = int(math.ceil(val / 2))
        outpad = int(pad * 2 - val)
        return pad, outpad
       
    def forward(self, latent):
        c,h,w = self.cnn_outshape
        x = self.fc_expand(latent)

    
        conv_x = x[:,:, :self.cnn_outdim].view(
            [-1,c, h, w]
        )
  
        conv_out = self.deconv(conv_x)
       

        mlp_x = x[:,:, self.cnn_outdim:]
  
        mlp_out = self.fc_mlp(mlp_x)
     
        scalar_out = mlp_out[:, :,:self.scalar_dim]
        step_embedding_out = mlp_out[:,:, self.scalar_dim:]
   
   

        return {"image": conv_out, "scalars": scalar_out, "step_embedding": step_embedding_out}

# endregion

# region Recurrent model

def swap_axes(x):
    """Rearrange (batch, time, ...) to (time, batch, ...)."""
    return x.permute(1, 0, *range(2, x.ndim))


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample

class ContDist:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        return self._dist.mean

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        return self._dist.log_prob(x)



class RSSM(nn.Module):
    """
    Recurrent State-Space Model (RSSM) that models transitions in the latent space.
    It maintains a hidden state using a GRU cell and predicts the next latent state.
    """
    def __init__(self, config,device):
        super(RSSM, self).__init__()
        self.stoch = config['stoch']  # Stochastic latent state size
        self.latent_dim = config['latent_dim']  # z_t (deterministic latent state size)
        self.hidden_dim = config['hidden_dim']  # h_t (hidden state)
        num_actions = config["num_actions"]
        num_units = config["num_units"]
        self.discrete_actions = config["discrete_actions"]
        self.batch_size = config["batch_size"]
        self.device = device
        self._unimix_ratio = config["unimix_ratio"]
        self._shared = config["shared"]
        self._temp_post = config["temp_post"]
        self._initial = config["initial"]
        self.cont_stoch_size = config["cont_stoch_size"]
        if self.discrete_actions:
            inp_dim = self.stoch * self.discrete_actions + num_actions*num_units
        else:
            inp_dim = self.stoch + num_actions*num_units
            
        
        self.fc_pre_gru = nn.Sequential(
            nn.Linear(inp_dim, self.latent_dim, bias=False), 
            nn.LayerNorm(self.latent_dim, eps=1e-03),
            nn.SiLU()
        )
        
        # Recurrent Model h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        self.gru = GRUCell(self.latent_dim, self.hidden_dim, norm=True)

        # Transition Predictor  \hat{z}_t ~ t(hat{z}_t  | h_t)
        self.fc_transition = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_dim, bias=False), 
            nn.LayerNorm(self.latent_dim, eps=1e-03),
            nn.SiLU()
        )
        # Posterior Refinement
        self.fc_posterior = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim, self.latent_dim, bias=False),
            nn.LayerNorm(self.latent_dim, eps=1e-03),
            nn.SiLU()
        )

        if self.discrete_actions:
            self._ims_stat_layer = nn.Linear(self.latent_dim, self.stoch * self.discrete_actions)
            self._obs_stat_layer = nn.Linear(self.latent_dim, self.stoch * self.discrete_actions)
       
        else:
            self._ims_stat_layer = nn.Linear(self.latent_dim, 2 * self.stoch)
            self._obs_stat_layer = nn.Linear(self.latent_dim, 2 * self.stoch)
        
        self.cont_feat_layers = []
        self.cont_feat_layers.append(nn.Linear(self.stoch * self.discrete_actions + self.hidden_dim, self.cont_stoch_size, bias=False))

        self._cont_feat_layers = nn.Sequential(*self.cont_feat_layers)
        self._cont_feat_layers.apply(utils.weight_init)
        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self.hidden_dim)).to(self.device),
                requires_grad=True,
            )

    def get_mlp_feat(self, feat):
        return self._cont_feat_layers(feat)

    def get_stoch(self, hidden):
        x = self.fc_transition(hidden)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()


    def initial(self, batch_size) -> Dict[str, torch.Tensor]:
        """ Initializes the latent state for a batch. """
        hidden = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        if self.discrete_actions:
            state = {
                      "logit": torch.zeros([batch_size, self.stoch, self.discrete_actions]).to(self.device),
                      "stoch": torch.zeros([batch_size, self.stoch, self.discrete_actions]).to(self.device),
                      "hidden": hidden,
                      }
        else:
            state = {
                "mean": torch.zeros([batch_size, self.stoch]).to(self.device),
                "std": torch.zeros([batch_size, self.stoch]).to(self.device),
                "stoch": torch.zeros([batch_size, self.stoch]).to(self.device),
                "hidden": hidden,
            }
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["hidden"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["hidden"])
            return state
        else:
            raise NotImplementedError(self._initial)
     
        return state
    def _suff_stats_layer(self, name, x):
        if self.discrete_actions:
            if name == "ims":
                x = self._ims_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self.stoch, self.discrete_actions])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._ims_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self.stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}
    
    def get_dist(self, state, dtype=None):
        if self.discrete_actions:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist


    def img_step(self, prev_state, prev_action, embed=None, sample=True):
        # (batch, stoch, discrete_num)
        prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()
        prev_stoch = prev_state["stoch"]
        if self.discrete_actions:
            shape = list(prev_stoch.shape[:-2]) + [self.stoch * self.discrete_actions]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        if self._shared:
            if embed is None:
                shape = list(prev_action.shape[:-1]) + [self.latent_dim]
                embed = torch.zeros(shape)
                
            # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action, embed)
            x = torch.cat([prev_stoch, prev_action, embed], -1)
        else:

            x = torch.cat([prev_stoch, prev_action], -1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self.fc_pre_gru(x)

        deter = prev_state["hidden"]
        # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
        x, deter = self.gru(x, [deter])
        deter = deter[0]  # Keras wraps the state in a list.

        # (batch, deter) -> (batch, hidden)
        x = self.fc_transition(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "hidden": deter, **stats}
        return prior

    def imagine(self, action, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = utils.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self.discrete_actions:
            shape = list(stoch.shape[:-2]) + [self.stoch * self.discrete_actions]
            stoch = stoch.reshape(shape)
        feat = torch.cat([stoch, state["hidden"]], -1)
        return feat
        


    def obs_step(self, prev_state, prev_action, latent, is_first, sample=True):
        # if shared is True, prior and post both use same networks(inp_layers, _img_out_layers, _ims_stat_layer)
        # otherwise, post use different network(_obs_out_layers) with prior[deter] and embed as inputs
        prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()

        if torch.sum(is_first) > 0:
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            if len(is_first.shape) >= 3:
                init_state = {key : value.unsqueeze(1).expand_as(prev_state[key]) for key, value in init_state.items()}
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        prior = self.img_step(prev_state, prev_action, None, sample)
        if self._shared:
            post = self.img_step(prev_state, prev_action, latent, sample)
        else:
            if self._temp_post:
                x = torch.cat([prior["hidden"], latent], -1)
            else:
                x = latent
            # (batch_size, prior_deter + embed) -> (batch_size, hidden)
            x = self.fc_posterior(x)
            # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
            stats = self._suff_stats_layer("obs", x)
            if sample:
                stoch = self.get_dist(stats).sample()
            else:
                stoch = self.get_dist(stats).mode()
            post = {"stoch": stoch, "hidden": prior["hidden"], **stats}
        return post, prior

    def observe(self, latent, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        # (batch, time, ch) -> (time, batch, ch)
        latent, action, is_first = swap(latent), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = utils.static_scan(
            lambda prev_state, prev_act, latent, is_first: self.obs_step(
                prev_state[0], prev_act, latent, is_first
            ),
            (action, latent, is_first),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def obs_step_by_prior(self, prev_post, now_prior, prev_action, true_latent, is_first, sample=True):
        # if shared is True, prior and post both use same networks(inp_layers, _img_out_layers, _ims_stat_layer)
        # otherwise, post use different network(_obs_out_layers) with prior[deter] and embed as inputs
        prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()

        if torch.sum(is_first) > 0:
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            if len(is_first.shape) >= 3:
                init_state = {key : value.unsqueeze(1).expand_as(prev_post[key]) for key, value in init_state.items()}
            for key, val in prev_post.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_post[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        # prior = self.img_step(prev_post, prev_action, None, sample)
        if self._shared:
            true_post = self.img_step(prev_post, prev_action, true_latent, sample)
        else:
            if self._temp_post:
                x = torch.cat([now_prior["hidden"], true_latent], -1)
            else:
                x = true_latent
            # (batch_size, prior_deter + embed) -> (batch_size, hidden)
            x = self.fc_posterior(x)
            # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
            stats = self._suff_stats_layer("obs", x)
            if sample:
                stoch = self.get_dist(stats).sample()
            else:
                stoch = self.get_dist(stats).mode()
            true_post = {"stoch": stoch, "hidden": now_prior["hidden"], **stats}
        true_init_state = self.initial(len(is_first))
        true_prev_post = {key : torch.cat([true_init_state[key].unsqueeze(1), value], dim=1)[:, :-1] for key, value in true_post.items()}
        if torch.sum(is_first) > 0:
            init_state = self.initial(len(is_first))
            if len(is_first.shape) >= 3:
                init_state = {key : value.unsqueeze(1).expand_as(true_prev_post[key]) for key, value in init_state.items()}
            for key, val in true_prev_post.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                true_prev_post[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )
        true_prior = self.img_step(true_prev_post, prev_action, None, sample)
        return true_post, true_prior
    

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self.discrete_actions else dist(post)._dist,
            dist(sg(prior)) if self.discrete_actions else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self.discrete_actions else dist(sg(post))._dist,
            dist(prior) if self.discrete_actions else dist(prior)._dist,
        )
        rep_loss = torch.mean(torch.clip(rep_loss, min=free))
        dyn_loss = torch.mean(torch.clip(dyn_loss, min=free))
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


    