
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from .subnetworks import Encoder, RSSM, Decoder, MLP, ActionHead
import numpy as np
from world_model import utils


to_np = lambda x: x.detach().cpu().numpy()

from .replay_memory import SequenceReplayMemory, Transition

import copy

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


class MBR(nn.Module):
    def __init__(
        self,
        encoder,
        img_size,
        mask_ratio,
        jumps,
        patch_size,
        block_size,
        device,
    ):
        super().__init__()

        self.device = device
        self.jumps = jumps

        self.img_size = img_size
        input_size = img_size // patch_size
   
        self.masker = CubeMaskGenerator(
            input_size=input_size,
            image_size=img_size,
            clip_size=self.jumps + 1,
            block_size=block_size,
            mask_ratio=mask_ratio,
        )  # 1 for mask, num_grid=input_size

        self.encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.transforms = []
        self.eval_transforms = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def apply_transforms(self, transforms, eval_transforms, image):
        if eval_transforms is None:
            for transform in transforms:
                image = transform(image)
        else:
            for transform, eval_transform in zip(transforms, eval_transforms):
                image = utils.maybe_transform(
                    image, transform, eval_transform, p=self.aug_prob
                )
        return image
    
    @torch.no_grad()
    def transform(self, images, augment=False):
        images = images.float() / 255.0 if images.dtype == torch.uint8 else images
        flat_images = images.reshape(-1, *images.shape[-3:])
        if augment:
            processed_images = self.apply_transforms(
                self.transforms, self.eval_transforms, flat_images
            )
        else:
            processed_images = self.apply_transforms(
                self.eval_transforms, None, flat_images
            )
        processed_images = processed_images.view(
            *images.shape[:-3], *processed_images.shape[1:]
        )
        return processed_images

    def spr_loss(self, latents, target_latents, no_grad=False):
        if no_grad:
            with torch.no_grad():
                global_latents = latents
        else:
            global_latents = latents

        with torch.no_grad():
            global_targets = target_latents
       
        loss = self.norm_mse_loss(global_latents, global_targets, mean=False).mean()

        return loss

    def norm_mse_loss(self, f_x1s, f_x2s, mean=True):
        f_x1 = F.normalize(
            f_x1s.float(), p=2.0, dim=-1, eps=1e-3
        )  # (bs*(1+jumps), 512)
        f_x2 = F.normalize(f_x2s.float(), p=2.0, dim=-1, eps=1e-3)
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1)
        loss = loss.mean(0) if mean else loss
        return loss

class RewardEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    """
    Simplified RSSM that integrates the encoder, recurrent model, and decoder.
    """
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._use_amp = True if config["precision"] == 16 else False
        self._config = config   
        self.encoder = Encoder(config).to(self.device) # Representation Model z_t = q(z_t | h_t, o_t)
        self.cnn_outshape = self.encoder.cnn_outshape
        
        self.dynamics = RSSM(config,self.device).to(self.device )     # Recurrent Model h_t = f(h_{t-1}, z_{t-1}, a_{t-1}) 
        #self.decoder = Decoder(config,self.encoder.cnn_outshape) # Observation Model o_t = p(o_t | h_t, z_t)
        

        self.MBR = MBR(
            encoder=self.encoder,
            img_size=config["image_size"][1],
            mask_ratio=config["mask_ratio"],
            jumps= config["batch_size"]+1,
            patch_size=config["patch_size"],
            block_size=config["block_size"],
            device = self.device,
        ).to(self.device)


        self.kl_free = config["kl_free"]
        self.dyn_scale = config["dyn_scale"]
        self.rep_scale = config["rep_scale"]
        self.discrete_actions = config["discrete_actions"] 

        self.target_dynamics = copy.deepcopy(self.dynamics)
        self.heads = nn.ModuleDict()
        
        feat_size = config["stoch"]*config["discrete_actions"] + config["hidden_dim"]
        


        self.sequence_length =config["memory_sequence_length"]
        self.memory = SequenceReplayMemory(capacity=config["memory_capacity"],
                                            sequence_length=self.sequence_length)
        self.batch_size = config["batch_size"]


        if config["reward_head"] == "symlog_disc":
            self.heads["reward"] = MLP(
                feat_size,  # pytorch version
                (255,),
                config["reward_layers"],
                config["units"],
                'SiLU',
                'LayerNorm',
                dist=config["reward_head"], 
                outscale=0.0,
                device=self.device,
            ).to(self.device)
        else:
            self.heads["reward"] = MLP(
                inp_dim=feat_size,  # pytorch version
                shape=[],
                layers=config["reward_layers"],
                units=config["units"],
                activation='SiLU',
                normalization='LayerNorm',
                dist=config["reward_head"],
                outscale=0.0,
                device=self.device,
            ).to(self.device)
        self.heads["cont"] = MLP(
            feat_size,  # pytorch version
            [],
            config["cont_layers"],
            config["units"],
            'SiLU',
            'LayerNorm',
            dist="binary",
            device=self.device,
        ).to(self.device)

        self._model_opt = utils.Optimizer(
            "model",
            self.parameters(),
            config["model_lr"],
            float(config["opt_eps"]),  
            config["grad_clip"],  
            config["weight_decay"],  
            opt=config["opt"],
            use_amp=self._use_amp,
        )
        self._scales = dict(reward=config["reward_scale"],
                            cont=config["cont_scale"],
                            simsr=1.0,
                            mbr=1.0)

    

 

    def save_model(self, file_path):
        """
        Save the state dictionary of the model.

        Args:
            file_path (str): The path where the state dictionary will be saved.
        """
        torch.save(self.state_dict(), file_path)
        print(f"Model's state_dict saved to {file_path}")
    
    def saveDescriptive(self, path, name):        
        self.save_model(path)
        print("Saved",name,"to file",path)
    
    
    def load_model(self, file_path):
        """
        Load the state dictionary into the model.

        Args:
            file_path (str): The path from where the state dictionary will be loaded.
            device (str): The device to map the loaded state dictionary ('cpu' or 'cuda').
        """
        state_dict = torch.load(file_path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.to(self.device)
        #print(f"Model's state_dict loaded from {file_path}")


    def update_SimSR(self, mask_post_feat, true_post_feat, mask_prior_feat, true_prior_feat, data):
        """
        Computes SimSR loss using bisimulation metrics.

        Args:
        mask_post_feat, true_post_feat (torch.Tensor): Post-step masked and true features.
        mask_prior_feat, true_prior_feat (torch.Tensor): Prior-step masked and true features.
        data (dict): Contains "reward" tensor.

        Returns:
            torch.Tensor: SimSR loss using reward and state differences.

        Notes:
            - Uses L2 normalization.
            - Computes cosine similarity for state distances.
            - Applies Huber loss on bisimulation metrics.
        """
        l2_norm = lambda x: nn.functional.normalize(x, dim=-1, p=2)
        
    
        z_a = l2_norm(mask_post_feat[:, :-1])
        z_b = l2_norm(true_post_feat[:, :-1])
        pred_a = l2_norm(mask_prior_feat[:, 1:])
        pred_b = l2_norm(true_prior_feat[:, 1:])
        reward = data["reward"][:, :-1]
        # if not reward.min() >= 0 and reward.max() <= 2:
        #     print(f"Reward range out of [0, 2], min reward = {reward.min()}, max reward = {reward.max()}")
        
        z_a, pred_a, reward = z_a.reshape(-1, z_a.shape[-1]), pred_a.reshape(-1, pred_a.shape[-1]), reward.reshape(-1, 1)
        z_b, pred_b = z_b.reshape(-1, z_b.shape[-1]), pred_b.reshape(-1, pred_b.shape[-1])
        
        def compute_dis(features_a, features_b):
            similarity_matrix = torch.matmul(features_a, features_b.T)
            dis = 1-similarity_matrix
            return dis
        
        r_diff = torch.abs(reward.T - reward)
        next_diff = compute_dis(pred_a, pred_b)
        z_diff = compute_dis(z_a, z_b)
        bisimilarity = r_diff + 0.997 * next_diff
        loss = torch.nn.HuberLoss()(z_diff, bisimilarity.detach())
        return loss


    def one_hot_flatten_actions(self,actions: torch.Tensor, num_actions: int) -> torch.Tensor:
        """
        Converts a tensor of discrete action indices into a one-hot encoded and flattened format.
        
        Args:
            actions (torch.Tensor): A tensor of shape (batch_size, sequence_length, num_units)
                                    containing discrete action indices.
            num_actions (int): The number of possible actions each unit can take.
        
        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, num_units * num_actions)
                        where each unit's action is one-hot encoded and flattened.
        """
        actions = actions.to(torch.int64)  # Ensure actions are LongTensors

        # One-hot encode the last dimension
        one_hot_actions = F.one_hot(actions, num_classes=num_actions)  # Shape: (batch, seq_len, num_units, num_actions)

        # Flatten the last two dimensions
        flattened_actions = one_hot_actions.view(*actions.shape[:2], -1).to(torch.float32)  # Shape: (batch, seq_len, num_units * num_actions)

        return flattened_actions


    def convert_sequence_to_tensor(self, sequences):
        # Convert list of sequences into batched tensors
        batch = {key: [] for key in sequences[0][0].state.keys()}
        actions, rewards, is_firsts, dones = [], [], [], []

        for seq in sequences:
            for t in seq:
                for key in t.state:
                    batch[key].append(np.array(t.state[key], dtype=np.float32))  # Ensure float32 type
            
                actions.append(np.array(t.action, dtype=np.int16))  # Ensure actions are ints
                rewards.append(np.array(t.reward, dtype=np.float32))
                is_firsts.append(np.array(t.is_first, dtype=np.int8))
                dones.append(np.array(t.done, dtype=np.int8))

        # Convert to (time, batch, feature) format
        batch = {key: torch.tensor(np.array(batch[key])).view(self.batch_size, self.sequence_length, *batch[key][0].shape[1:]).to(self.device) for key in batch}
        actions = torch.tensor(np.array(actions)).view(self.batch_size, self.sequence_length, -1)
        actions = self.one_hot_flatten_actions(actions, self._config["num_actions"]).to(self.device)
        rewards = torch.tensor(np.array(rewards)).view(self.batch_size, self.sequence_length, -1).sum(-1).to(self.device)
        dones = torch.tensor(np.array(dones)).view(self.batch_size, self.sequence_length,-1).to(self.device)
        is_firsts = torch.tensor(np.array(is_firsts)).view(self.batch_size, self.sequence_length,-1).to(self.device)    

        # for key in batch:
        #     print(key, batch[key].shape)
        # print("convert_sequence_to_tensor actions", actions.shape)
        # print("rewards", rewards.shape)
        # print("is_firsts", is_firsts.shape)
        # print("dones", dones.shape)

        return batch, actions, rewards, is_firsts, dones

    def _train(self, data, actions, is_first):
        # data {
        #       image (batch_size, batch_length, h, w, ch)
        #       scalar (batch_size, batch_length, scalar_dim)
        #       step_embedding (batch_size, batch_length, step_dim)
        #       reward (batch_size, batch_length)
        #       }
        # action (batch_size, batch_length, act_dim)
        # 
        # discount (batch_size, batch_length)

        with torch.no_grad():
            true_embed = self.MBR.target_encoder(data)
        
        mask = self.MBR.masker()
        mask = mask[:,None].expand(-1,data['image'].shape[1], data['image'].shape[2],-1,-1)
       
        masked_image = data['image'] * (1 - mask.float().to(self.device))
        masked_image = self.MBR.transform(masked_image, augment=True)
        data['image'] = masked_image
        
        with utils.RequiresGrad(self):
            with torch.amp.autocast('cuda', enabled=self._use_amp):
                embed = self.MBR.encoder(data)

                post, prior = self.dynamics.observe(
                    embed, actions, is_first
                )



                
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, self.kl_free, self.dyn_scale, self.rep_scale
                )
                feat = self.dynamics.get_feat(post)
                preds = {}
               
                for name, head in self.heads.items():
                    
                    grad_head = name in self._config["grad_heads"]
                    inp = feat if grad_head else feat.detach()
                    pred = head(inp)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                
                losses = {}
                for name, pred in preds.items():
                    if not self.discrete_actions and name == 'reward':
                        like = pred.log_prob(data[name].unsqueeze(-1))
                    else:
                        like = pred.log_prob(data[name])
                    losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
                prior_feat = self.dynamics.get_feat(prior)
                post_cont_feat = self.dynamics.get_mlp_feat(feat)
                prior_cont_feat = self.dynamics.get_mlp_feat(prior_feat)
                with torch.no_grad():
                    init_state = self.target_dynamics.initial(actions.shape[0])
                    prev_post = {key : torch.cat([init_state[key].unsqueeze(1), value], dim=1)[:, :-1] for key, value in post.items()}
                    true_post, true_prior = self.target_dynamics.obs_step_by_prior(prev_post, prior, actions, true_embed, is_first)
                    true_post_feat = self.target_dynamics.get_feat(true_post)
                    true_prior_feat = self.target_dynamics.get_feat(true_prior)
                    true_post_cont_feat = self.target_dynamics.get_mlp_feat(true_post_feat)
                if not self._config["nomlr"]:
                    losses["mbr"] = self.MBR.spr_loss(post_cont_feat, true_post_cont_feat) * self._scales.get("mbr", 1.0)
                if not self._config["nosimsr"]:
                    losses['simsr'] = self.update_SimSR(feat, true_post_feat, prior_feat, true_prior_feat, data) * self._scales.get("simsr", 1.0)
                model_loss = sum(losses.values()) + kl_loss
            metrics = self._model_opt(model_loss, self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = self.kl_free
        metrics["dyn_scale"] = self.dyn_scale
        metrics["rep_scale"] = self.rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        metrics["min_reward"] = to_np(data["reward"].min())
        metrics["max_reward"] = to_np(data["reward"].max())
        with torch.amp.autocast('cuda', enabled = self._use_amp):
            # assert "update MRB encoder"
            utils.soft_update_params(self.MBR.encoder, self.MBR.target_encoder, self._config["tau"])
            utils.soft_update_params(self.dynamics, self.target_dynamics, self._config["tau"])
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        true_post = {k: v.detach() for k, v in true_post.items()}
        return true_post, context, metrics

    def predict(self, x, state, is_first):
        with torch.no_grad():
            batch_size = 1
            sequence_length = 1
            latent, action = state
            x = {key: torch.tensor(x[key],dtype=torch.float32).view(1, 1, *x[key].shape[1:]).to(self.device) for key in x}
            action = torch.tensor(action).clone().detach().view(batch_size, sequence_length, -1)
            action = self.one_hot_flatten_actions(action, self._config["num_actions"]).squeeze(1).to(self.device)
          
            is_first = torch.tensor(is_first,dtype=torch.int8).view(batch_size,-1).to(self.device)
            embed = self.encoder(x).squeeze(1)
            latent, _ = self.dynamics.obs_step(latent, action, embed, is_first)
            feat = self.dynamics.get_feat(latent) # latent representation of observation

        return feat, latent

    def add_to_memory(self,step, state, action, reward, is_first, done):
        if step == 0:
            self.memory.clear_sequence()
            return
        self.memory.push(state, action[:, 0], reward, is_first, done)    
        return
        

    def train(self):
        metrics = {}
        # if step == 0:
        #     #print(step,state["image"][:, 12, :5, :5])
        #     return metrics
        # if is_first:
        #     #print(step, state["image"][:, 12, :5, :5])
        #self.memory.push(state, action[:, 0], reward, is_first, done)
        if len(self.memory) >= self.batch_size:
            sequences = self.memory.sample(self.batch_size)
            batch, actions, rewards, is_first, done = self.convert_sequence_to_tensor(sequences)
            batch["reward"] = rewards
            batch["cont"] = torch.Tensor(1.0 - done).to(self.device)

            #latent = self.dynamics.initial(self.batch_size)
            #feat = self.dynamics.get_feat(latent)
            #print("feat", feat.shape)
            true_post, context, metrics = self._train(batch, actions, is_first)



            # emb = self.MBR.encoder(batch)
            # post, prior = self.dynamics(emb, actions, is_first)


            # recon = self.decoder(emb)
            # print("recon", recon["image"].shape,
            #        recon["scalars"].shape, recon["step_embedding"].shape)
        
        
            
        return metrics 
    





# class ImagBehavior(nn.Module):
#     def __init__(self, config, world_model, stop_grad_actor=True, reward=None):
#         super(ImagBehavior, self).__init__()
#         self._use_amp = True if config.precision == 16 else False
#         self._config = config
#         self._world_model = world_model
#         self._stop_grad_actor = stop_grad_actor
#         self._reward = reward
#         self.discrete_actions = config["discrete_actions"]
#         if self.discrete_actions:
#             feat_size = config["stoch"] * self.discrete_actions + config["hidden_dim"]
#         else:
#             feat_size = config["stoch"] + config["hidden_dim"]
#         self.actor = ActionHead(
#             feat_size,
#             config["num_actions"],
#             config.actor_layers,
#             config.units,
#             config.act,
#             config.norm,
#             config.actor_dist,
#             config.actor_init_std,
#             config.actor_min_std,
#             config.actor_max_std,
#             config.actor_temp,
#             outscale=1.0,
#             unimix_ratio=config.action_unimix_ratio,
#         )
#         if config.value_head == "symlog_disc":
#             self.value = MLP(
#                 feat_size,
#                 (255,),
#                 config.value_layers,
#                 config.units,
#                 config.act,
#                 config.norm,
#                 config.value_head,
#                 outscale=0.0,
#                 device=config.device,
#             )
#         else:
#             self.value = MLP(
#                 feat_size,
#                 [],
#                 config.value_layers,
#                 config.units,
#                 config.act,
#                 config.norm,
#                 config.value_head,
#                 outscale=0.0,
#                 device=config.device,
#             )
#         if config.slow_value_target:
#             self._slow_value = copy.deepcopy(self.value)
#             self._updates = 0
#         kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
#         self._actor_opt = utils.Optimizer(
#             "actor",
#             self.actor.parameters(),
#             config.actor_lr,
#             config.ac_opt_eps,
#             config.actor_grad_clip,
#             **kw,
#         )
#         self._value_opt = utils.Optimizer(
#             "value",
#             self.value.parameters(),
#             config.value_lr,
#             config.ac_opt_eps,
#             config.value_grad_clip,
#             **kw,
#         )
#         if self._config.reward_EMA:
#             self.reward_ema = RewardEMA(device=self._config.device)

#     def _train(
#         self,
#         start,
#         objective=None,
#         action=None,
#         reward=None,
#         imagine=None,
#         tape=None,
#         repeats=None,
#     ):
#         objective = objective or self._reward
#         self._update_slow_target()
#         metrics = {}

#         with tools.RequiresGrad(self.actor):
#             with torch.cuda.amp.autocast(self._use_amp):
#                 imag_feat, imag_state, imag_action = self._imagine(
#                     start, self.actor, self._config.imag_horizon, repeats
#                 )
#                 reward = objective(imag_feat, imag_state, imag_action)
#                 actor_ent = self.actor(imag_feat).entropy()
#                 state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
#                 # this target is not scaled
#                 # slow is flag to indicate whether slow_target is used for lambda-return
#                 target, weights, base = self._compute_target(
#                     imag_feat, imag_state, imag_action, reward, actor_ent, state_ent
#                 )
#                 actor_loss, mets = self._compute_actor_loss(
#                     imag_feat,
#                     imag_state,
#                     imag_action,
#                     target,
#                     actor_ent,
#                     state_ent,
#                     weights,
#                     base,
#                 )
#                 metrics.update(mets)
#                 value_input = imag_feat

#         with tools.RequiresGrad(self.value):
#             with torch.cuda.amp.autocast(self._use_amp):
#                 value = self.value(value_input[:-1].detach())
#                 target = torch.stack(target, dim=1)
#                 # (time, batch, 1), (time, batch, 1) -> (time, batch)
#                 value_loss = -value.log_prob(target.detach())
#                 slow_target = self._slow_value(value_input[:-1].detach())
#                 if self._config.slow_value_target:
#                     value_loss = value_loss - value.log_prob(
#                         slow_target.mode().detach()
#                     )
#                 if self._config.value_decay:
#                     value_loss += self._config.value_decay * value.mode()
#                 # (time, batch, 1), (time, batch, 1) -> (1,)
#                 value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

#         metrics.update(tools.tensorstats(value.mode(), "value"))
#         metrics.update(tools.tensorstats(target, "target"))
#         metrics.update(tools.tensorstats(reward, "imag_reward"))
#         if self._config.actor_dist in ["onehot"]:
#             metrics.update(
#                 tools.tensorstats(
#                     torch.argmax(imag_action, dim=-1).float(), "imag_action"
#                 )
#             )
#         else:
#             metrics.update(tools.tensorstats(imag_action, "imag_action"))
#         metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
#         with tools.RequiresGrad(self):
#             metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
#             metrics.update(self._value_opt(value_loss, self.value.parameters()))
#         return imag_feat, imag_state, imag_action, weights, metrics

#     def _imagine(self, start, policy, horizon, repeats=None):
#         dynamics = self._world_model.dynamics
#         if repeats:
#             raise NotImplemented("repeats is not implemented in this version")
#         flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
#         start = {k: flatten(v) for k, v in start.items()}

#         def step(prev, _):
#             state, _, _ = prev
#             feat = dynamics.get_feat(state)
#             inp = feat.detach() if self._stop_grad_actor else feat
#             action = policy(inp).sample()
#             succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
#             return succ, feat, action

#         succ, feats, actions = static_scan(
#             step, [torch.arange(horizon)], (start, None, None)
#         )
#         states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
#         if repeats:
#             raise NotImplemented("repeats is not implemented in this version")

#         return feats, states, actions

#     def _compute_target(
#         self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent
#     ):
#         if "cont" in self._world_model.heads:
#             inp = self._world_model.dynamics.get_feat(imag_state)
#             discount = self._config.discount * self._world_model.heads["cont"](inp).mean
#         else:
#             discount = self._config.discount * torch.ones_like(reward)
#         if self._config.future_entropy and self._config.actor_entropy > 0:
#             reward += self._config.actor_entropy * actor_ent
#         if self._config.future_entropy and self._config.actor_state_entropy > 0:
#             reward += self._config.actor_state_entropy * state_ent
#         value = self.value(imag_feat).mode()
#         target = utils.lambda_return(
#             reward[1:],
#             value[:-1],
#             discount[1:],
#             bootstrap=value[-1],
#             lambda_=self._config.discount_lambda,
#             axis=0,
#         )
#         weights = torch.cumprod(
#             torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
#         ).detach()
#         return target, weights, value[:-1]

#     def _compute_actor_loss(
#         self,
#         imag_feat,
#         imag_state,
#         imag_action,
#         target,
#         actor_ent,
#         state_ent,
#         weights,
#         base,
#     ):
#         metrics = {}
#         inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
#         policy = self.actor(inp)
#         actor_ent = policy.entropy()
#         # Q-val for actor is not transformed using symlog
#         target = torch.stack(target, dim=1)
#         if self._config.reward_EMA:
#             offset, scale = self.reward_ema(target)
#             normed_target = (target - offset) / scale
#             normed_base = (base - offset) / scale
#             adv = normed_target - normed_base
#             metrics.update(tools.tensorstats(normed_target, "normed_target"))
#             values = self.reward_ema.values
#             metrics["EMA_005"] = to_np(values[0])
#             metrics["EMA_095"] = to_np(values[1])

#         if self._config.imag_gradient == "dynamics":
#             actor_target = adv
#         elif self._config.imag_gradient == "reinforce":
#             actor_target = (
#                 policy.log_prob(imag_action)[:-1][:, :, None]
#                 * (target - self.value(imag_feat[:-1]).mode()).detach()
#             )
#         elif self._config.imag_gradient == "both":
#             actor_target = (
#                 policy.log_prob(imag_action)[:-1][:, :, None]
#                 * (target - self.value(imag_feat[:-1]).mode()).detach()
#             )
#             mix = self._config.imag_gradient_mix
#             actor_target = mix * target + (1 - mix) * actor_target
#             metrics["imag_gradient_mix"] = mix
#         else:
#             raise NotImplementedError(self._config.imag_gradient)
#         if not self._config.future_entropy and self._config.actor_entropy > 0:
#             actor_entropy = self._config.actor_entropy * actor_ent[:-1][:, :, None]
#             actor_target += actor_entropy
#         if not self._config.future_entropy and (self._config.actor_state_entropy > 0):
#             state_entropy = self._config.actor_state_entropy * state_ent[:-1]
#             actor_target += state_entropy
#             metrics["actor_state_entropy"] = to_np(torch.mean(state_entropy))
#         actor_loss = -torch.mean(weights[:-1] * actor_target)
#         return actor_loss, metrics

#     def _update_slow_target(self):
#         if self._config.slow_value_target:
#             if self._updates % self._config.slow_target_update == 0:
#                 mix = self._config.slow_target_fraction
#                 for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
#                     d.data = mix * s.data + (1 - mix) * d.data
#             self._updates += 1



