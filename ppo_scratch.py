import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as spaces
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.distributions import Categorical, Normal
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm  

# --- Utils ---

def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)

# --- Buffer ---

class RolloutBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device='cpu', gae_lambda=1, gamma=0.99, n_envs=1):
        self.buffer_size = buffer_size
        self.obs_shape = observation_space.shape
        self.action_dim = action_space.shape[0] if len(action_space.shape) > 0 else 1
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.observations = np.zeros((buffer_size, n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.returns = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.advantages = np.zeros((buffer_size, n_envs), dtype=np.float32)

    def reset(self):
        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, episode_start, value, log_prob):
        if len(log_prob.shape) == 0: log_prob = log_prob.reshape(-1, 1)
        
        self.observations[self.pos] = np.array(obs).copy()
        # Ensure action is broadcastable to storage shape
        self.actions[self.pos] = np.array(action).reshape(self.n_envs, self.action_dim).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values, dones):
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def get(self, batch_size=None):
        assert self.full
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        
        # Flatten
        obs = self.observations.reshape((-1, *self.obs_shape))
        act = self.actions.reshape((-1, self.action_dim))
        log_prob = self.log_probs.reshape(-1)
        tgt_val = self.values.reshape(-1)
        adv = self.advantages.reshape(-1)
        ret = self.returns.reshape(-1)

        # To Tensor
        obs = th.tensor(obs).to(self.device)
        act = th.tensor(act).to(self.device)
        log_prob = th.tensor(log_prob).to(self.device)
        tgt_val = th.tensor(tgt_val).to(self.device)
        adv = th.tensor(adv).to(self.device)
        ret = th.tensor(ret).to(self.device)

        start_idx = 0
        count = self.buffer_size * self.n_envs
        if batch_size is None: batch_size = count
        
        while start_idx < count:
            idx = indices[start_idx : start_idx + batch_size]
            yield obs[idx], act[idx], tgt_val[idx], log_prob[idx], adv[idx], ret[idx]
            start_idx += batch_size

# --- Policy ---

class MlpPolicy(nn.Module):
    def __init__(self, observation_space, action_space, net_arch=[64, 64], activation_fn=nn.Tanh):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        input_dim = np.prod(observation_space.shape)
        
        # Shared feature extractor (Flatten)
        self.flatten = nn.Flatten()
        
        # Policy Network
        pi_layers = []
        last_dim = input_dim
        for hidden in net_arch:
            pi_layers.append(nn.Linear(last_dim, hidden))
            pi_layers.append(activation_fn())
            last_dim = hidden
        self.pi_net = nn.Sequential(*pi_layers)
        
        # Value Network
        vf_layers = []
        last_dim_vf = input_dim
        for hidden in net_arch:
            vf_layers.append(nn.Linear(last_dim_vf, hidden))
            vf_layers.append(activation_fn())
            last_dim_vf = hidden
        self.vf_net = nn.Sequential(*vf_layers)
        
        self.value_head = nn.Linear(last_dim_vf, 1)
        
        # Action Head
        if hasattr(action_space, 'n'):  # Discrete
            self.action_head = nn.Linear(last_dim, action_space.n)
            self.is_discrete = True
        else: # Box (Continuous)
            self.action_mean = nn.Linear(last_dim, action_space.shape[0])
            self.log_std = nn.Parameter(th.zeros(action_space.shape[0]))
            self.is_discrete = False

    def forward(self, obs):
        features = self.flatten(obs)
        pi_latent = self.pi_net(features)
        vf_latent = self.vf_net(features)
        
        values = self.value_head(vf_latent)
        
        if self.is_discrete:
            logits = self.action_head(pi_latent)
            dist = Categorical(logits=logits)
        else:
            mean = self.action_mean(pi_latent)
            dist = Normal(mean, self.log_std.exp())
            
        return dist, values

    def predict(self, obs, deterministic=False):
        with th.no_grad():
            dist, _ = self.forward(obs)
            if deterministic:
                if self.is_discrete: return dist.probs.argmax(dim=1).cpu().numpy(), None
                else: return dist.mean.cpu().numpy(), None
            return dist.sample().cpu().numpy(), None

    def evaluate_actions(self, obs, actions):
        dist, values = self.forward(obs)
        if self.is_discrete:
            log_prob = dist.log_prob(actions.squeeze(-1))
        else:
            log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1) if not self.is_discrete else dist.entropy()
        return values, log_prob, entropy

# --- Algorithm ---

class PPO:
    def __init__(
        self,
        policy: str,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
    ):
        self.env = env
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.policy_kwargs = policy_kwargs if policy_kwargs else {}
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log
        
        # Initialize TensorBoard writer
        self.writer = None
        if self.tensorboard_log is not None:
            self.writer = SummaryWriter(log_dir=self.tensorboard_log)

        if device == "auto":
            self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        else:
            self.device = th.device(device)

        if seed is not None:
            th.manual_seed(seed)
            np.random.seed(seed)

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.policy = MlpPolicy(self.observation_space, self.action_space, **self.policy_kwargs).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        
        self.rollout_buffer = RolloutBuffer(
            self.n_steps, self.observation_space, self.action_space, 
            device=self.device, gae_lambda=self.gae_lambda, gamma=self.gamma, n_envs=env.num_envs
        )
        
        self._last_obs = self.env.reset()
        self._last_episode_starts = np.ones((env.num_envs,), dtype=bool)
        self.num_timesteps = 0
        
        # Buffers for tracking episode stats
        self.ep_rew_buffer = deque(maxlen=100)
        self.ep_len_buffer = deque(maxlen=100)
        
        # Temporary storage for current episode
        self.curr_episode_rewards = np.zeros(env.num_envs)
        self.curr_episode_lengths = np.zeros(env.num_envs)

    def learn(self, total_timesteps, progress_bar=False):
        if self.verbose > 0: print("Start training")
        n_updates = 0
        
        # Setup progress bar
        pbar = tqdm(total=total_timesteps, disable=not progress_bar, desc="Training")

        while self.num_timesteps < total_timesteps:
            self.policy.train(False)
            self.rollout_buffer.reset()
            
            # Collect Rollouts
            for _ in range(self.n_steps):
                with th.no_grad():
                    obs_tensor = th.as_tensor(self._last_obs).to(self.device).float()
                    dist, values = self.policy(obs_tensor)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                    
                    if not self.policy.is_discrete:
                        log_probs = log_probs.sum(dim=-1)

                actions_np = actions.cpu().numpy()
                new_obs, rewards, dones, infos = self.env.step(actions_np)
                self.num_timesteps += self.env.num_envs
                pbar.update(self.env.num_envs) # Update progress bar

                # Update stats for current episodes
                self.curr_episode_rewards += rewards
                self.curr_episode_lengths += 1

                # Handle Done Episodes
                for i, done in enumerate(dones):
                    if done:
                        # Log episode info
                        self.ep_rew_buffer.append(self.curr_episode_rewards[i])
                        self.ep_len_buffer.append(self.curr_episode_lengths[i])
                        self.curr_episode_rewards[i] = 0
                        self.curr_episode_lengths[i] = 0

                    # Handle Timeouts (bootstrapping)
                    if done and infos[i].get("TimeLimit.truncated", False):
                        terminal_obs = infos[i]["terminal_observation"]
                        with th.no_grad():
                            terminal_obs_t = th.as_tensor(terminal_obs).to(self.device).float()
                            _, term_val = self.policy(terminal_obs_t.unsqueeze(0))
                        rewards[i] += self.gamma * term_val.squeeze().item()

                self.rollout_buffer.add(
                    self._last_obs, 
                    # Fix: Always reshape to (N, 1) if discrete to match buffer dims
                    actions_np.reshape(self.env.num_envs, -1), 
                    rewards, 
                    self._last_episode_starts, 
                    values, 
                    log_probs
                )
                self._last_obs = new_obs
                self._last_episode_starts = dones

            # Compute GAE
            with th.no_grad():
                obs_tensor = th.as_tensor(new_obs).to(self.device).float()
                _, last_values = self.policy(obs_tensor)
            self.rollout_buffer.compute_returns_and_advantage(last_values, dones)

            # Train
            self.policy.train(True)
            policy_losses, value_losses, entropy_losses = [], [], []
            
            for epoch in range(self.n_epochs):
                for obs, acts, old_vals, old_log_probs, advs, rets in self.rollout_buffer.get(self.batch_size):
                    if self.policy.is_discrete: acts = acts.long().flatten()
                    
                    vals, log_prob, entropy = self.policy.evaluate_actions(obs, acts)
                    vals = vals.flatten()
                    
                    # Normalize Adv
                    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                    ratio = th.exp(log_prob - old_log_probs)
                    policy_loss_1 = advs * ratio
                    policy_loss_2 = advs * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                    
                    value_loss = nn.functional.mse_loss(rets, vals)
                    entropy_loss = -th.mean(entropy)
                    
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                    # Log losses
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(entropy_loss.item())
            
            n_updates += 1
            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

            # --- Write logs to TensorBoard ---
            if self.writer is not None:
                # Log Rollout stats
                if len(self.ep_rew_buffer) > 0:
                    self.writer.add_scalar("rollout/ep_rew_mean", np.mean(self.ep_rew_buffer), self.num_timesteps)
                    self.writer.add_scalar("rollout/ep_len_mean", np.mean(self.ep_len_buffer), self.num_timesteps)
                
                # Log Train stats
                self.writer.add_scalar("train/loss", loss.item(), self.num_timesteps)
                self.writer.add_scalar("train/policy_gradient_loss", np.mean(policy_losses), self.num_timesteps)
                self.writer.add_scalar("train/value_loss", np.mean(value_losses), self.num_timesteps)
                self.writer.add_scalar("train/entropy_loss", np.mean(entropy_losses), self.num_timesteps)
                self.writer.add_scalar("train/explained_variance", explained_var, self.num_timesteps)
            
        pbar.close()
        if self.writer:
            self.writer.flush()
            self.writer.close()

        if self.verbose > 0: print("Training finished")
        return self

    def save(self, path):
        th.save(self.policy.state_dict(), path + ".pth")

# --- Main Script ---

import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
from warehouse_env import WarehouseEnv
import os

N_ENVS = 8 
TOTAL_STEPS = 2_000_000 
GRID_SIZE = 8

def train_robust_model():
    print(f"\n>>> STARTING TRAINING: STRICT REWARDS MODE")
    
    env = make_vec_env(lambda: WarehouseEnv(grid_size=GRID_SIZE, reward_type="shaped"), n_envs=N_ENVS)
    device = "cuda" if th.cuda.is_available() else "cpu"
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=0, 
        device=device, 
        batch_size=512, 
        n_steps=4096, 
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log="./logs/strict/"
    )
    
    model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)
    model.save("warehouse_strict_agent")
    print(">>> TRAINING COMPLETE.")

if __name__ == "__main__":
    train_robust_model()