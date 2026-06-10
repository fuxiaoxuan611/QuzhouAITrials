import warnings
from copy import deepcopy
from typing import Any, ClassVar, Generator, NamedTuple, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from torch import nn
from torch.nn import functional as F

try:
    from sb3_contrib import RecurrentPPO
    from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer, create_sequencers
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
    from sb3_contrib.common.recurrent.type_aliases import RNNStates
except ImportError:  # pragma: no cover - sb3-contrib is a project dependency but keep imports defensive.
    RecurrentPPO = None
    RecurrentRolloutBuffer = None
    RecurrentActorCriticPolicy = None
    RNNStates = None
    create_sequencers = None


class CostRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    costs: th.Tensor
    cost_values: th.Tensor
    cost_advantages: th.Tensor
    cost_returns: th.Tensor


class CostRecurrentRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    costs: th.Tensor
    cost_values: th.Tensor
    cost_advantages: th.Tensor
    cost_returns: th.Tensor
    lstm_states: Any
    episode_starts: th.Tensor
    mask: th.Tensor


class Lagrange:
    """Non-negative Lagrange multiplier updated from observed episode cost."""

    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float,
        lagrangian_multiplier_lr: float,
        lagrangian_upper_bound: float | None = None,
    ) -> None:
        self.cost_limit = float(cost_limit)
        self.lagrangian_multiplier_lr = float(lagrangian_multiplier_lr)
        self.lagrangian_upper_bound = lagrangian_upper_bound
        init_value = max(float(lagrangian_multiplier_init), 0.0)
        self._lagrangian_multiplier = th.nn.Parameter(th.as_tensor(init_value), requires_grad=True)
        self.lambda_optimizer = th.optim.Adam([self._lagrangian_multiplier], lr=self.lagrangian_multiplier_lr)
        self.last_lambda_loss = 0.0
        self.last_cost_violation = 0.0

    @property
    def lagrangian_multiplier(self) -> float:
        return max(float(self._lagrangian_multiplier.detach().cpu().item()), 0.0)

    def compute_lambda_loss(self, mean_ep_cost: float) -> th.Tensor:
        return -self._lagrangian_multiplier * (float(mean_ep_cost) - self.cost_limit)

    def update_lagrange_multiplier(self, mean_ep_cost: float) -> None:
        self.last_cost_violation = float(mean_ep_cost) - self.cost_limit
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(mean_ep_cost)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self._lagrangian_multiplier.data.clamp_(0.0, self.lagrangian_upper_bound)
        self.last_lambda_loss = float(lambda_loss.detach().cpu().item())


def _sb3_distribution_kl(current: Any, old: Any) -> th.Tensor:
    """Compute per-sample KL for SB3 distribution wrappers."""
    current_dist = getattr(current, "distribution", None)
    old_dist = getattr(old, "distribution", None)
    if isinstance(current_dist, list) and isinstance(old_dist, list):
        return sum(th.distributions.kl_divergence(curr, prev) for curr, prev in zip(current_dist, old_dist))
    if current_dist is None or old_dist is None:
        raise TypeError(f"Unsupported distribution pair: {type(current).__name__}, {type(old).__name__}")
    kl = th.distributions.kl_divergence(current_dist, old_dist)
    if kl.ndim > 1:
        kl = kl.sum(dim=-1)
    return kl


class CostRolloutBuffer(RolloutBuffer):
    costs: np.ndarray
    cost_values: np.ndarray
    cost_advantages: np.ndarray
    cost_returns: np.ndarray

    def reset(self) -> None:
        super().reset()
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.current_episode_costs = np.zeros(self.n_envs, dtype=np.float32)
        self.completed_episode_costs: list[float] = []
        self.component_sums: dict[str, float] = {}

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        cost_value: th.Tensor,
        log_prob: th.Tensor,
        cost: np.ndarray,
        dones: np.ndarray | None = None,
        cost_components: list[dict[str, float]] | None = None,
    ) -> None:
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        action = action.reshape((self.n_envs, self.action_dim))
        costs = np.asarray(cost, dtype=np.float32).reshape((self.n_envs,))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.cost_values[self.pos] = cost_value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.costs[self.pos] = costs

        self.current_episode_costs += costs
        if cost_components is not None:
            for components in cost_components:
                for key, value_ in components.items():
                    self.component_sums[key] = self.component_sums.get(key, 0.0) + float(value_)

        if dones is not None:
            for env_idx, done in enumerate(dones):
                if done:
                    self.completed_episode_costs.append(float(self.current_episode_costs[env_idx]))
                    self.current_episode_costs[env_idx] = 0.0

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(
        self,
        last_values: th.Tensor,
        last_cost_values: th.Tensor,
        dones: np.ndarray,
    ) -> None:
        last_values = last_values.clone().cpu().numpy().flatten()
        last_cost_values = last_cost_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        last_cost_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
                next_cost_values = last_cost_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
                next_cost_values = self.cost_values[step + 1]

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            cost_delta = self.costs[step] + self.gamma * next_cost_values * next_non_terminal - self.cost_values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            last_cost_gae_lam = cost_delta + self.gamma * self.gae_lambda * next_non_terminal * last_cost_gae_lam
            self.advantages[step] = last_gae_lam
            self.cost_advantages[step] = last_cost_gae_lam

        self.returns = self.advantages + self.values
        self.cost_returns = self.cost_advantages + self.cost_values

    def mean_episode_cost(self) -> float:
        if self.completed_episode_costs:
            return float(np.mean(self.completed_episode_costs))
        return float(np.mean(np.sum(self.costs, axis=0)))

    def mean_cost_components(self) -> dict[str, float]:
        denominator = max(self.buffer_size * self.n_envs, 1)
        return {key: value / denominator for key, value in self.component_sums.items()}

    def get(self, batch_size: int | None = None) -> Generator[CostRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "costs",
                "cost_values",
                "cost_advantages",
                "cost_returns",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Any = None) -> CostRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.costs[batch_inds].flatten(),
            self.cost_values[batch_inds].flatten(),
            self.cost_advantages[batch_inds].flatten(),
            self.cost_returns[batch_inds].flatten(),
        )
        return CostRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class CostActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy with an additional scalar cost-value head."""

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self.cost_value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            self.cost_value_net.apply(self.init_weights)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_latents(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return latent_pi, latent_vf

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        latent_pi, latent_vf = self._get_latents(obs)
        values = self.value_net(latent_vf)
        cost_values = self.cost_value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, cost_values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor | None]:
        latent_pi, latent_vf = self._get_latents(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        cost_values = self.cost_value_net(latent_vf)
        return values, cost_values, log_prob, distribution.entropy()

    def predict_values(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf), self.cost_value_net(latent_vf)


SelfLagrangianPPO = TypeVar("SelfLagrangianPPO", bound="LagrangianPPO")


class LagrangianPPO(PPO):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": CostActorCriticPolicy,
    }

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        *args: Any,
        cost_key: str = "cost",
        cost_limit: float = 0.0,
        lambda_init: float = 1.0,
        lambda_lr: float = 0.05,
        lambda_max: float | None = None,
        cost_vf_coef: float = 0.7,
        use_cost_value_function: bool = True,
        lagrangian_update_frequency: str = "rollout",
        constraint_fn: Any | None = None,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        **kwargs: Any,
    ) -> None:
        if constraint_fn is not None:
            warnings.warn(
                "constraint_fn is deprecated and ignored. Define constraints in the environment via info['cost'].",
                DeprecationWarning,
                stacklevel=2,
            )
        self.cost_key = cost_key
        self.cost_vf_coef = cost_vf_coef
        self.use_cost_value_function = use_cost_value_function
        self.lagrangian_update_frequency = lagrangian_update_frequency
        self.lagrange = Lagrange(
            cost_limit=cost_limit,
            lagrangian_multiplier_init=lambda_init,
            lagrangian_multiplier_lr=lambda_lr,
            lagrangian_upper_bound=lambda_max,
        )
        self._last_mean_ep_cost = 0.0
        self._last_cost_components: dict[str, float] = {}
        super().__init__(
            policy,
            env,
            *args,
            rollout_buffer_class=rollout_buffer_class or CostRolloutBuffer,
            **kwargs,
        )

    def _coerce_costs(self, infos: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([float(info.get(self.cost_key, 0.0)) for info in infos], dtype=np.float32)

    @staticmethod
    def _coerce_cost_components(infos: list[dict[str, Any]]) -> list[dict[str, float]]:
        components = []
        for info in infos:
            raw_components = info.get("cost_components", {})
            if isinstance(raw_components, dict):
                components.append({str(key): float(value) for key, value in raw_components.items()})
            else:
                components.append({})
        return components

    def _update_lagrange_from_rollout(self) -> None:
        assert isinstance(self.rollout_buffer, CostRolloutBuffer)
        self._last_mean_ep_cost = self.rollout_buffer.mean_episode_cost()
        self._last_cost_components = self.rollout_buffer.mean_cost_components()
        if self.lagrangian_update_frequency == "rollout":
            self.lagrange.update_lagrange_multiplier(self._last_mean_ep_cost)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert isinstance(rollout_buffer, CostRolloutBuffer), "LagrangianPPO requires CostRolloutBuffer"
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, cost_values, log_probs = self.policy(obs_tensor)
                if not self.use_cost_value_function:
                    cost_values = th.zeros_like(values)
            actions = actions.cpu().numpy()
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            costs = self._coerce_costs(infos)
            cost_components = self._coerce_cost_components(infos)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value, terminal_cost_value = self.policy.predict_values(terminal_obs)
                    rewards[idx] += self.gamma * terminal_value[0]
                    if self.use_cost_value_function:
                        costs[idx] += float(self.gamma * terminal_cost_value[0].detach().cpu().item())

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                cost_values,
                log_probs,
                costs,
                dones=dones,
                cost_components=cost_components,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values, cost_values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
            if not self.use_cost_value_function:
                cost_values = th.zeros_like(values)

        rollout_buffer.compute_returns_and_advantage(last_values=values, last_cost_values=cost_values, dones=dones)
        self._update_lagrange_from_rollout()

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, cost_value_losses = [], [], []
        clip_fractions = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, cost_values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                cost_values = cost_values.flatten()

                advantages = rollout_data.advantages
                cost_advantages = rollout_data.cost_advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)

                lagrangian_multiplier = self.lagrange.lagrangian_multiplier
                combined_advantages = (advantages - lagrangian_multiplier * cost_advantages) / (1.0 + lagrangian_multiplier)
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = combined_advantages * ratio
                policy_loss_2 = combined_advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if self.use_cost_value_function:
                    cost_value_loss = F.mse_loss(rollout_data.cost_returns, cost_values)
                else:
                    cost_value_loss = th.zeros((), device=self.device)
                cost_value_losses.append(float(cost_value_loss.detach().cpu().item()))

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.cost_vf_coef * cost_value_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        cost_explained_var = explained_variance(
            self.rollout_buffer.cost_values.flatten(),
            self.rollout_buffer.cost_returns.flatten(),
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/cost_value_loss", np.mean(cost_value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/cost_explained_variance", cost_explained_var)
        self.logger.record("train/lagrangian_multiplier", self.lagrange.lagrangian_multiplier)
        self.logger.record("train/cost_limit", self.lagrange.cost_limit)
        self.logger.record("train/mean_ep_cost", self._last_mean_ep_cost)
        self.logger.record("train/cost_violation", self.lagrange.last_cost_violation)
        self.logger.record("train/lambda_loss", self.lagrange.last_lambda_loss)
        for key, value in self._last_cost_components.items():
            self.logger.record(f"train/cost_component/{key}", value)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfLagrangianPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "LagrangianPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfLagrangianPPO:
        return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)


if RecurrentRolloutBuffer is not None:

    class CostRecurrentRolloutBuffer(RecurrentRolloutBuffer):
        def reset(self) -> None:
            super().reset()
            self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
            self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
            self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
            self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
            self.current_episode_costs = np.zeros(self.n_envs, dtype=np.float32)
            self.completed_episode_costs: list[float] = []
            self.component_sums: dict[str, float] = {}

        def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            cost_value: th.Tensor,
            log_prob: th.Tensor,
            cost: np.ndarray,
            lstm_states: Any,
            dones: np.ndarray | None = None,
            cost_components: list[dict[str, float]] | None = None,
        ) -> None:
            self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0].cpu().numpy())
            self.cell_states_pi[self.pos] = np.array(lstm_states.pi[1].cpu().numpy())
            self.hidden_states_vf[self.pos] = np.array(lstm_states.vf[0].cpu().numpy())
            self.cell_states_vf[self.pos] = np.array(lstm_states.vf[1].cpu().numpy())

            if len(log_prob.shape) == 0:
                log_prob = log_prob.reshape(-1, 1)
            if isinstance(self.observation_space, spaces.Discrete):
                obs = obs.reshape((self.n_envs, *self.obs_shape))
            action = action.reshape((self.n_envs, self.action_dim))
            costs = np.asarray(cost, dtype=np.float32).reshape((self.n_envs,))

            self.observations[self.pos] = np.array(obs)
            self.actions[self.pos] = np.array(action)
            self.rewards[self.pos] = np.array(reward)
            self.episode_starts[self.pos] = np.array(episode_start)
            self.values[self.pos] = value.clone().cpu().numpy().flatten()
            self.cost_values[self.pos] = cost_value.clone().cpu().numpy().flatten()
            self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
            self.costs[self.pos] = costs
            self.current_episode_costs += costs

            if cost_components is not None:
                for components in cost_components:
                    for key, value_ in components.items():
                        self.component_sums[key] = self.component_sums.get(key, 0.0) + float(value_)
            if dones is not None:
                for env_idx, done in enumerate(dones):
                    if done:
                        self.completed_episode_costs.append(float(self.current_episode_costs[env_idx]))
                        self.current_episode_costs[env_idx] = 0.0

            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True

        def compute_returns_and_advantage(
            self,
            last_values: th.Tensor,
            last_cost_values: th.Tensor,
            dones: np.ndarray,
        ) -> None:
            last_values = last_values.clone().cpu().numpy().flatten()
            last_cost_values = last_cost_values.clone().cpu().numpy().flatten()
            last_gae_lam = 0
            last_cost_gae_lam = 0
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32)
                    next_values = last_values
                    next_cost_values = last_cost_values
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_values = self.values[step + 1]
                    next_cost_values = self.cost_values[step + 1]
                delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
                cost_delta = self.costs[step] + self.gamma * next_cost_values * next_non_terminal - self.cost_values[step]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                last_cost_gae_lam = cost_delta + self.gamma * self.gae_lambda * next_non_terminal * last_cost_gae_lam
                self.advantages[step] = last_gae_lam
                self.cost_advantages[step] = last_cost_gae_lam
            self.returns = self.advantages + self.values
            self.cost_returns = self.cost_advantages + self.cost_values

        def mean_episode_cost(self) -> float:
            if self.completed_episode_costs:
                return float(np.mean(self.completed_episode_costs))
            return float(np.mean(np.sum(self.costs, axis=0)))

        def mean_cost_components(self) -> dict[str, float]:
            denominator = max(self.buffer_size * self.n_envs, 1)
            return {key: value / denominator for key, value in self.component_sums.items()}

        def get(self, batch_size: int | None = None) -> Generator[CostRecurrentRolloutBufferSamples, None, None]:
            assert self.full, "Rollout buffer must be full before sampling from it"
            if not self.generator_ready:
                for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                    self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)
                for tensor in [
                    "observations",
                    "actions",
                    "values",
                    "log_probs",
                    "advantages",
                    "returns",
                    "costs",
                    "cost_values",
                    "cost_advantages",
                    "cost_returns",
                    "hidden_states_pi",
                    "cell_states_pi",
                    "hidden_states_vf",
                    "cell_states_vf",
                    "episode_starts",
                ]:
                    self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
                self.generator_ready = True

            if batch_size is None:
                batch_size = self.buffer_size * self.n_envs

            split_index = np.random.randint(self.buffer_size * self.n_envs)
            indices = np.arange(self.buffer_size * self.n_envs)
            indices = np.concatenate((indices[split_index:], indices[:split_index]))
            env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
            env_change[0, :] = 1.0
            env_change = self.swap_and_flatten(env_change)

            start_idx = 0
            while start_idx < self.buffer_size * self.n_envs:
                batch_inds = indices[start_idx : start_idx + batch_size]
                yield self._get_samples(batch_inds, env_change)
                start_idx += batch_size

        def _get_samples(
            self,
            batch_inds: np.ndarray,
            env_change: np.ndarray,
            env: Any = None,
        ) -> CostRecurrentRolloutBufferSamples:
            self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
                self.episode_starts[batch_inds], env_change[batch_inds], self.device
            )
            n_seq = len(self.seq_start_indices)
            max_length = self.pad(self.actions[batch_inds]).shape[1]
            padded_batch_size = n_seq * max_length
            lstm_states_pi = (
                self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
                self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            )
            lstm_states_vf = (
                self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
                self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            )
            lstm_states_pi = (self.to_torch(lstm_states_pi[0]).contiguous(), self.to_torch(lstm_states_pi[1]).contiguous())
            lstm_states_vf = (self.to_torch(lstm_states_vf[0]).contiguous(), self.to_torch(lstm_states_vf[1]).contiguous())
            return CostRecurrentRolloutBufferSamples(
                observations=self.pad(self.observations[batch_inds]).reshape((padded_batch_size, *self.obs_shape)),
                actions=self.pad(self.actions[batch_inds]).reshape((padded_batch_size, *self.actions.shape[1:])),
                old_values=self.pad_and_flatten(self.values[batch_inds]),
                old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
                advantages=self.pad_and_flatten(self.advantages[batch_inds]),
                returns=self.pad_and_flatten(self.returns[batch_inds]),
                costs=self.pad_and_flatten(self.costs[batch_inds]),
                cost_values=self.pad_and_flatten(self.cost_values[batch_inds]),
                cost_advantages=self.pad_and_flatten(self.cost_advantages[batch_inds]),
                cost_returns=self.pad_and_flatten(self.cost_returns[batch_inds]),
                lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
                episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
                mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
            )


if RecurrentActorCriticPolicy is not None:

    class CostRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.cost_value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
            if self.ortho_init:
                self.cost_value_net.apply(self.init_weights)
            self.optimizer = self.optimizer_class(self.parameters(), lr=self.optimizer.param_groups[0]["lr"], **self.optimizer_kwargs)

        def forward(
            self,
            obs: th.Tensor,
            lstm_states: Any,
            episode_starts: th.Tensor,
            deterministic: bool = False,
        ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, Any]:
            features = self.extract_features(obs)
            if self.share_features_extractor:
                pi_features = vf_features = features
            else:
                pi_features, vf_features = features
            latent_pi, lstm_states_pi = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
            if self.lstm_critic is not None:
                latent_vf, lstm_states_vf = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
            elif self.shared_lstm:
                latent_vf = latent_pi.detach()
                lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
            else:
                latent_vf = self.critic(vf_features)
                lstm_states_vf = lstm_states_pi
            latent_pi = self.mlp_extractor.forward_actor(latent_pi)
            latent_vf = self.mlp_extractor.forward_critic(latent_vf)
            values = self.value_net(latent_vf)
            cost_values = self.cost_value_net(latent_vf)
            distribution = self._get_action_dist_from_latent(latent_pi)
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            actions = actions.reshape((-1, *self.action_space.shape))
            return actions, values, cost_values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

        def evaluate_actions(
            self,
            obs: th.Tensor,
            actions: th.Tensor,
            lstm_states: Any,
            episode_starts: th.Tensor,
        ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor | None]:
            features = self.extract_features(obs)
            if self.share_features_extractor:
                pi_features = vf_features = features
            else:
                pi_features, vf_features = features
            latent_pi, _ = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
            if self.lstm_critic is not None:
                latent_vf, _ = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
            elif self.shared_lstm:
                latent_vf = latent_pi.detach()
            else:
                latent_vf = self.critic(vf_features)
            latent_pi = self.mlp_extractor.forward_actor(latent_pi)
            latent_vf = self.mlp_extractor.forward_critic(latent_vf)
            distribution = self._get_action_dist_from_latent(latent_pi)
            log_prob = distribution.log_prob(actions)
            values = self.value_net(latent_vf)
            cost_values = self.cost_value_net(latent_vf)
            return values, cost_values, log_prob, distribution.entropy()

        def predict_values(
            self,
            obs: th.Tensor,
            lstm_states: tuple[th.Tensor, th.Tensor],
            episode_starts: th.Tensor,
        ) -> tuple[th.Tensor, th.Tensor]:
            features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)
            if self.lstm_critic is not None:
                latent_vf, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_critic)
            elif self.shared_lstm:
                latent_pi, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
                latent_vf = latent_pi.detach()
            else:
                latent_vf = self.critic(features)
            latent_vf = self.mlp_extractor.forward_critic(latent_vf)
            return self.value_net(latent_vf), self.cost_value_net(latent_vf)


if RecurrentPPO is not None:

    SelfRecurrentLagrangianPPO = TypeVar("SelfRecurrentLagrangianPPO", bound="RecurrentLagrangianPPO")

    class RecurrentLagrangianPPO(RecurrentPPO):
        policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
            "MlpLstmPolicy": CostRecurrentActorCriticPolicy,
        }

        def __init__(
            self,
            policy: str | type[RecurrentActorCriticPolicy],
            env: GymEnv | str,
            *args: Any,
            cost_key: str = "cost",
            cost_limit: float = 0.0,
            lambda_init: float = 1.0,
            lambda_lr: float = 0.05,
            lambda_max: float | None = None,
            cost_vf_coef: float = 0.7,
            use_cost_value_function: bool = True,
            lagrangian_update_frequency: str = "rollout",
            **kwargs: Any,
        ) -> None:
            self.cost_key = cost_key
            self.cost_vf_coef = cost_vf_coef
            self.use_cost_value_function = use_cost_value_function
            self.lagrangian_update_frequency = lagrangian_update_frequency
            self.lagrange = Lagrange(cost_limit, lambda_init, lambda_lr, lambda_max)
            self._last_mean_ep_cost = 0.0
            self._last_cost_components: dict[str, float] = {}
            super().__init__(policy, env, *args, **kwargs)

        def _setup_model(self) -> None:
            self._setup_lr_schedule()
            self.set_random_seed(self.seed)
            self.policy = self.policy_class(
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                use_sde=self.use_sde,
                **self.policy_kwargs,
            )
            self.policy = self.policy.to(self.device)
            if not isinstance(self.policy, RecurrentActorCriticPolicy):
                raise ValueError("Policy must subclass RecurrentActorCriticPolicy")
            lstm = self.policy.lstm_actor
            single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
            self._last_lstm_states = RNNStates(
                (th.zeros(single_hidden_state_shape, device=self.device), th.zeros(single_hidden_state_shape, device=self.device)),
                (th.zeros(single_hidden_state_shape, device=self.device), th.zeros(single_hidden_state_shape, device=self.device)),
            )
            hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            self.rollout_buffer = CostRecurrentRolloutBuffer(
                self.n_steps,
                self.observation_space,
                self.action_space,
                hidden_state_buffer_shape,
                self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
            )
            self.clip_range = FloatSchedule(self.clip_range)
            if self.clip_range_vf is not None:
                if isinstance(self.clip_range_vf, (float, int)):
                    assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"
                self.clip_range_vf = FloatSchedule(self.clip_range_vf)

        def _coerce_costs(self, infos: list[dict[str, Any]]) -> np.ndarray:
            return np.asarray([float(info.get(self.cost_key, 0.0)) for info in infos], dtype=np.float32)

        def _update_lagrange_from_rollout(self) -> None:
            self._last_mean_ep_cost = self.rollout_buffer.mean_episode_cost()
            self._last_cost_components = self.rollout_buffer.mean_cost_components()
            if self.lagrangian_update_frequency == "rollout":
                self.lagrange.update_lagrange_multiplier(self._last_mean_ep_cost)

        def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
        ) -> bool:
            assert isinstance(rollout_buffer, CostRecurrentRolloutBuffer)
            assert self._last_obs is not None, "No previous observation was provided"
            self.policy.set_training_mode(False)
            n_steps = 0
            rollout_buffer.reset()
            if self.use_sde:
                self.policy.reset_noise(env.num_envs)
            callback.on_rollout_start()
            lstm_states = deepcopy(self._last_lstm_states)

            while n_steps < n_rollout_steps:
                if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                    self.policy.reset_noise(env.num_envs)
                with th.no_grad():
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    episode_starts = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
                    actions, values, cost_values, log_probs, lstm_states = self.policy(obs_tensor, lstm_states, episode_starts)
                    if not self.use_cost_value_function:
                        cost_values = th.zeros_like(values)
                actions = actions.cpu().numpy()
                clipped_actions = actions
                if isinstance(self.action_space, spaces.Box):
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
                new_obs, rewards, dones, infos = env.step(clipped_actions)
                costs = self._coerce_costs(infos)
                cost_components = LagrangianPPO._coerce_cost_components(infos)
                self.num_timesteps += env.num_envs
                callback.update_locals(locals())
                if not callback.on_step():
                    return False
                self._update_info_buffer(infos, dones)
                n_steps += 1
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.reshape(-1, 1)
                for idx, done_ in enumerate(dones):
                    if (
                        done_
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                    ):
                        terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                        with th.no_grad():
                            terminal_lstm_state = (
                                lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                                lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                            )
                            episode_starts = th.tensor([False], dtype=th.float32, device=self.device)
                            terminal_value, terminal_cost_value = self.policy.predict_values(
                                terminal_obs, terminal_lstm_state, episode_starts
                            )
                        rewards[idx] += self.gamma * terminal_value[0]
                        if self.use_cost_value_function:
                            costs[idx] += float(self.gamma * terminal_cost_value[0].detach().cpu().item())
                rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    cost_values,
                    log_probs,
                    costs,
                    lstm_states=self._last_lstm_states,
                    dones=dones,
                    cost_components=cost_components,
                )
                self._last_obs = new_obs
                self._last_episode_starts = dones
                self._last_lstm_states = lstm_states

            with th.no_grad():
                episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
                values, cost_values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts)
                if not self.use_cost_value_function:
                    cost_values = th.zeros_like(values)
            rollout_buffer.compute_returns_and_advantage(values, cost_values, dones)
            self._update_lagrange_from_rollout()
            callback.on_rollout_end()
            return True

        def train(self) -> None:
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)
            clip_range = self.clip_range(self._current_progress_remaining)
            if self.clip_range_vf is not None:
                clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
            entropy_losses = []
            pg_losses, value_losses, cost_value_losses = [], [], []
            clip_fractions = []
            continue_training = True

            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        actions = rollout_data.actions.long().flatten()
                    mask = rollout_data.mask > 1e-8
                    values, cost_values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations,
                        actions,
                        rollout_data.lstm_states,
                        rollout_data.episode_starts,
                    )
                    values = values.flatten()
                    cost_values = cost_values.flatten()
                    advantages = rollout_data.advantages
                    cost_advantages = rollout_data.cost_advantages
                    if self.normalize_advantage:
                        advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)
                        cost_advantages = (cost_advantages - cost_advantages[mask].mean()) / (cost_advantages[mask].std() + 1e-8)
                    lagrangian_multiplier = self.lagrange.lagrangian_multiplier
                    combined_advantages = (advantages - lagrangian_multiplier * cost_advantages) / (1.0 + lagrangian_multiplier)
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    policy_loss_1 = combined_advantages * ratio
                    policy_loss_2 = combined_advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])
                    pg_losses.append(policy_loss.item())
                    clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item())
                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])
                    value_losses.append(value_loss.item())
                    if self.use_cost_value_function:
                        cost_value_loss = th.mean(((rollout_data.cost_returns - cost_values) ** 2)[mask])
                    else:
                        cost_value_loss = th.zeros((), device=self.device)
                    cost_value_losses.append(float(cost_value_loss.detach().cpu().item()))
                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob[mask])
                    else:
                        entropy_loss = -th.mean(entropy[mask])
                    entropy_losses.append(entropy_loss.item())
                    loss = (
                        policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                        + self.cost_vf_coef * cost_value_loss
                    )
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)
                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                self._n_updates += 1
                if not continue_training:
                    break

            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
            cost_explained_var = explained_variance(
                self.rollout_buffer.cost_values.flatten(),
                self.rollout_buffer.cost_returns.flatten(),
            )
            self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
            self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("train/cost_value_loss", np.mean(cost_value_losses))
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))
            self.logger.record("train/loss", loss.item())
            self.logger.record("train/explained_variance", explained_var)
            self.logger.record("train/cost_explained_variance", cost_explained_var)
            self.logger.record("train/lagrangian_multiplier", self.lagrange.lagrangian_multiplier)
            self.logger.record("train/cost_limit", self.lagrange.cost_limit)
            self.logger.record("train/mean_ep_cost", self._last_mean_ep_cost)
            self.logger.record("train/cost_violation", self.lagrange.last_cost_violation)
            for key, value in self._last_cost_components.items():
                self.logger.record(f"train/cost_component/{key}", value)
            if hasattr(self.policy, "log_std"):
                self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/clip_range", clip_range)
            if self.clip_range_vf is not None:
                self.logger.record("train/clip_range_vf", clip_range_vf)

        def learn(
            self: SelfRecurrentLagrangianPPO,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "RecurrentLagrangianPPO",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
        ) -> SelfRecurrentLagrangianPPO:
            return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)

    SelfRecurrentFOCOPS = TypeVar("SelfRecurrentFOCOPS", bound="RecurrentFOCOPS")

    class RecurrentFOCOPS(RecurrentLagrangianPPO):
        def __init__(
            self,
            policy: str | type[RecurrentActorCriticPolicy],
            env: GymEnv | str,
            *args: Any,
            focops_eta: float = 0.02,
            focops_lam: float = 1.5,
            **kwargs: Any,
        ) -> None:
            self.focops_eta = float(focops_eta)
            self.focops_lam = float(focops_lam)
            kwargs["lagrangian_update_frequency"] = "train"
            super().__init__(policy, env, *args, **kwargs)

        def _update_lagrange_from_rollout(self) -> None:
            self._last_mean_ep_cost = self.rollout_buffer.mean_episode_cost()
            self._last_cost_components = self.rollout_buffer.mean_cost_components()

        def train(self) -> None:
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)
            clip_range = self.clip_range(self._current_progress_remaining)
            if self.clip_range_vf is not None:
                clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

            self.lagrange.update_lagrange_multiplier(self._last_mean_ep_cost)

            focops_batches = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()
                with th.no_grad():
                    old_distribution, _ = self.policy.get_distribution(
                        rollout_data.observations,
                        rollout_data.lstm_states.pi,
                        rollout_data.episode_starts,
                    )
                    old_distribution = deepcopy(old_distribution)
                focops_batches.append((rollout_data, actions, old_distribution))

            entropy_losses = []
            pg_losses, value_losses, cost_value_losses = [], [], []
            focops_kls = []
            focops_ratios = []
            focops_gate_fractions = []
            continue_training = True
            focops_stop_iter = self.n_epochs

            for epoch in range(self.n_epochs):
                epoch_kls = []
                for rollout_data, actions, old_distribution in focops_batches:
                    mask = rollout_data.mask > 1e-8
                    values, cost_values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations,
                        actions,
                        rollout_data.lstm_states,
                        rollout_data.episode_starts,
                    )
                    distribution, _ = self.policy.get_distribution(
                        rollout_data.observations,
                        rollout_data.lstm_states.pi,
                        rollout_data.episode_starts,
                    )
                    values = values.flatten()
                    cost_values = cost_values.flatten()
                    advantages = rollout_data.advantages
                    cost_advantages = rollout_data.cost_advantages
                    if self.normalize_advantage:
                        advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)
                        cost_advantages = (cost_advantages - cost_advantages[mask].mean()) / (cost_advantages[mask].std() + 1e-8)

                    lagrangian_multiplier = self.lagrange.lagrangian_multiplier
                    combined_advantages = (advantages - lagrangian_multiplier * cost_advantages) / (1.0 + lagrangian_multiplier)
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    kl = _sb3_distribution_kl(distribution, old_distribution)
                    gate = kl.detach() <= self.focops_eta
                    effective_mask = mask & gate
                    if th.any(effective_mask):
                        policy_loss = th.mean(
                            (
                                kl
                                - (1.0 / self.focops_lam) * ratio * combined_advantages
                            )[effective_mask]
                        )
                    else:
                        policy_loss = th.zeros((), device=self.device)
                    pg_losses.append(float(policy_loss.detach().cpu().item()))
                    focops_ratios.append(float(th.mean(ratio[mask]).detach().cpu().item()))
                    focops_gate_fractions.append(float(th.mean(gate[mask].float()).detach().cpu().item()))

                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])
                    value_losses.append(value_loss.item())

                    if self.use_cost_value_function:
                        cost_value_loss = th.mean(((rollout_data.cost_returns - cost_values) ** 2)[mask])
                    else:
                        cost_value_loss = th.zeros((), device=self.device)
                    cost_value_losses.append(float(cost_value_loss.detach().cpu().item()))

                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob[mask])
                    else:
                        entropy_loss = -th.mean(entropy[mask])
                    entropy_losses.append(entropy_loss.item())

                    loss = (
                        policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                        + self.cost_vf_coef * cost_value_loss
                    )
                    mean_kl = float(th.mean(kl[mask]).detach().cpu().item())
                    epoch_kls.append(mean_kl)
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                mean_epoch_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
                focops_kls.append(mean_epoch_kl)
                self._n_updates += 1
                if self.target_kl is not None and mean_epoch_kl > self.target_kl:
                    continue_training = False
                    focops_stop_iter = epoch + 1
                    if self.verbose >= 1:
                        print(f"Early stopping FOCOPS at step {epoch + 1} due to reaching max kl: {mean_epoch_kl:.2f}")
                    break
                if not continue_training:
                    break

            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
            cost_explained_var = explained_variance(
                self.rollout_buffer.cost_values.flatten(),
                self.rollout_buffer.cost_returns.flatten(),
            )
            self._last_focops_metrics = {
                "train/focops_policy_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
                "train/focops_kl": float(np.mean(focops_kls)) if focops_kls else 0.0,
                "train/focops_stop_iter": focops_stop_iter,
                "train/focops_policy_ratio": float(np.mean(focops_ratios)) if focops_ratios else 0.0,
            }

            self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
            self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("train/cost_value_loss", np.mean(cost_value_losses))
            self.logger.record("train/focops_policy_loss", self._last_focops_metrics["train/focops_policy_loss"])
            self.logger.record("train/focops_kl", self._last_focops_metrics["train/focops_kl"])
            self.logger.record("train/focops_stop_iter", focops_stop_iter)
            self.logger.record("train/focops_policy_ratio", self._last_focops_metrics["train/focops_policy_ratio"])
            if focops_gate_fractions:
                self.logger.record("train/focops_gate_fraction", np.mean(focops_gate_fractions))
            self.logger.record("train/loss", loss.item())
            self.logger.record("train/explained_variance", explained_var)
            self.logger.record("train/cost_explained_variance", cost_explained_var)
            self.logger.record("train/lagrangian_multiplier", self.lagrange.lagrangian_multiplier)
            self.logger.record("train/cost_limit", self.lagrange.cost_limit)
            self.logger.record("train/mean_ep_cost", self._last_mean_ep_cost)
            self.logger.record("train/cost_violation", self.lagrange.last_cost_violation)
            self.logger.record("train/lambda_loss", self.lagrange.last_lambda_loss)
            for key, value in self._last_cost_components.items():
                self.logger.record(f"train/cost_component/{key}", value)
            if hasattr(self.policy, "log_std"):
                self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/clip_range", clip_range)
            if self.clip_range_vf is not None:
                self.logger.record("train/clip_range_vf", clip_range_vf)

        def learn(
            self: SelfRecurrentFOCOPS,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "RecurrentFOCOPS",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
        ) -> SelfRecurrentFOCOPS:
            return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)

    SelfRecurrentCUP = TypeVar("SelfRecurrentCUP", bound="RecurrentCUP")

    class RecurrentCUP(RecurrentLagrangianPPO):
        def __init__(
            self,
            policy: str | type[RecurrentActorCriticPolicy],
            env: GymEnv | str,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            gamma = kwargs.get("gamma", args[4] if len(args) >= 5 else 0.99)
            if float(gamma) >= 1.0:
                raise ValueError("RecurrentCUP requires agent.gamma < 1.0 because CUP divides by 1 - gamma.")
            kwargs["lagrangian_update_frequency"] = "train"
            super().__init__(policy, env, *args, **kwargs)

        def _update_lagrange_from_rollout(self) -> None:
            self._last_mean_ep_cost = self.rollout_buffer.mean_episode_cost()
            self._last_cost_components = self.rollout_buffer.mean_cost_components()

        def train(self) -> None:
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)
            clip_range = self.clip_range(self._current_progress_remaining)
            if self.clip_range_vf is not None:
                clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

            self.lagrange.update_lagrange_multiplier(self._last_mean_ep_cost)

            entropy_losses = []
            pg_losses, value_losses, cost_value_losses = [], [], []
            clip_fractions = []
            continue_training = True

            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        actions = rollout_data.actions.long().flatten()
                    mask = rollout_data.mask > 1e-8
                    values, cost_values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations,
                        actions,
                        rollout_data.lstm_states,
                        rollout_data.episode_starts,
                    )
                    values = values.flatten()
                    cost_values = cost_values.flatten()
                    advantages = rollout_data.advantages
                    if self.normalize_advantage:
                        advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])
                    pg_losses.append(policy_loss.item())
                    clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item())

                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])
                    value_losses.append(value_loss.item())

                    if self.use_cost_value_function:
                        cost_value_loss = th.mean(((rollout_data.cost_returns - cost_values) ** 2)[mask])
                    else:
                        cost_value_loss = th.zeros((), device=self.device)
                    cost_value_losses.append(float(cost_value_loss.detach().cpu().item()))

                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob[mask])
                    else:
                        entropy_loss = -th.mean(entropy[mask])
                    entropy_losses.append(entropy_loss.item())

                    loss = (
                        policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                        + self.cost_vf_coef * cost_value_loss
                    )
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)
                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                self._n_updates += 1
                if not continue_training:
                    break

            cup_batches = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()
                with th.no_grad():
                    old_distribution, _ = self.policy.get_distribution(
                        rollout_data.observations,
                        rollout_data.lstm_states.pi,
                        rollout_data.episode_starts,
                    )
                    old_distribution = deepcopy(old_distribution)
                cup_batches.append((rollout_data, actions, old_distribution))

            cup_cost_losses = []
            cup_kls = []
            cup_entropies = []
            cup_ratios = []
            cup_stop_iter = self.n_epochs
            cup_coef = (1.0 - self.gamma * self.gae_lambda) / (1.0 - self.gamma)
            for epoch in range(self.n_epochs):
                epoch_kls = []
                for rollout_data, actions, old_distribution in cup_batches:
                    mask = rollout_data.mask > 1e-8
                    cost_advantages = rollout_data.cost_advantages
                    if self.normalize_advantage:
                        cost_advantages = (
                            cost_advantages - cost_advantages[mask].mean()
                        ) / (cost_advantages[mask].std() + 1e-8)

                    distribution, _ = self.policy.get_distribution(
                        rollout_data.observations,
                        rollout_data.lstm_states.pi,
                        rollout_data.episode_starts,
                    )
                    log_prob = distribution.log_prob(actions)
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    kl = _sb3_distribution_kl(distribution, old_distribution)
                    loss_cost = th.mean(
                        (
                            self.lagrange.lagrangian_multiplier * cup_coef * ratio * cost_advantages
                            + kl
                        )[mask]
                    )

                    self.policy.optimizer.zero_grad()
                    loss_cost.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                    cup_cost_losses.append(float(loss_cost.detach().cpu().item()))
                    epoch_kls.append(float(th.mean(kl[mask]).detach().cpu().item()))
                    entropy = distribution.entropy()
                    if entropy is not None:
                        cup_entropies.append(float(th.mean(entropy[mask]).detach().cpu().item()))
                    cup_ratios.append(float(th.mean(ratio[mask]).detach().cpu().item()))

                mean_epoch_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
                cup_kls.append(mean_epoch_kl)
                if self.target_kl is not None and mean_epoch_kl > self.target_kl:
                    cup_stop_iter = epoch + 1
                    if self.verbose >= 1:
                        print(f"Early stopping CUP second step at step {epoch + 1} due to reaching max kl: {mean_epoch_kl:.2f}")
                    break

            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
            cost_explained_var = explained_variance(
                self.rollout_buffer.cost_values.flatten(),
                self.rollout_buffer.cost_returns.flatten(),
            )
            self._last_cup_metrics = {
                "train/cup_cost_loss": float(np.mean(cup_cost_losses)) if cup_cost_losses else 0.0,
                "train/cup_second_step_kl": float(np.mean(cup_kls)) if cup_kls else 0.0,
                "train/cup_second_step_stop_iter": cup_stop_iter,
            }

            self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
            self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("train/cost_value_loss", np.mean(cost_value_losses))
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))
            self.logger.record("train/loss", loss.item())
            self.logger.record("train/explained_variance", explained_var)
            self.logger.record("train/cost_explained_variance", cost_explained_var)
            self.logger.record("train/lagrangian_multiplier", self.lagrange.lagrangian_multiplier)
            self.logger.record("train/cost_limit", self.lagrange.cost_limit)
            self.logger.record("train/mean_ep_cost", self._last_mean_ep_cost)
            self.logger.record("train/cost_violation", self.lagrange.last_cost_violation)
            self.logger.record("train/lambda_loss", self.lagrange.last_lambda_loss)
            self.logger.record("train/cup_cost_loss", self._last_cup_metrics["train/cup_cost_loss"])
            self.logger.record("train/cup_second_step_kl", self._last_cup_metrics["train/cup_second_step_kl"])
            self.logger.record("train/cup_second_step_stop_iter", cup_stop_iter)
            if cup_entropies:
                self.logger.record("train/cup_second_step_entropy", np.mean(cup_entropies))
            if cup_ratios:
                self.logger.record("train/cup_second_step_policy_ratio", np.mean(cup_ratios))
            for key, value in self._last_cost_components.items():
                self.logger.record(f"train/cost_component/{key}", value)
            if hasattr(self.policy, "log_std"):
                self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/clip_range", clip_range)
            if self.clip_range_vf is not None:
                self.logger.record("train/clip_range_vf", clip_range_vf)

        def learn(
            self: SelfRecurrentCUP,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "RecurrentCUP",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
        ) -> SelfRecurrentCUP:
            return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)

else:
    RecurrentLagrangianPPO = None
    RecurrentFOCOPS = None
    RecurrentCUP = None


def fertilization_action_constraint(*args: Any, **kwargs: Any) -> tuple[float, float, float, float, float]:
    warnings.warn(
        "fertilization_action_constraint is deprecated. Use ConstraintCostWrapper to expose info['cost'].",
        DeprecationWarning,
        stacklevel=2,
    )
    return 0.0, 0.0, 0.0, 0.0, 0.0


RolloutBufferSteps = CostRolloutBuffer
RolloutBufferNitrogen = CostRolloutBuffer
RolloutBufferSamplesStep = CostRolloutBufferSamples
