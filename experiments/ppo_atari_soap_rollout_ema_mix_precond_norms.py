import os
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro

from experiments.ppo_atari_soap_rollout_ema_mix import Agent, make_env
from experiments.soap.soap_rollout import SOAPRolloutMix


@dataclass
class Args:
    env_id: str = "BreakoutNoFrameskip-v4"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    total_rollouts: int = 3

    # rollout params
    num_envs: int = 8
    num_steps: int = 128

    # PPO params
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 1
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # SOAP params
    soap_precondition_frequency: int = 1
    soap_normalize_grads: bool = False
    soap_trace_normalize: bool = True
    soap_trace_normalize_mode: str = "trace"
    soap_grad_mix_ratio: float = 0.5
    soap_mix_normalize_mode: str = "none"
    soap_update_clip_norm: float = 1.0

    # runtime filled
    batch_size: int = 0
    minibatch_size: int = 0


def set_seed(seed: int, torch_deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def grad_global_norm(params) -> float:
    total_sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        total_sq += p.grad.data.pow(2).sum().item()
    return total_sq**0.5


def gg_list_norm(gg) -> float:
    if gg is None:
        return 0.0
    total_sq = 0.0
    for mat in gg:
        if isinstance(mat, list) or mat is None:
            continue
        if isinstance(mat, torch.Tensor):
            if mat.numel() == 0:
                continue
            total_sq += mat.pow(2).sum().item()
        else:
            total_sq += torch.as_tensor(mat).pow(2).sum().item()
    return total_sq**0.5


def compute_precond_norms(optimizer: SOAPRolloutMix):
    rollout_sq = 0.0
    ema_sq = 0.0
    mixed_sq = 0.0
    ema_found = False

    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if not state or "GG" not in state:
                continue

            gg_rollout = optimizer._clone_preconditioner_stats(state["GG"])
            gg_rollout_for_mix = optimizer._clone_preconditioner_stats(gg_rollout)

            if optimizer.mix_normalize_mode in ("rollout", "both"):
                optimizer._trace_normalize_stats(gg_rollout_for_mix)

            rollout_sq += gg_list_norm(gg_rollout_for_mix) ** 2

            gg_ema = state.get("GG_ema")
            if gg_ema is not None:
                ema_found = True
                ema_sq += gg_list_norm(gg_ema) ** 2

            gg_ema_for_mix = gg_ema
            if gg_ema is not None and optimizer.mix_normalize_mode == "both":
                gg_ema_for_mix = optimizer._clone_preconditioner_stats(gg_ema)
                optimizer._trace_normalize_stats(gg_ema_for_mix)

            gg_mixed = optimizer._clone_preconditioner_stats(gg_rollout_for_mix)
            if optimizer.grad_mix_ratio > 0 and gg_ema_for_mix is not None:
                for gg_calc, gg_ema_mat in zip(gg_mixed, gg_ema_for_mix):
                    if isinstance(gg_calc, list) or isinstance(gg_ema_mat, list):
                        continue
                    if not torch.is_tensor(gg_calc) or not torch.is_tensor(gg_ema_mat):
                        continue
                    if gg_calc.numel() == 0 or gg_ema_mat.numel() == 0:
                        continue
                    if gg_calc.shape != gg_ema_mat.shape:
                        continue
                    gg_calc.mul_(1.0 - optimizer.grad_mix_ratio).add_(
                        gg_ema_mat, alpha=optimizer.grad_mix_ratio
                    )

            if optimizer.trace_normalize:
                optimizer._trace_normalize_stats(gg_mixed)

            mixed_sq += gg_list_norm(gg_mixed) ** 2

    rollout_norm = rollout_sq**0.5
    ema_norm = (ema_sq**0.5) if ema_found else float("nan")
    mixed_norm = mixed_sq**0.5
    return rollout_norm, ema_norm, mixed_norm


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    set_seed(args.seed, args.torch_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, False, os.devnull) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    agent = Agent(envs).to(device)
    optimizer = SOAPRolloutMix(
        params=agent.parameters(),
        lr=args.learning_rate,
        normalize_grads=args.soap_normalize_grads,
        precondition_frequency=args.soap_precondition_frequency,
        trace_normalize=args.soap_trace_normalize,
        trace_normalize_mode=args.soap_trace_normalize_mode,
        grad_mix_ratio=args.soap_grad_mix_ratio,
        mix_normalize_mode=args.soap_mix_normalize_mode,
        update_clip_norm=args.soap_update_clip_norm,
    )
    named_params = list(agent.named_parameters())
    param_list = [p for _, p in named_params]

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for rollout_idx in range(1, args.total_rollouts + 1):
        for step in range(args.num_steps):
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, _ = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        optimizer.zero_grad()
        _, full_logprob, full_entropy, full_value = agent.get_action_and_value(b_obs, b_actions.long())
        full_logratio = full_logprob - b_logprobs
        full_ratio = full_logratio.exp()
        full_advantages = b_advantages
        if args.norm_adv:
            full_advantages = (full_advantages - full_advantages.mean()) / (full_advantages.std() + 1e-8)

        full_pg_loss1 = -full_advantages * full_ratio
        full_pg_loss2 = -full_advantages * torch.clamp(full_ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        full_pg_loss = torch.max(full_pg_loss1, full_pg_loss2).mean()

        full_value = full_value.view(-1)
        if args.clip_vloss:
            full_v_loss_unclipped = (full_value - b_returns) ** 2
            full_v_clipped = b_values + torch.clamp(
                full_value - b_values,
                -args.clip_coef,
                args.clip_coef,
            )
            full_v_loss_clipped = (full_v_clipped - b_returns) ** 2
            full_v_loss_max = torch.max(full_v_loss_unclipped, full_v_loss_clipped)
            full_v_loss = 0.5 * full_v_loss_max.mean()
        else:
            full_v_loss = 0.5 * ((full_value - b_returns) ** 2).mean()

        full_entropy_loss = full_entropy.mean()
        full_loss = full_pg_loss - args.ent_coef * full_entropy_loss + full_v_loss * args.vf_coef
        full_loss.backward()
        if args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        grad_norm = grad_global_norm(agent.parameters())
        optimizer.update_preconditioner_from_grads()

        rollout_norm, ema_norm, mixed_norm = compute_precond_norms(optimizer)

        optimizer.zero_grad()

        b_inds = np.arange(args.batch_size)
        update_norms = []
        per_layer_updates = {name: [] for name, _ in named_params}
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                with torch.no_grad():
                    params_before = [p.detach().clone() for p in param_list]
                optimizer.step()
                with torch.no_grad():
                    delta_sq = 0.0
                    for (name, p1), p0 in zip(named_params, params_before):
                        d = p1.detach() - p0
                        per_layer_updates[name].append(float(d.pow(2).sum().sqrt().item()))
                        delta_sq += d.pow(2).sum().item()
                    update_norms.append(delta_sq**0.5)

        update_norm_mean = float(np.mean(update_norms)) if update_norms else float("nan")
        update_norm_max = float(np.max(update_norms)) if update_norms else float("nan")
        print(
            f"[rollout {rollout_idx}] "
            f"grad_norm_used={grad_norm:.6g} "
            f"rollout_precond_norm={rollout_norm:.6g} "
            f"ema_precond_norm={ema_norm:.6g} "
            f"mixed_precond_norm={mixed_norm:.6g} "
            f"update_norm_mean={update_norm_mean:.6g} "
            f"update_norm_max={update_norm_max:.6g}"
        )
        for name in sorted(per_layer_updates.keys()):
            values = per_layer_updates[name]
            layer_mean = float(np.mean(values)) if values else float("nan")
            layer_max = float(np.max(values)) if values else float("nan")
            print(f"  [layer] {name} update_norm_mean={layer_mean:.6g} update_norm_max={layer_max:.6g}")

    envs.close()
