# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from soap.soap_rollout import SOAPRolloutMix


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    soap_precondition_frequency: int = 1
    """how often to recompute the preconditioner (in rollouts)"""
    soap_normalize_grads: bool = False
    """whether to normalize preconditioned gradients per layer"""
    soap_trace_normalize: bool = True
    """whether to trace-normalize the preconditioner matrices"""
    soap_trace_normalize_mode: str = "trace"
    """trace normalization mode: 'trace' or 'mean' (mean eigenvalue)"""
    soap_grad_mix_ratio: float = 0.5
    """mix ratio for EMA L/R stats when building the preconditioner (0=rollout, 1=ema)"""
    soap_update_clip_norm: float = 1.0
    """clip L2 norm for each parameter update; set <= 0 to disable"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    warmup_steps: int = 0
    """number of environment steps to linearly warm up the learning rate"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, video_dir):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, video_dir)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{args.soap_precondition_frequency}"
    run_group = os.environ.get("RUN_GROUP", args.exp_name)
    run_dir = os.path.join("runs", run_group, run_name)
    video_dir = os.path.join("videos", run_group, run_name)
    if args.capture_video:
        os.makedirs(video_dir, exist_ok=True)
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, video_dir) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = SOAPRolloutMix(
        params=agent.parameters(),
        lr=args.learning_rate,
        normalize_grads=args.soap_normalize_grads,
        precondition_frequency=args.soap_precondition_frequency,
        trace_normalize=args.soap_trace_normalize,
        trace_normalize_mode=args.soap_trace_normalize_mode,
        grad_mix_ratio=args.soap_grad_mix_ratio,
        update_clip_norm=args.soap_update_clip_norm,
    )

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Warmup + annealing schedule (step-based).
        progress = min(global_step, args.total_timesteps)
        if args.warmup_steps > 0 and progress < args.warmup_steps:
            lrnow = args.learning_rate * (progress / args.warmup_steps)
        else:
            lrnow = args.learning_rate
            if args.anneal_lr:
                remaining = max(args.total_timesteps - args.warmup_steps, 1)
                lrnow = lrnow * (1.0 - (progress - args.warmup_steps) / remaining)
        optimizer.param_groups[0]["lr"] = max(lrnow, 0.0)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # --- extra rollout diagnostics ---
            writer.add_scalar("rollout/terminations_rate", np.mean(terminations), global_step)
            writer.add_scalar("rollout/truncations_rate", np.mean(truncations), global_step)
            writer.add_scalar("rollout/done_rate", np.mean(np.logical_or(terminations, truncations)), global_step)

            writer.add_scalar("rollout/reward_mean", float(np.mean(reward)), global_step)
            writer.add_scalar("rollout/reward_std", float(np.std(reward)), global_step)
            writer.add_scalar("rollout/reward_nonzero_rate", float(np.mean(np.array(reward) != 0)), global_step)

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
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
        
        # --- target diagnostics ---
        td_delta = returns - values  # not exactly delta, but useful scale check
        writer.add_scalar("targets/adv_mean", advantages.mean().item(), global_step)
        writer.add_scalar("targets/adv_std", advantages.std().item(), global_step)
        writer.add_scalar("targets/adv_absmax", advantages.abs().max().item(), global_step)

        writer.add_scalar("targets/ret_mean", returns.mean().item(), global_step)
        writer.add_scalar("targets/ret_std", returns.std().item(), global_step)
        writer.add_scalar("targets/ret_absmax", returns.abs().max().item(), global_step)

        writer.add_scalar("targets/val_pred_mean_old", values.mean().item(), global_step)
        writer.add_scalar("targets/val_pred_absmax_old", values.abs().max().item(), global_step)

        writer.add_scalar("targets/td_like_mean", td_delta.mean().item(), global_step)
        writer.add_scalar("targets/td_like_std", td_delta.std().item(), global_step)
        writer.add_scalar("targets/td_like_absmax", td_delta.abs().max().item(), global_step)


        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Update the preconditioner from the full rollout batch.
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
        optimizer.update_preconditioner_from_grads()
        optimizer.zero_grad()

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            stop_update = False
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
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
                
                writer.add_scalar("sanity/loss_isfinite", float(torch.isfinite(loss).all().item()), global_step)
                writer.add_scalar("sanity/value_isfinite", float(torch.isfinite(newvalue).all().item()), global_step)
                writer.add_scalar("sanity/logprob_isfinite", float(torch.isfinite(newlogprob).all().item()), global_step)


                optimizer.zero_grad()
                loss.backward()
                # --- grad norm (pre-clip) ---
                total_norm_sq = 0.0
                for p in agent.parameters():
                    if p.grad is None:
                        continue
                    param_norm = p.grad.data.norm(2)
                    total_norm_sq += param_norm.item() ** 2
                grad_norm_preclip = total_norm_sq ** 0.5
                writer.add_scalar("update/grad_norm_preclip", grad_norm_preclip, global_step)

                writer.add_scalar("update/grad_norm_postclip", min(grad_norm_preclip, args.max_grad_norm), global_step)

                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                
                # --- param delta norm ---
                with torch.no_grad():
                    params_before = [p.detach().clone() for p in agent.parameters()]
                optimizer.step()
                with torch.no_grad():
                    delta_sq = 0.0
                    for p0, p1 in zip(params_before, agent.parameters()):
                        d = (p1.detach() - p0)
                        delta_sq += d.pow(2).sum().item()
                    writer.add_scalar("update/delta_param_l2", delta_sq ** 0.5, global_step)

                if args.target_kl is not None and approx_kl > args.target_kl:
                    stop_update = True
                    break
                
                # after optimizer.step()
                p0 = next(iter(optimizer.param_groups[0]["params"]))
                st = optimizer.state[p0]
                if "last_update_norm" in st:
                    writer.add_scalar("update/update_norm_postprecond", st["last_update_norm"].item(), global_step)
                    writer.add_scalar("update/update_rms_postprecond",  st["last_update_rms"].item(), global_step)


            if stop_update:
                break
        
        # explained variance (old: from rollout-time values)
        y_true = b_returns.cpu().numpy()
        y_pred_old = b_values.cpu().numpy()
        var_y = np.var(y_true)
        explained_var_old = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred_old) / var_y
        writer.add_scalar("losses/explained_variance_old", explained_var_old, global_step)

        # explained variance (new: after updates)
        with torch.no_grad():
            v_pred_new = agent.get_value(b_obs).view(-1)
        y_pred_new = v_pred_new.cpu().numpy()
        explained_var_new = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred_new) / var_y
        writer.add_scalar("losses/explained_variance_new", explained_var_new, global_step)

        writer.add_scalar("critic/v_pred_absmax_new", float(np.max(np.abs(y_pred_new))), global_step)
        writer.add_scalar("critic/v_pred_mean_new", float(np.mean(y_pred_new)), global_step)


        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
