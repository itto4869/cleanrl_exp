from itertools import chain
from typing import Optional

import torch

from experiments.soap.soap import SOAP


class SOAPRollout(SOAP):
    """
    SOAP variant that refreshes the preconditioner from rollout gradients
    and keeps it fixed during optimizer updates.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 1,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        trace_normalize: bool = True,
        trace_normalize_mode: str = "trace",
        update_clip_norm: Optional[float] = 1.0,
        data_format: str = "channels_first",
        correct_bias: bool = True,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            merge_dims=merge_dims,
            precondition_1d=precondition_1d,
            normalize_grads=normalize_grads,
            data_format=data_format,
            correct_bias=correct_bias,
        )
        self.rollout_step = 0
        self.trace_normalize = trace_normalize
        if trace_normalize_mode not in ("trace", "mean"):
            raise ValueError("trace_normalize_mode must be 'trace' or 'mean'")
        self.trace_normalize_mode = trace_normalize_mode
        self.update_clip_norm = update_clip_norm

    @torch.no_grad()
    def update_preconditioner_from_grads(self):
        """
        Recompute the preconditioner from the current gradients.
        Call this once per rollout with full-batch gradients.
        """
        self.rollout_step += 1
        for group in self.param_groups:
            if self.rollout_step % group["precondition_frequency"] != 0:
                continue
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                if state.get("Q") is not None:
                    state["exp_avg"] = self.project_back(
                        state["exp_avg"],
                        state,
                        merge_dims=group["merge_dims"],
                        max_precond_dim=group["max_precond_dim"],
                    )

                if state.get("GG") is None or state.get("Q") is None:
                    self.init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=group["precondition_frequency"],
                        precondition_1d=group["precondition_1d"],
                        shampoo_beta=(group["shampoo_beta"] if group["shampoo_beta"] >= 0 else group["betas"][1]),
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                    )

                self.update_preconditioner(
                    grad,
                    state,
                    max_precond_dim=group["max_precond_dim"],
                    merge_dims=group["merge_dims"],
                    precondition_1d=group["precondition_1d"],
                    precondition_step=group["precondition_frequency"],
                )

    def update_preconditioner(
        self,
        grad,
        state,
        max_precond_dim=10000,
        merge_dims=False,
        precondition_1d=False,
        precondition_step=None,
    ):
        """
        Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
        Overridden to include trace normalization.
        """
        if precondition_step is None:
            precondition_step = state.get("step", 0)
        if isinstance(precondition_step, torch.Tensor):
            precondition_step = int(precondition_step.item())

        if state["Q"] is not None:
            state["exp_avg"] = self.project_back(
                state["exp_avg"],
                state,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
            )
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state["GG"][0].lerp_(
                    grad.unsqueeze(1) @ grad.unsqueeze(0), 1 - state["shampoo_beta"]
                )
        else:
            if merge_dims:
                new_grad = self.merge_dims(grad, max_precond_dim)
                for idx, sh in enumerate(new_grad.shape):
                    if sh <= max_precond_dim:
                        outer_product = torch.tensordot(
                            new_grad,
                            new_grad,
                            dims=[
                                [
                                    *chain(
                                        range(idx), range(idx + 1, len(new_grad.shape))
                                    )
                                ]
                            ]
                            * 2,
                        )
                        state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])
            else:
                for idx, sh in enumerate(grad.shape):
                    if sh <= max_precond_dim:
                        outer_product = torch.tensordot(
                            grad,
                            grad,
                            # Contracts across all dimensions except for k.
                            dims=[
                                [*chain(range(idx), range(idx + 1, len(grad.shape)))]
                            ]
                            * 2,
                        )
                        state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])

        # --- Trace Normalization ---
        if self.trace_normalize:
            for mat in state["GG"]:
                if len(mat) > 0:
                    trace = torch.trace(mat)
                    if trace > 1e-12:
                        denom = trace
                        if self.trace_normalize_mode == "mean":
                            denom = trace / mat.shape[0]
                        mat.div_(denom)
        # ---------------------------

        if state["Q"] is None:
            state["Q"] = self.get_orthogonal_matrix(state["GG"])
        if (
            precondition_step > 0
            and precondition_step % state["precondition_frequency"] == 0
        ):
            state["Q"] = self.get_orthogonal_matrix_QR(
                state, max_precond_dim, merge_dims
            )
            # state['Q'] = self.get_fast_QR(state, max_precond_dim, merge_dims)

        if precondition_step > 0:
            state["exp_avg"] = self.project(
                state["exp_avg"],
                state,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
            )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                if state.get("Q") is None:
                    raise RuntimeError(
                        "Preconditioner is not initialized. "
                        "Call update_preconditioner_from_grads() after a rollout."
                    )

                grad_projected = self.project(
                    grad,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                step_for_bias = state["step"]
                if isinstance(step_for_bias, torch.Tensor):
                    step_for_bias = step_for_bias.item()

                exp_avg.mul_(beta1).add_(grad_projected, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=(1.0 - beta2))

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                exp_avg_projected = exp_avg

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** (step_for_bias)
                    bias_correction2 = 1.0 - beta2 ** (step_for_bias)
                    step_size = step_size * (bias_correction2**0.5) / bias_correction1

                norm_grad = self.project_back(
                    exp_avg_projected / denom,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )


                if group["normalize_grads"]:
                    norm_grad = norm_grad / (1e-6 + torch.mean(norm_grad**2) ** 0.5)
                    
                # inside SOAPRollout.step(), after norm_grad computed
                state["last_update_norm"] = norm_grad.norm().detach()
                state["last_update_rms"]  = (norm_grad.pow(2).mean().sqrt()).detach()
                
                # --- Update Clipping (added) ---
                update = -step_size * norm_grad
                if self.update_clip_norm is not None and self.update_clip_norm > 0:
                    update_norm = update.norm()
                    if update_norm > self.update_clip_norm:
                        update = update * (self.update_clip_norm / update_norm)
                p.add_(update)
                # -------------------------------

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


class SOAPRolloutMix(SOAPRollout):
    """
    SOAP rollout variant that mixes rollout preconditioner statistics (L/R)
    with a persistent EMA of per-step preconditioner statistics.
    """

    def __init__(
        self,
        *args,
        grad_mix_ratio: float = 0.5,
        mix_normalize_mode: str = "none",
        reset_rollout_stats: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= grad_mix_ratio <= 1.0:
            raise ValueError("grad_mix_ratio must be in [0, 1]")
        if mix_normalize_mode not in ("none", "rollout", "both"):
            raise ValueError("mix_normalize_mode must be 'none', 'rollout', or 'both'")
        self.grad_mix_ratio = grad_mix_ratio
        self.mix_normalize_mode = mix_normalize_mode
        self.reset_rollout_stats = reset_rollout_stats

    @torch.no_grad()
    def _init_preconditioner_stats(self, grad, group):
        gg = []
        max_precond_dim = group["max_precond_dim"]
        if grad.dim() == 1:
            if not group["precondition_1d"] or grad.shape[0] > max_precond_dim:
                gg.append([])
            else:
                gg.append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device))
        else:
            if group["merge_dims"]:
                grad = self.merge_dims(grad, max_precond_dim)
            for sh in grad.shape:
                if sh > max_precond_dim:
                    gg.append([])
                else:
                    gg.append(torch.zeros(sh, sh, device=grad.device))
        return gg

    def _clone_preconditioner_stats(self, gg):
        gg_clone = []
        for mat in gg:
            if len(mat) == 0:
                gg_clone.append([])
            else:
                gg_clone.append(mat.clone())
        return gg_clone

    def _trace_normalize_stats(self, gg):
        for mat in gg:
            if len(mat) == 0:
                continue
            trace = torch.trace(mat)
            if trace <= 1e-12:
                continue
            denom = trace
            if self.trace_normalize_mode == "mean":
                denom = trace / mat.shape[0]
            mat.div_(denom)

    @torch.no_grad()
    def _update_preconditioner_stats(self, grad, gg, group, beta):
        max_precond_dim = group["max_precond_dim"]
        if grad.dim() == 1:
            if (
                group["precondition_1d"]
                and grad.shape[0] <= max_precond_dim
                and len(gg) > 0
                and len(gg[0]) > 0
            ):
                gg[0].lerp_(grad.unsqueeze(1) @ grad.unsqueeze(0), 1 - beta)
            return

        if group["merge_dims"]:
            grad = self.merge_dims(grad, max_precond_dim)
        for idx, sh in enumerate(grad.shape):
            if sh > max_precond_dim:
                continue
            if len(gg[idx]) == 0:
                continue
            outer_product = torch.tensordot(
                grad,
                grad,
                dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
            )
            gg[idx].lerp_(outer_product, 1 - beta)

    @torch.no_grad()
    def update_preconditioner(
        self,
        grad,
        state,
        max_precond_dim=10000,
        merge_dims=False,
        precondition_1d=False,
        precondition_step=None,
    ):
        if precondition_step is None:
            precondition_step = state.get("step", 0)
        if isinstance(precondition_step, torch.Tensor):
            precondition_step = int(precondition_step.item())

        if state["Q"] is not None:
            state["exp_avg"] = self.project_back(
                state["exp_avg"],
                state,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
            )
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state["GG"][0].lerp_(
                    grad.unsqueeze(1) @ grad.unsqueeze(0), 1 - state["shampoo_beta"]
                )
        else:
            if merge_dims:
                new_grad = self.merge_dims(grad, max_precond_dim)
                for idx, sh in enumerate(new_grad.shape):
                    if sh <= max_precond_dim:
                        outer_product = torch.tensordot(
                            new_grad,
                            new_grad,
                            dims=[
                                [
                                    *chain(
                                        range(idx), range(idx + 1, len(new_grad.shape))
                                    )
                                ]
                            ]
                            * 2,
                        )
                        state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])
            else:
                for idx, sh in enumerate(grad.shape):
                    if sh <= max_precond_dim:
                        outer_product = torch.tensordot(
                            grad,
                            grad,
                            # Contracts across all dimensions except for k.
                            dims=[
                                [*chain(range(idx), range(idx + 1, len(grad.shape)))]
                            ]
                            * 2,
                        )
                        state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])

        gg_for_computation = self._clone_preconditioner_stats(state["GG"])
        gg_ema_for_mix = state.get("GG_ema")
        if self.mix_normalize_mode in ("rollout", "both"):
            self._trace_normalize_stats(gg_for_computation)
        if gg_ema_for_mix is not None and self.mix_normalize_mode == "both":
            gg_ema_for_mix = self._clone_preconditioner_stats(gg_ema_for_mix)
            self._trace_normalize_stats(gg_ema_for_mix)

        if self.grad_mix_ratio > 0 and gg_ema_for_mix is not None:
            for gg_calc, gg_ema in zip(gg_for_computation, gg_ema_for_mix):
                if len(gg_calc) == 0 or len(gg_ema) == 0:
                    continue
                if gg_calc.shape != gg_ema.shape:
                    continue
                gg_calc.mul_(1.0 - self.grad_mix_ratio).add_(gg_ema, alpha=self.grad_mix_ratio)

        # --- Trace Normalization ---
        if self.trace_normalize:
            self._trace_normalize_stats(gg_for_computation)
        # ---------------------------

        if state["Q"] is None:
            state["Q"] = self.get_orthogonal_matrix(gg_for_computation)
        if (
            precondition_step > 0
            and precondition_step % state["precondition_frequency"] == 0
        ):
            original_gg = state["GG"]
            state["GG"] = gg_for_computation
            state["Q"] = self.get_orthogonal_matrix_QR(
                state, max_precond_dim, merge_dims
            )
            state["GG"] = original_gg
            # state['Q'] = self.get_fast_QR(state, max_precond_dim, merge_dims)

        if precondition_step > 0:
            state["exp_avg"] = self.project(
                state["exp_avg"],
                state,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
            )

    @torch.no_grad()
    def update_preconditioner_from_grads(self):
        super().update_preconditioner_from_grads()
        if self.reset_rollout_stats:
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state.get(p)
                    if not state:
                        continue
                    if "exp_avg" in state:
                        state["exp_avg"].zero_()
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"].zero_()
                    if "step" in state:
                        state["step"] = 0

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta = group["shampoo_beta"] if group["shampoo_beta"] >= 0 else group["betas"][1]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "GG_ema" not in state:
                    state["GG_ema"] = self._init_preconditioner_stats(p.grad, group)
                self._update_preconditioner_stats(p.grad, state["GG_ema"], group, beta)
        return super().step(closure=closure)


class SOAPRolloutResetStats(SOAPRollout):
    """
    SOAP rollout variant that resets optimizer statistics each rollout
    while keeping the preconditioner.
    """

    @torch.no_grad()
    def reset_rollout_stats(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state:
                    continue
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()
                if "step" in state:
                    state["step"] = 0

    @torch.no_grad()
    def update_preconditioner_from_grads(self):
        super().update_preconditioner_from_grads()
        self.reset_rollout_stats()
