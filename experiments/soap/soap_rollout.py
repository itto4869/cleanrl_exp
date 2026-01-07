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

                p.add_(norm_grad, alpha=-step_size)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


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
