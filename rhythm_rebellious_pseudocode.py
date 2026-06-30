import torch


def compute_rhythm_gate(rhythm_token_weights: "T[B,T]",   # 来自 compute_temporal_rhythm_multipliers
                        response_mask: "T[B,T]",
                        multiplier_max: float,            # cfg.rhythm_aux_multiplier_max
                        gate_floor: float = 0.0,
                        gate_power: float = 1.0) -> "T[B,T]":
    """把 Rhythm token 权重转成 [0,1] 门控：>1 的节奏权重 → 门开。"""
    valid = response_mask > 0
    denom = max(multiplier_max - 1.0, 1e-6)
    gate = ((rhythm_token_weights - 1.0) / denom).clamp(min=0.0, max=1.0)
    if gate_power != 1.0:
        gate = gate.pow(gate_power)
    if gate_floor > 0.0:
        gate = gate_floor + (1.0 - gate_floor) * gate
    gate = torch.where(valid, gate, torch.zeros_like(gate))
    return gate


def compute_rebellious_token_weights(
        log_prob: "T[B,T]",                # 学生当前 token log-prob
        teacher_log_prob: "T[B,T]",        # 自教师 token log-prob
        response_mask: "T[B,T]",
        rebellious_sample_weight: "T[B]",  # 来自 pass-rate 软路由（机制 3）
        rebellious_rhythm_gate: "Optional[T[B,T]]",  # 来自上面 compute_rhythm_gate
        cfg) -> "Optional[T[B,T]]":
    """在 GRPO 分支上：学生 > 教师 的 token 给叛逆 bonus，并被 Rhythm 门控。"""
    if rebellious_sample_weight is None or not cfg.problem_pass_rate_rebellious_enabled:
        return None

    margin     = float(cfg.problem_pass_rate_rebellious_margin)    # 默认 0.2
    gamma      = float(cfg.problem_pass_rate_rebellious_gamma)     # 默认 1.0
    max_bonus  = float(cfg.problem_pass_rate_rebellious_max_bonus) # 默认 2.0

    with torch.no_grad():
        # 学生超过教师多少（带 margin）
        delta = (log_prob.detach() - teacher_log_prob.detach() - margin).clamp(min=0.0)  # [B,T]
        bonus = torch.exp(gamma * delta)                              # ≥1
        bonus = bonus.clamp(min=1.0, max=max_bonus)                  # 封顶
        bonus_delta = bonus - 1.0                                     # 增量

        # 用 Rhythm 门控（未启用时 gate=1）
        if rebellious_rhythm_gate is None:
            gate = torch.ones_like(bonus_delta)
        else:
            gate = rebellious_rhythm_gate
        gated_delta = bonus_delta * gate                              # 只在节奏强处放大

        # 组装 token 权重：1 + 样本系数 × 门控增量
        w = 1.0 + rebellious_sample_weight.unsqueeze(1) * gated_delta
        token_weights = torch.where(response_mask > 0, w, torch.ones_like(w))
    return token_weights


# ── 注入 GRPO 损失 ──
# rebellious_token_weights 作为 policy_token_weights 传给 vanilla 策略损失，
# 效果等价于：每个 token 的损失 × token_weights
rebellious_token_weights = compute_rebellious_token_weights(
    log_prob, teacher_log_prob, response_mask,
    rebellious_sample_weight, rebellious_rhythm_gate, cfg)

grpo_loss = vanilla_policy_loss(
    old_log_prob=old_log_prob,
    log_prob=log_prob,
    advantages=advantages,
    response_mask=grpo_route_mask,
    policy_token_weights=rebellious_token_weights,   # ← bonus 在此生效
)