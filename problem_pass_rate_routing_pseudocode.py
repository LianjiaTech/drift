import math
import torch

# ===== 全局状态 =====
# router_state: dict[uid, {"success": float, "failure": float, "last_step": float}]
router_state: dict = {}


def maybe_periodic_reset(cfg, step: int) -> None:
    """周期性全量重置，防止状态僵化。"""
    if not (cfg.problem_pass_rate_routing_enabled
            and cfg.problem_pass_rate_periodic_reset_enabled):
        return
    transition_end = cfg.sample_routing_warmup_steps + cfg.sample_routing_transition_steps
    if step <= transition_end:
        return
    if (step - transition_end) % cfg.problem_pass_rate_periodic_reset_steps == 0:
        router_state.clear()


def compute_soft_routing_weights(
        correctness_mask: "T[B]",            # 0/1 答对
        uids: list,                          # 每样本的题目 id
        teacher_available_mask: "T[B]",      # 是否有教师示范
        cfg, step: int) -> tuple["T[B]", "T[B]", "T[B]", dict]:
    """返回 (sdpo_weight, grpo_weight, rebellious_sample_weight, metrics)。"""
    device = correctness_mask.device
    corr = correctness_mask.detach().float()

    # 1) 本 batch 各 UID 的成功/总数
    succ, tot = {}, {}
    for i, uid in enumerate(uids):
        tot[uid]  = tot.get(uid, 0.0) + 1.0
        succ[uid] = succ.get(uid, 0.0) + float(corr[i].item())

    # 2) EMA 平滑 + Beta 先验 + 置信度收缩 → smoothed_pass_rate
    decay         = float(cfg.problem_pass_rate_state_decay)        # 默认 0.85
    prior_a       = float(cfg.problem_pass_rate_prior_alpha)        # 默认 1.0
    prior_b       = float(cfg.problem_pass_rate_prior_beta)         # 默认 1.0
    conf_scale    = float(cfg.problem_pass_rate_confidence_scale)   # 默认 8.0
    neutral       = float(cfg.problem_pass_rate_neutral_pass_rate)  # 默认 0.5

    smoothed = {}
    for uid, total in tot.items():
        s = router_state.get(uid, {"success": 0.0, "failure": 0.0, "last_step": float(step)})
        delta = max(float(step) - s["last_step"], 0.0)
        k = decay ** delta                              # 跨步衰减
        succ_stat = k * s["success"] + succ[uid]
        fail_stat = k * s["failure"] + (total - succ[uid])
        router_state[uid] = {"success": succ_stat, "failure": fail_stat, "last_step": float(step)}

        raw_pass  = (succ_stat + prior_a) / (succ_stat + fail_stat + prior_a + prior_b)
        eff       = succ_stat + fail_stat
        conf      = eff / (eff + conf_scale) if conf_scale > 0 else 1.0
        smoothed[uid] = conf * raw_pass + (1.0 - conf) * neutral

    pass_rate = torch.tensor([smoothed[uid] for uid in uids], dtype=corr.dtype, device=device)

    # 3) sigmoid 软门 → 难/中/易 隶属度（归一化）
    temp        = float(cfg.problem_pass_rate_soft_temperature)     # 默认 0.08
    hard_thr    = float(cfg.problem_pass_rate_hard_threshold)       # 默认 0.3
    easy_thr    = float(cfg.problem_pass_rate_easy_threshold)       # 默认 0.7
    hard_gate   = torch.sigmoid((hard_thr - pass_rate) / temp)      # pass 低 → 难
    easy_gate   = torch.sigmoid((pass_rate - easy_thr) / temp)      # pass 高 → 易
    medium_gate = (1.0 - hard_gate) * (1.0 - easy_gate)
    g_sum = (hard_gate + medium_gate + easy_gate).clamp(min=1e-6)
    hard_gate, medium_gate, easy_gate = hard_gate / g_sum, medium_gate / g_sum, easy_gate / g_sum

    teacher_mask   = teacher_available_mask.to(corr.dtype)
    correct_mask   = corr
    incorrect_mask = 1.0 - correct_mask

    # 4) 失败样本的 SDPO 权重
    easy_fail_w  = float(cfg.problem_pass_rate_easy_fail_sdpo_weight)    # 0.35
    medium_w     = float(cfg.problem_pass_rate_medium_sdpo_weight)       # 0.0
    hard_fail_w  = float(cfg.problem_pass_rate_hard_fail_sdpo_weight)    # 1.0
    fail_sdpo_w  = easy_fail_w * easy_gate + medium_w * medium_gate + hard_fail_w * hard_gate

    # 5) 成功样本的 SDPO 权重（仅当不强制二元路由才给成功样本也蒸）
    hard_success_sdpo_w = float(cfg.problem_pass_rate_hard_success_sdpo_weight)  # 0.7
    success_sdpo_w = hard_success_sdpo_w * hard_gate

    binary = bool(cfg.problem_pass_rate_binary_sample_routing)        # 默认 True
    if binary:        # SRPO 语义：成功全归 GRPO，失败(有教师)归 SDPO
        sdpo_weight = teacher_mask * incorrect_mask * fail_sdpo_w
    else:
        sdpo_weight = teacher_mask * (incorrect_mask * fail_sdpo_w
                                      + correct_mask * success_sdpo_w)

    # 6) 成功样本的 GRPO 权重（易题成功几乎不学，难题成功仍学）
    easy_succ_grpo = float(cfg.problem_pass_rate_easy_success_grpo_weight)  # 0.05
    medium_grpo    = float(cfg.problem_pass_rate_medium_grpo_weight)        # 1.0
    hard_succ_grpo = float(cfg.problem_pass_rate_hard_success_grpo_weight)  # 0.3
    grpo_weight = correct_mask * (easy_succ_grpo * easy_gate
                                  + medium_grpo * medium_gate
                                  + hard_succ_grpo * hard_gate)
    grpo_weight = torch.where(teacher_mask > 0, grpo_weight, torch.ones_like(grpo_weight))

    # 7) 叛逆 bonus 的样本权重（默认仅中难度成功样本）
    rebellious_sw = torch.zeros_like(grpo_weight)
    if cfg.problem_pass_rate_rebellious_enabled:
        gate = torch.zeros_like(grpo_weight)
        if cfg.problem_pass_rate_rebellious_medium_enabled:
            gate = gate + medium_gate * float(cfg.problem_pass_rate_rebellious_medium_weight)
        if cfg.problem_pass_rate_rebellious_easy_enabled:
            gate = gate + easy_gate * float(cfg.problem_pass_rate_rebellious_easy_weight)
        if cfg.problem_pass_rate_rebellious_hard_enabled:
            gate = gate + hard_gate * float(cfg.problem_pass_rate_rebellious_hard_weight)
        rebellious_sw = teacher_mask * correct_mask * gate

    metrics = {
        "routing/problem_pass_rate_smoothed_mean": float(pass_rate.mean()),
        "routing/problem_pass_rate_hard_fraction": float(hard_gate.mean()),
        "routing/problem_pass_rate_sdpo_weight_mean": float(sdpo_weight.mean()),
        "routing/problem_pass_rate_grpo_weight_mean": float(grpo_weight.mean()),
        "routing/problem_pass_rate_rebellious_weight_mean": float(rebellious_sw.mean()),
    }
    return sdpo_weight, grpo_weight, rebellious_sw, metrics