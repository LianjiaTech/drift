import math
from typing import Optional


# ===== 全局状态 =====
# buffer: dict[uid, list[entry]]
# entry = {"response_text": str, "score": float, "global_step": int, "response_length": int}
buffer: dict = {}


def update_success_buffer(batch, response_texts, seq_scores, correctness_mask,
                          cfg, step) -> dict:
    """训练每步调用：把答对的样本存入缓冲区。"""
    if not cfg.success_buffer_enabled:
        return {"success_buffer/enabled": 0.0}

    max_per_uid = int(cfg.success_buffer_max_per_uid)   # 默认 4
    max_uids    = int(cfg.success_buffer_max_uids)      # 默认 50000
    prefer_recent = bool(cfg.success_buffer_prefer_recent)
    min_reward = cfg.success_buffer_min_reward          # 可选

    added = 0
    for i, uid in enumerate(batch.uids):
        if float(correctness_mask[i]) < 0.5:            # 只存答对的
            continue
        score = float(seq_scores[i])
        if min_reward is not None and score < min_reward:
            continue
        text = response_texts[i]
        if not isinstance(text, str) or not text:
            continue

        entry = {"response_text": text, "score": score,
                 "global_step": step, "response_length": len(text)}
        entries = buffer.setdefault(uid, [])

        # 去重：相同文本仅在分数更高/更近期时覆盖
        existing_idx = next((j for j, e in enumerate(entries)
                             if e["response_text"] == text), None)
        if existing_idx is None:
            entries.append(entry)
            added += 1
        else:
            e = entries[existing_idx]
            if e["score"] < score or e["global_step"] < step:
                entries[existing_idx] = entry
                added += 1

        # 排序后截断
        entries.sort(key=lambda e: (e["global_step"], e["score"]) if prefer_recent
                     else (e["score"], e["global_step"]), reverse=True)
        buffer[uid] = entries[:max_per_uid]

    # UID 总数超限：淘汰最差/最旧的
    if len(buffer) > max_uids:
        def best_key(uid):
            e = buffer[uid][0]
            return ((e["global_step"], e["score"]) if prefer_recent
                    else (e["score"], e["global_step"]))
        keep = set(sorted(buffer, key=best_key, reverse=True)[:max_uids])
        for uid in list(buffer):
            if uid not in keep:
                del buffer[uid]

    return {"success_buffer/added_count": float(added),
            "success_buffer/uid_count": float(len(buffer))}


def get_success_buffer_candidates(uid, topk, current_text, cfg) -> tuple[list[str], list[float]]:
    """当本 batch 某题没有成功 sibling 时，取回退教师候选。"""
    entries = buffer.get(uid, [])
    if not entries:
        return [], []

    prefer_recent = bool(cfg.success_buffer_prefer_recent)
    entries = sorted(entries, key=lambda e: (e["global_step"], e["score"]) if prefer_recent
                     else (e["score"], e["global_step"]), reverse=True)

    texts, rewards = [], []
    for e in entries:
        text = e["response_text"]
        if cfg.dont_reprompt_on_self_success and text == current_text:
            continue
        if cfg.remove_thinking_from_demonstration:
            text = strip_think_tags(text)
        texts.append(text)
        rewards.append(e["score"])
        if len(texts) >= topk:
            break
    return texts, rewards


def is_fallback_enabled(cfg, global_step, steps_per_epoch) -> bool:
    """含 step / epoch 延迟的启用判定。"""
    if not cfg.success_buffer_enabled:
        return False
    start_step = int(cfg.success_buffer_start_step or 0)
    if cfg.success_buffer_start_epoch is not None:
        start_step = max(start_step,
                         int(math.ceil(cfg.success_buffer_start_epoch * steps_per_epoch)))
    return global_step >= start_step