
# DRIFT: Difficulty Routing Self-DIstillation with Rhythm-Gated Exploration and Success BuFfer Training
<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>
<p align="center">
  <a href="https://lianjiatech.github.io/drift/blog"><img src="https://img.shields.io/badge/Blog-DRIFT-green" alt="Blog"></a>
  <a href="https://arxiv.org/abs/2606.30345"><img src="https://img.shields.io/badge/arXiv-2606.30345-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/linglingdan"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow" alt="Hugging Face"></a>
</p>

DRIFT 是由 Beike Language and Intelligence 团队提出的一个开源在线自演化策略优化框架。该框架旨在使大型语言模型在不依赖外部专家监督的情况下，实现复杂推理和科学问题求解能力的持续提升。

---

## 🌟 核心亮点 (Key Results)

* **🏆 综合性能第一**：在 5 个学术与多轮推理基准测试中平均得分 **79.5 (Avg@16)**，位列第一。
* **🛠️ 工具调用跃升**：在 Tool Use 任务上达到 **79.2**，相较于 SRPO 显著提升 **+8.0**。
* **🚀 基础模型大幅强化**：相比于基座模型 Qwen3-8B，全维度平均提升达 **+30.0**。

---

## 📖 简介与方法 (Methodology)

现有大模型在自我蒸馏和强化学习中常遭遇学分分配不精确的问题。DRIFT 通过协同整合以下四大核心机制，构建了一个结构化的自我改进循环：

1. **动态难度路由 (Dynamic Difficulty Routing)**：利用历史通过率作为模型掌握程度的稳定估计，在宏观层面上动态决定何时从成功的轨迹中提取经验（自蒸馏）、何时探索新策略（RL）以及何时减少冗余训练。
2. **结构化节奏门控探索 (Rhythm Gating Exploration)**：在微观（Token）层面上，引入一种保持结构的探索机制。仅当一个 Token 在局部表现优于教师模型，并且是关键的结构锚点（如逻辑或因果节点）时，才放大策略更新。
3. **两阶段课程学习 (Two-Stage Curriculum Learning)**：第一阶段通过熵驱动的自蒸馏（Entropy-Driven Self-Distillation）快速预热并积累高质量的成功缓冲区；第二阶段转向平衡的探索、纠错与收敛的自演化路由。
4. **基于成功缓冲的经验回放 (Experience Replay via Success Buffer)**：复用高质量历史轨迹，为边缘问题提供结构化探索奖励，支持模型平滑演化。

---

## 📊 实验结果 (Experimental Results)

基于 Qwen3-8B，DRIFT 在多个领域（生物、化学、材料、物理及工具调用）中表现出了强大的跨领域泛化能力和训练稳定性。

| Method / Model | Biology | Chemistry | Materials | Physics | Tool Use | Average |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-8B (Base) | 30.8 | 41.2 | 58.9 | 59.2 | 57.5 | 49.5 |
| SDPO (Paper) | 56.8 | 80.9 | 78.4 | 75.6 | 68.5 | 72.0 |
| SDPO (Reproduced) | 64.8 | 78.9 | 76.1 | 72.7 | 67.7 | 72.0 |
| GRPO (Paper) | 59.9 | 74.5 | 77.1 | 72.7 | 65.7 | 70.0 |
| GRPO (Reproduced) | 47.4 | 65.6 | 73.5 | 60.6 | 67.7 | 63.0 |
| SRPO (Paper) | 72.8 | **83.0** | **81.5** | 78.4 | 71.2 | 77.4 |
| SC-SDPO (Paper) | 65.4 | 80.6 | 79.3 | **81.6** | 67.3 | 74.8 |
| **DRIFT (Ours)** | **74.4** | 82.0 | 81.4 | 80.5 | **79.2** | **79.5** |

---

## 一、 获取代码与补丁合并

请在您的终端中执行以下命令获取项目最终代码：

```bash
git clone https://github.com/LianjiaTech/drift.git && cd drift && \
cat patch/sdpo_patch_part_* > sdpo_modifications.patch && \
git clone https://github.com/LDLINGLINGLING/SDPO.git temp_sdpo_base && \
cd temp_sdpo_base && git apply --whitespace=nowarn ../sdpo_modifications.patch && cd .. && \
rsync -av \
  --exclude='.git' \
  --exclude='README.md' \
  --exclude='.gitignore' \
  --exclude='*pseudocode.py' \
  temp_sdpo_base/ ./ && \
rm -rf temp_sdpo_base sdpo_modifications.patch
```

---

## 二、 实验脚本运行指南

合并完成后，所有的运行代码和配置都将应用至 `DRIFT-main` 文件夹中。您需要先进入 `DRIFT-main` 目录，然后即可直接运行以下实验脚本：

```bash
cd DRIFT-main
```

### 1. 科学推理评估实验 (SciKnowEval)

针对不同的学科数据集（以 Qwen3-8B 为基础），我们提供了以下运行脚本（路径位于 `experiments/generalization/run_sciknoweval_qwen3-8b/` 下）：

* **材料科学 (Material)**：
  ```bash
  bash experiments/generalization/run_sciknoweval_qwen3-8b/run_sciknoweval_material.sh
  ```
* **生物学 (Biology)**：
  ```bash
  bash experiments/generalization/run_sciknoweval_qwen3-8b/run_sciknoweval_biology.sh
  ```
* **物理学 (Physics)**：
  ```bash
  bash experiments/generalization/run_sciknoweval_qwen3-8b/run_sciknoweval_physics.sh
  ```
* **化学 (Chemistry)**：
  ```bash
  bash experiments/generalization/run_sciknoweval_qwen3-8b/run_sciknoweval_chemistry.sh
  ```

### 2. 工具调用实验 (ToolUse)

针对工具使用任务（以 Qwen3-8B 为基础），运行脚本路径位于 `experiments/generalization/run_qwen3-8b/` 下：

* **工具使用 (ToolUse)**：
  ```bash
  bash experiments/generalization/run_qwen3-8b/run_tooluse_dift.sh
  ```

> 💡 **运行提示**：在 `DRIFT-main` 目录下可直接一键执行上述对应的 `.sh` 脚本，每个脚本对应的名称即代表了研究所采用的特定数据集和任务。

---

## 三、 核心机制伪代码说明

我们在 `drift` 根目录下提供了 DRIFT 框架三大核心机制的独立 Python 伪代码，供快速阅读与研究：

* **`buffer_pseudocode.py` (Success Buffer 机制)**
  * **机制说明**：展示了历史高质量成功轨迹的在线复用与动态替换逻辑。通过维护一个基于奖励（Reward-based）的轻量级 Success Buffer，在当前批次探索失败时提供特权信息指导，避免样本浪费，保持模型稳定的自演化记忆。
* **`problem_pass_rate_routing_pseudocode.py` (Difficulty Routing 机制)**
  * **机制说明**：展示了样本难度的自适应路由控制。通过平滑维护每个问题的历史通过率（EMA Pass Rate），动态评估模型当前的认知边界。根据掌握程度，将困难样本路由至自蒸馏纠错通道，边界样本路由至强化探索通道，
* **`rhythm_rebellious_pseudocode.py` (Rhythm-Gated Rebellious Bonus 机制)**
  * **机制说明**：展示了 Token 级别的结构化探索增强。结合学生与教师模型在推理路径上的局部条件熵塌缩（Entropy Collapse）差异，定位关键的逻辑转折与因果锚点（Anchor Rhythm）。仅当学生在这些结构性位置上局部超越（Rebellious Bonus）教师时，才选择性放大策略更新梯度，从而在控制噪声的同时实现高精度的信用分配。