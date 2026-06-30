# DRIFT: Difficulty Routing Self-DIstillation with Rhythm-Gated Exploration and Success BuFfer Training

<p align="center">
  <b>English</b> | <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
  <a href="https://lianjiatech.github.io/drift/blog/"><img src="https://img.shields.io/badge/Blog-DRIFT-green" alt="Blog"></a>
  <a href="https://arxiv.org/abs/2606.30345"><img src="https://img.shields.io/badge/arXiv-2606.30345-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/linglingdan"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow" alt="Hugging Face"></a>
</p>

DRIFT is an open-source online self-evolving policy optimization framework proposed by the Beike Language and Intelligence team. This framework is designed to enable large language models to continuously improve their complex reasoning and scientific problem-solving capabilities without relying on external expert supervision.

---

## 🌟 Key Results

* **🏆 Top Comprehensive Performance**: Achieved a leading average score of **79.5 (Avg@16)** across 5 academic and multi-turn reasoning benchmarks.
* **🛠️ Significant Tool-Use Improvement**: Reached **79.2** on Tool Use tasks, representing a notable **+8.0** improvement over SRPO.
* **🚀 Substantial Base Model Enhancement**: Delivered an average improvement of **+30.0** across all dimensions compared to the Qwen3-8B base model.

---

## 📖 Methodology

Existing large language models often face inaccurate credit assignment challenges in self-distillation and reinforcement learning. DRIFT addresses this by integrating four core mechanisms to build a structured self-improvement loop:

1. **Dynamic Difficulty Routing**: Utilizes historical pass rates as a stable estimation of model mastery. At a macro level, it dynamically decides when to extract experience from successful trajectories (self-distillation), when to explore new policies (RL), and when to reduce redundant training.
2. **Structured Rhythm Gating Exploration**: Introduces a structure-preserving exploration mechanism at the micro (token) level. Policy updates are amplified only when a token locally outperforms the teacher model and serves as a critical structural anchor (such as logical or causal nodes).
3. **Two-Stage Curriculum Learning**: Stage one uses Entropy-Driven Self-Distillation to quickly warm up the model and build a high-quality success buffer. Stage two transitions to a balanced self-evolving routing of exploration, error correction, and convergence.
4. **Experience Replay via Success Buffer**: Reuses high-quality historical trajectories to provide structured exploration rewards for marginal problems, facilitating a smooth evolution of the model.

---

## 📊 Experimental Results

Evaluated on Qwen3-8B, DRIFT demonstrates strong cross-domain generalization and training stability across multiple fields, including biology, chemistry, materials, physics, and tool use.

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

## 1. Code Acquisition and Patch Merging

Run the following commands in your terminal to clone the repository and apply the patches:

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

## 2. Experimental Script Execution Guide

Once the merge is complete, all execution code and configurations will be applied to the `DRIFT-main` folder. Navigate to the `DRIFT-main` directory to run the experimental scripts:

```bash
cd DRIFT-main
```

### 1. Scientific Reasoning Evaluation (SciKnowEval)

We provide scripts for different scientific domains based on the Qwen3-8B model (located under `experiments/generalization/run_sciknoweval_qwen3-8b/`):

* **Materials Science (Material)**:
  ```bash
  bash experiments/generalization/run_sciknoweval_qwen3-8b/run_sciknoweval_material.sh
  ```
* **Biology (Biology)**:
  ```bash
  bash experiments/generalization/run_sciknoweval_qwen3-8b/run_sciknoweval_biology.sh
  ```
* **Physics (Physics)**:
  ```bash
  bash experiments/generalization/run_sciknoweval_qwen3-8b/run_sciknoweval_physics.sh
  ```
* **Chemistry (Chemistry)**:
  ```bash
  bash experiments/generalization/run_sciknoweval_qwen3-8b/run_sciknoweval_chemistry.sh
  ```

### 2. Tool Use Experiments (ToolUse)

For tool-use tasks based on Qwen3-8B, the script is located in the `experiments/generalization/run_qwen3-8b/` directory:

* **Tool Use**:
  ```bash
  bash experiments/generalization/run_qwen3-8b/run_tooluse_dift.sh
  ```

> 💡 **Running Tip**: You can execute the corresponding `.sh` scripts directly from the `DRIFT-main` directory. The name of each script indicates the specific dataset and task evaluated.

---

## 3. Core Mechanism Pseudocode

In the root directory of the `drift` repository, we provide independent Python pseudocode for DRIFT's three core mechanisms for reference and research:

* **`buffer_pseudocode.py` (Success Buffer Mechanism)**
  * **Description**: Demonstrates the online reuse and dynamic replacement logic of high-quality historical successful trajectories. By maintaining a lightweight, reward-based Success Buffer, it provides privileged guidance when the current exploration batch fails, preventing sample waste and maintaining stable self-evolving model memory.
* **`problem_pass_rate_routing_pseudocode.py` (Difficulty Routing Mechanism)**
  * **Description**: Demonstrates adaptive routing based on problem difficulty. By tracking a smoothed historical pass rate (EMA Pass Rate) for each problem, the system dynamically estimates the model's current cognitive boundary. Based on the mastery level, difficult samples are routed to the self-distillation error correction channel, while boundary samples are routed to the reinforcement exploration channel.
* **`rhythm_rebellious_pseudocode.py` (Rhythm-Gated Rebellious Bonus Mechanism)**
  * **Description**: Demonstrates token-level structured exploration enhancement. By analyzing the differences in local conditional entropy collapse of both student and teacher models along the reasoning path, it localizes critical logical transitions and causal decision anchors (Anchor Rhythm). It selectively amplifies policy gradients only when the student locally outperforms the teacher (Rebellious Bonus) at these structural positions, ensuring fine-grained credit assignment while effectively filtering background noise.