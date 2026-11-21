# MCARE: Need-Anchored Preference Learning for User-Directed Support under Autism Disclosure

Reference implementation accompanying:

> **Empathy Without Presumption: Need-Anchored Preference Learning for User-Directed Support under Autism Disclosure**

---

## Repository Structure

```text
MCARE_Code/
├── configs/
│   └── paths.example.env                  # Environment template for model weights, data paths, and judge endpoints
├── src/
│   ├── common/
│   │   ├── config.py                      # Path and model configuration with environment overrides
│   │   ├── io_utils.py                    # Atomic JSON/JSONL I/O, canonical hashing, and split parent utilities
│   │   ├── prompts_and_schemas.py         # Phase-A/B evaluation prompts, JSON schemas, and empathy digit expectation
│   │   └── hf_worker.py                   # Non-thinking bf16 decoder, K=8 sampler, and OpenAI-compatible HTTP server
│   ├── data/
│   │   ├── prepare_policy_datasets.py     # ESConv, EmpatheticDialogues, and HelpSteer2 parent-level split builder
│   │   ├── build_natural_pool.py          # 2,000 parent-isolated natural dialogue training prompts
│   │   └── build_evaluation_panels.py     # M1/M2 evaluation panels and controlled negative-boundary probes
│   ├── stage1/
│   │   ├── build_base_pairs.py            # Offline pi_0 base pool builder (Eq. 31: s_i = 0.50 Q + 0.25 W + 0.10 S + 0.15 H)
│   │   ├── build_stage1_dataset.py        # Merges P_base + structured P_rev/P_need/P_warm/P_req/P_acc (Eq. 17 -> 10,583 rows)
│   │   └── train_stage1.sh                # Stage-1 DPO + positive NLL training -> pi_1 (Qwen3.5-9B + LoRA r=16, alpha=32)
│   ├── stage2/
│   │   ├── losses.py                      # Specialist distillation, margin ranking, and DPO losses
│   │   ├── prepare_specialist_and_contrasts.py  # Prepare specialist histories & structured need-contrast inputs
│   │   ├── compile_specialist_and_contrasts.py  # Compile P_rev/P_need/P_warm/P_req/P_acc & specialist_piE sets
│   │   ├── train_specialist.py            # Empathy specialist (pi_E) trainer (Section III-C)
│   │   ├── build_stage2_histories.py      # Select natural (2,714) and controlled (1,020) Stage-2 histories
│   │   ├── annotate_candidates.py         # Offline five-flag semantic compliance + warmth annotator (Qwen3.8-27B)
│   │   ├── score_candidates.py            # Candidate scorers: empathy (Qwen3.8-27B), helpfulness (Qwen3.8-27B), support RM
│   │   ├── pair_selection.py              # Within-prompt percentile ranking and length-matched (+/-25%) negative selection
│   │   ├── build_stage2_pairs.py          # Build Stage-2 MCARE pairs (5,372 rows) and Table VI self-sampled / nogate ablations
│   │   ├── train_stage2.sh                # Stage-2 DPO + NLL continuing from pi_1 checkpoint with pi_1 fixed ref -> pi_2 (MCARE)
│   │   └── interpolate_adapters.py        # Exact LoRA adapter interpolation baseline (0.75 pi_E + 0.25 pi_1)
│   ├── baselines/
│   │   ├── matched_pool_common.py         # Shared 2k matched-pool paths & feedback audit helpers
│   │   ├── matched_pool_data.py           # Matched-pool feedback audit and preference pool construction
│   │   ├── matched_pool_compute.py        # Sampling, candidate scoring, Joint-DPO, and SafeDPO pair generation
│   │   ├── safedpo_rlhf.py                # SafeDPO compliance-first margin trainer wrapper
│   │   ├── train_matched_dpo.sh           # Launcher for Vanilla-SFT, Quality/Support/Joint-DPO, and SafeDPO
│   │   └── external/                      # RLFF, PERM, PEER-SFT/RUR, ESCRAG-R1, and RLVER reproduction pipelines
│   ├── evaluation/
│   │   ├── generate_responses.py          # Greedy non-thinking evaluation generation (max_tokens=768)
│   │   ├── run_need_judge.py              # Phase-B semantic compliance + empathy logprob judge (Gemma-4-31B-it)
│   │   ├── remote_judge_client.py         # OpenAI-compatible HTTP client wrapper for evaluation
│   │   ├── score_armorm.py                # ArmoRM-Llama3-8B-v0.1 quality scorer
│   │   ├── score_esc_rank.py              # ESC-RANK InternLM2-7B LoRA support/empathy/overall scorer
│   │   ├── need_analysis.py               # Status logic (dim_status, need_status, mar_status) & clustered bootstrap CIs
│   │   ├── text_metrics.py                # Per-response ROUGE-L, BERTScore, and Distinct-n evaluation
│   │   ├── compute_lexical_metrics.py     # Table II metrics: EAD-2, Ent-4, and ROUGE-L Recall on ESConv & ED
│   │   ├── matched_baseline_metrics.py    # Unified per-model metric extractor
│   │   ├── score_m2_empathy.py            # Same-context M2 empathy scoring (D0, D1, D1L)
│   │   ├── build_m2_empathy_table.py      # M2 same-context empathy & disclosure deltas (Table III)
│   │   ├── compute_m2_joint_table.py      # M2 need transitions & attribution-free retention J_0,1 / J_0,1L (Table III, Fig. 7)
│   │   ├── compute_boundary_breakdown.py  # Controlled negative-boundary NVR breakdown + exact two-sided McNemar test (Table IV)
│   │   ├── compute_mechanism_e1_e3.py     # Mechanism transfer & forgetting diagnostics (Table VI, Fig. 8)
│   │   └── probes_fg/                     # Controlled Probe F (personal evidence adaptation) & Probe G (need updates)
│   └── analysis/
│       ├── compute_tables_and_se.py       # Across-sample SE (s/sqrt(n)), 14-metric competitive ranking & win/tie/loss (Table V)
│       └── make_figures.py                # Vector PDF figures (Figs. 3-9) with +/- 1 SE error bars
├── tests/
│   └── test_mcare_core.py                 # CPU unit and sanity tests
├── requirements.txt
└── environment.yml
```

---

## Paper Section Mapping

| Paper Section / Asset | Key Implementation Files |
|---|---|
| Task Formulation & Panels (Section III-A, IV-A) | `src/data/prepare_policy_datasets.py`, `src/data/build_natural_pool.py`, `src/data/build_evaluation_panels.py` |
| Stage 1 Need-First Policy ($\pi_1$, Section III-B, IV-C) | `src/stage1/build_base_pairs.py`, `src/stage1/build_stage1_dataset.py`, `src/stage1/train_stage1.sh` |
| Empathy Specialist ($\pi_E$, Section III-C) | `src/stage2/prepare_specialist_and_contrasts.py`, `src/stage2/compile_specialist_and_contrasts.py`, `src/stage2/losses.py`, `src/stage2/train_specialist.py` |
| Stage 2 Transfer ($\pi_2$ / MCARE, Section III-C, IV-C) | `src/stage2/build_stage2_histories.py`, `src/stage2/annotate_candidates.py`, `src/stage2/score_candidates.py`, `src/stage2/pair_selection.py`, `src/stage2/build_stage2_pairs.py`, `src/stage2/train_stage2.sh` |
| 13 Baselines (Section IV-B) | `src/baselines/` & `src/baselines/external/` |
| Evaluation & Statistical Tests (Section IV-D, Tables I–V) | `src/evaluation/`, `src/evaluation/compute_lexical_metrics.py`, `src/analysis/compute_tables_and_se.py` |
| Mechanism Ablations (Section V-E, Table VI, Fig. 8) | `src/stage2/interpolate_adapters.py`, `src/stage2/build_stage2_pairs.py --mode ablations`, `src/evaluation/compute_mechanism_e1_e3.py` |
| Probes F & G (Section V-F/G, Tables VII–VIII, Fig. 9) | `src/evaluation/probes_fg/` |
| Manuscript Figures (Figs. 3–9) | `src/analysis/make_figures.py` |

---

## Quick Start

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate mcare
cp configs/paths.example.env .env
# Edit .env with local model paths, training annotator endpoint, and test judge endpoint
```

### 2. Verify Installation (CPU Unit Tests)

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

### 3. Training & Inference Workflow (Summary)

```bash
# 1) Stage 1: build pi_0 base pool + structured contrasts -> train pi_1
PYTHONPATH=src python src/stage1/build_base_pairs.py
PYTHONPATH=src python src/stage1/build_stage1_dataset.py
bash src/stage1/train_stage1.sh outputs/stage1/stage1_dataset.jsonl outputs/stage1/pi1_s42 42

# 2) Stage 2: train specialist pi_E, annotate & score candidates, build Stage-2 pairs, and continue DPO+NLL from pi_1
PYTHONPATH=src python src/stage2/build_stage2_histories.py
PYTHONPATH=src python src/stage2/annotate_candidates.py --candidates outputs/stage2/data/stage2_candidates.jsonl --out-dir outputs/stage2
PYTHONPATH=src python src/stage2/score_candidates.py --metric empathy --candidates outputs/stage2/data/stage2_candidates.jsonl --out outputs/stage2/data/empathy_scores.jsonl
PYTHONPATH=src python src/stage2/score_candidates.py --metric helpfulness --candidates outputs/stage2/data/stage2_candidates.jsonl --out outputs/stage2/data/helpfulness_scores.jsonl
PYTHONPATH=src python src/stage2/score_candidates.py --metric support --candidates outputs/stage2/data/stage2_candidates.jsonl --out outputs/stage2/data/support_scores.jsonl
PYTHONPATH=src python src/stage2/build_stage2_pairs.py --mode mcare
bash src/stage2/train_stage2.sh outputs/stage2/data/stage2_pairs.jsonl outputs/stage1/pi1_s42/final outputs/stage2/MCARE_s42 42

# 3) Evaluation (greedy non-thinking decoding, max_tokens=768)
PYTHONPATH=src python src/evaluation/generate_responses.py \
  --model outputs/stage2/MCARE_s42/final \
  --out outputs/evaluation/gen/MCARE.jsonl \
  --max-tokens 768
PYTHONPATH=src python src/evaluation/compute_lexical_metrics.py --help
PYTHONPATH=src python src/analysis/compute_tables_and_se.py --help
```

---

## Model & Data Configuration

- **Policy Backbone**: `Qwen/Qwen3.5-9B` (`enable_thinking=False`) with LoRA (`r=16`, `alpha=32`, `dropout=0.05` on all linear layers).
- **Offline Annotation & Scoring**: `Qwen/Qwen3.8-27B` (`MCARE_TRAIN_ANNOTATOR_URL`) for five-flag semantic review, warmth/empathy scoring, and general helpfulness scoring, paired with the support-utility reward model (`MCARE_TRAIN_SUPPORT_RM`) and HelpSteer2 critic (`MCARE_TRAIN_HELPFULNESS_CRITIC`).
- **Evaluation Suite**: `google/gemma-4-31b-it` (`MCARE_EVAL_JUDGE_URL`) for semantic compliance and empathy evaluation, `RLHFlow/ArmoRM-Llama3-8B-v0.1` for general response quality, `ESC-RANK` (`internlm2-chat-7b` + LoRA) for ESC support quality, and `roberta-large` for BERTScore.
- **Public Datasets**: Download ESConv, EmpatheticDialogues, and HelpSteer2 into `data/raw/` and run `src/data/prepare_policy_datasets.py` to build the processed JSONL splits. Large checkpoints and dataset files are ignored via `.gitignore`.
