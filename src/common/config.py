"""Environment-driven path and model configuration for MCARE."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    env_root = os.environ.get("MCARE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MCAREPaths:
    root: Path
    data_dir: Path
    policy_dir: Path
    exp_dir: Path
    eval_dir: Path
    stage1_dir: Path
    stage2_dir: Path
    specialist_dir: Path
    baselines_dir: Path
    base_model: str
    roberta_model: str
    # Offline training annotator & reward models
    train_annotator_model: str
    train_annotator_url: str
    train_annotator_served_name: str
    train_annotator_api_key: str
    train_workers: int
    train_support_rm: str
    train_helpfulness_critic: str
    # Evaluation models
    eval_judge_model: str
    eval_judge_url: str
    eval_judge_served_name: str
    eval_judge_api_key: str
    eval_workers: int
    eval_armorm_model: str
    eval_escrank_base_model: str
    eval_escrank_adapter_dir: str
    eval_esc_score_script: str

    # Aliases preserved for evaluation modules
    @property
    def judge_model(self) -> str:
        return self.eval_judge_model

    @property
    def judge_url(self) -> str:
        return self.eval_judge_url

    @property
    def judge_served_name(self) -> str:
        return self.eval_judge_served_name

    @property
    def judge_api_key(self) -> str:
        return self.eval_judge_api_key

    @property
    def judge_workers(self) -> int:
        return self.eval_workers

    @property
    def training_annotator_model(self) -> str:
        return self.train_annotator_model

    @property
    def armorm_model(self) -> str:
        return self.eval_armorm_model

    @property
    def escrank_base_model(self) -> str:
        return self.eval_escrank_base_model

    @property
    def escrank_adapter_dir(self) -> str:
        return self.eval_escrank_adapter_dir

    @property
    def esc_eval_score(self) -> str:
        return self.eval_esc_score_script


def get_paths() -> MCAREPaths:
    root = repo_root()
    data_dir = Path(os.environ.get("MCARE_DATA_DIR", str(root / "data"))).expanduser().resolve()
    exp_dir = Path(os.environ.get("MCARE_EXP_DIR", str(root / "outputs"))).expanduser().resolve()
    eval_dir = Path(os.environ.get("NEED_EVAL_DIR", os.environ.get("MCARE_EVAL_DIR", str(exp_dir / "evaluation")))).expanduser().resolve()
    return MCAREPaths(
        root=root,
        data_dir=data_dir,
        policy_dir=Path(os.environ.get("MCARE_POLICY_DIR", str(data_dir / "processed" / "policy"))).expanduser().resolve(),
        exp_dir=exp_dir,
        eval_dir=eval_dir,
        stage1_dir=exp_dir / "stage1",
        stage2_dir=exp_dir / "stage2",
        specialist_dir=exp_dir / "specialist",
        baselines_dir=exp_dir / "baselines",
        base_model=os.environ.get("MCARE_BASE_MODEL", "Qwen/Qwen3.5-9B"),
        roberta_model=os.environ.get("MCARE_ROBERTA_DIR", "roberta-large"),
        train_annotator_model=os.environ.get("MCARE_TRAIN_ANNOTATOR_MODEL", "Qwen/Qwen3.8-27B"),
        train_annotator_url=os.environ.get("MCARE_TRAIN_ANNOTATOR_URL", "http://127.0.0.1:8078/v1").rstrip("/"),
        train_annotator_served_name=os.environ.get("MCARE_TRAIN_ANNOTATOR_NAME", "qwen"),
        train_annotator_api_key=os.environ.get("MCARE_TRAIN_ANNOTATOR_API_KEY", "EMPTY"),
        train_workers=int(os.environ.get("MCARE_TRAIN_WORKERS", "16")),
        train_support_rm=os.environ.get("MCARE_TRAIN_SUPPORT_RM", str(exp_dir / "models" / "support_rm")),
        train_helpfulness_critic=os.environ.get("MCARE_TRAIN_HELPFULNESS_CRITIC", str(exp_dir / "models" / "helpsteer2_critic.pkl")),
        eval_judge_model=os.environ.get("MCARE_EVAL_JUDGE_MODEL", os.environ.get("MCARE_JUDGE_MODEL", "google/gemma-4-31b-it")),
        eval_judge_url=os.environ.get("MCARE_EVAL_JUDGE_URL", os.environ.get("MCARE_JUDGE_URL", "http://127.0.0.1:8077/v1")).rstrip("/"),
        eval_judge_served_name=os.environ.get("MCARE_EVAL_JUDGE_NAME", "gemma"),
        eval_judge_api_key=os.environ.get("MCARE_EVAL_JUDGE_API_KEY", "EMPTY"),
        eval_workers=int(os.environ.get("MCARE_EVAL_WORKERS", "16")),
        eval_armorm_model=os.environ.get("MCARE_EVAL_ARMORM_MODEL", "RLHFlow/ArmoRM-Llama3-8B-v0.1"),
        eval_escrank_base_model=os.environ.get("MCARE_EVAL_ESCRANK_BASE", "internlm/internlm2-chat-7b"),
        eval_escrank_adapter_dir=os.environ.get("MCARE_EVAL_ESCRANK_ADAPTER", str(root / "models" / "evaluators" / "ESC-RANK")),
        eval_esc_score_script=os.environ.get("MCARE_EVAL_ESC_SCORE", str(root / "external" / "ESC-Eval" / "score.py")),
    )
