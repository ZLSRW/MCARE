"""bge-m3 dense retrieval exactly as in the released ESCRAG-R1 code (verl/workers/fsdp_workers.py retrieve_top_chunks).

* the query is the full decoded initial output (<think> and <response> included);
* similarity = dot product of L2-normalised bge-m3 CLS embeddings against the embeddings of the corpus *responses*;
* the top-n hits are re-ordered by corpus index, not by similarity;
* the second-stage prompt uses RAG_Source/rag-generation-{system,user}.txt with chunks formatted as
  '[chunk i]:\\nhistory:{history}\\nresponse:{response}\\n' joined by '\\n'.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from baselines.external.esc_repro_common import E, ESCRAG_REPO, ROOT

BGE = os.environ.get("MCARE_BGE_MODEL", "BAAI/bge-m3")
import os
INDEX = Path(os.environ.get("ESCRAG_INDEX_DIR", str(E / "escrag_repro/data/rag_index")))
def load_rag_templates():
    return (
        (ESCRAG_REPO / "verl/RAG_Source/rag-generation-system.txt").read_text(encoding="utf-8"),
        (ESCRAG_REPO / "verl/RAG_Source/rag-generation-user.txt").read_text(encoding="utf-8"),
    )


class BgeM3:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(BGE)
        self.model = AutoModel.from_pretrained(BGE, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32).to(device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 16, max_length: int = 8192) -> np.ndarray:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
            cls = self.model(**batch).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls.float(), dim=-1)
            out[idx] = cls.cpu().numpy()
        return out


class Retriever:
    def __init__(self, encoder: BgeM3, index_dir: Path = INDEX):
        self.encoder = encoder
        self.entries = json.loads((index_dir / "entries.json").read_text(encoding="utf-8"))
        self.embeddings = torch.from_numpy(np.load(index_dir / "embeddings.npy").astype("float32"))

    def top(self, queries: list[str], n: int = 3) -> list[list[dict]]:
        q = torch.from_numpy(self.encoder.encode(queries))
        sims = q @ self.embeddings.T
        result = []
        for row in sims:
            ids = sorted(torch.topk(row, k=min(n, row.numel())).indices.tolist())
            result.append([self.entries[i] for i in ids if self.entries[i].get("response")])
        return result


def format_chunks(chunks: list[dict]) -> str:
    return "\n".join(f"[chunk {i}]:\nhistory:{(c.get('history') or '').strip()}\nresponse:{(c.get('response') or '').strip()}\n"
                     for i, c in enumerate(chunks, start=1)).strip()


def rag_messages(history: str, initial_response: str, chunks: list[dict]) -> list[dict]:
    user = load_rag_templates()[1].format(dialogue_history=history, final_chunks=format_chunks(chunks), initial_response=initial_response)
    return [{"role": "system", "content": load_rag_templates()[0]}, {"role": "user", "content": user}]


if __name__ == "__main__":  # consistency check of this encoder against the shipped embeddings
    encoder = BgeM3("cuda")
    shipped = np.load(ESCRAG_REPO / "verl/RAG_Source/rag_index/embeddings.npy")
    entries = json.loads((ESCRAG_REPO / "verl/RAG_Source/rag_index/entries.json").read_text(encoding="utf-8"))
    pick = list(range(0, len(entries), 400))[:8]
    mine = encoder.encode([entries[i]["response"] for i in pick])
    cos = (mine * shipped[pick]).sum(-1) / (np.linalg.norm(mine, axis=-1) * np.linalg.norm(shipped[pick], axis=-1))
    print("cosine(mine, shipped):", np.round(cos, 5).tolist())
