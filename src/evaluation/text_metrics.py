"""Local automatic text metrics: ROUGE-L F1, BERTScore-F1 (roberta-large), Distinct-1/2, words, and hit_cap.

Computed on M1 main splits (esconv_main, ed_main) against the original supporter response; M2 has no reference reply.
Output: text_metrics.jsonl and text_metrics_corpus.json under MCARE_EVAL_DIR.
usage: PYTHONPATH=src python src/evaluation/text_metrics.py --labels Base MCARE ...
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common import prompts_and_schemas as C  # noqa: E402


def distinct(texts, n):
    grams = Counter()
    for t in texts:
        toks = re.findall(r"\w+|[^\w\s]", t.lower(), flags=re.UNICODE)
        grams.update(zip(*(toks[i:] for i in range(n))))
    total = sum(grams.values())
    return len(grams) / total if total else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    from bert_score import score as bert_score
    from rouge_score.rouge_scorer import RougeScorer
    gen_inputs = {r["id"]: r for r in C.read_jsonl(C.EVAL / "generation_inputs.jsonl")}
    refs = json.load(open(C.EVAL / "references.json"))
    out_path = C.EVAL / "text_metrics.jsonl"
    have = {(r["label"], r["id"]) for r in C.read_jsonl(out_path)} if out_path.exists() else set()
    corpus_path = C.EVAL / "text_metrics_corpus.json"
    corpus = json.load(open(corpus_path)) if corpus_path.exists() else {}
    scorer = RougeScorer(["rougeL"], use_stemmer=True)
    for label in args.labels:
        rows = C.read_jsonl(C.EVAL / "gen" / f"{label}.jsonl")
        todo = [r for r in rows if r["id"] in refs and set(gen_inputs[r["id"]]["sets"]) & {"esconv_main", "ed_main"} and (label, r["id"]) not in have]
        if todo:
            cands = [r["response"] for r in todo]
            rs = [refs[r["id"]] for r in todo]
            _, _, f1 = bert_score(cands, rs, model_type="bert-base-uncased", num_layers=9, lang="en", idf=False, rescale_with_baseline=False, device=args.device, batch_size=32, verbose=False)
            with open(out_path, "a", encoding="utf-8") as handle:
                for r, c, ref, b in zip(todo, cands, rs, f1.tolist()):
                    handle.write(json.dumps({"label": label, "id": r["id"], "rougeL": scorer.score(ref, c)["rougeL"].fmeasure, "bert_f1": b, "words": len(c.split())}) + "\n")
        for set_name in ("esconv_main", "esconv_sens", "ed_main", "m2_d0", "m2_d1", "m2_d1l"):
            sel = [r for r in rows if set_name in gen_inputs[r["id"]]["sets"]]
            if not sel:
                continue
            texts = [r["response"] for r in sel]
            corpus.setdefault(label, {})[set_name] = {"n": len(sel), "distinct1": distinct(texts, 1), "distinct2": distinct(texts, 2), "mean_words": sum(len(t.split()) for t in texts) / len(texts),
                                                      "hit_cap": sum(bool(r["hit_cap"]) for r in sel), "empty": sum(not r["response"].strip() for r in sel)}
        print("done", label, flush=True)
    json.dump(corpus, open(corpus_path, "w"), indent=1)


if __name__ == "__main__":
    main()
