"""CPU unit and sanity tests for core MCARE modules."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.compute_tables_and_se import competitive_ranks, compute_rank_summary, mean_and_se
from common.config import get_paths
from common.io_utils import canonical_json_key, dump_json, load_json, read_jsonl, sha256_file, write_jsonl
from common.prompts_and_schemas import digit_expectation, verify_audit
from evaluation.compute_boundary_breakdown import exact_mcnemar_p
from evaluation.compute_lexical_metrics import compute_subset_metrics, ead_n, entropy_n, lcs_length, tokenize
from evaluation.need_analysis import dim_status, mar_status, need_status
from stage2.pair_selection import percentile, pick, values_for


class TestMCARECore(unittest.TestCase):
    def test_config_and_io_roundtrip(self) -> None:
        cfg = get_paths()
        self.assertTrue(cfg.root.exists())
        self.assertFalse(cfg.train_annotator_url.endswith("/"))
        self.assertFalse(cfg.eval_judge_url.endswith("/"))
        self.assertNotEqual(cfg.train_annotator_model, cfg.eval_judge_model)

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            rows = [{"id": "a", "x": 1}, {"id": "b", "x": 2}]
            jpath = tdir / "rows.jsonl"
            write_jsonl(jpath, rows)
            self.assertEqual(read_jsonl(jpath), rows)
            mpath = tdir / "meta.json"
            dump_json(mpath, {"sha": sha256_file(jpath)})
            self.assertEqual(len(load_json(mpath)["sha"]), 64)
            self.assertEqual(canonical_json_key({"b": 2, "a": 1}), canonical_json_key({"a": 1, "b": 2}))

    def test_digit_expectation(self) -> None:
        # Token probs concentrated on digit "4" -> expectation ~ 4.0
        top = {"4": -0.01, "3": -5.0, "5": -6.0}
        val, mass = digit_expectation(top)
        self.assertIsNotNone(val)
        assert val is not None
        self.assertGreater(mass, 0.98)
        self.assertAlmostEqual(val, 4.0, delta=0.05)

    def test_verify_audit_and_need_status(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Please just listen, do not give advice."},
        ]
        parsed = {
            "positive_need": [{"kind": "listen", "description": "just listen", "span": "Please just listen"}],
            "negative_need": [{"kind": "no_advice", "description": "no advice", "span": "do not give advice"}],
            "facts": [],
            "updates": [],
        }
        verified, drops = verify_audit(parsed, messages)
        self.assertEqual(len(verified["positive_need"]), 1)
        self.assertEqual(len(verified["negative_need"]), 1)
        self.assertEqual(sum(drops.values()), 0)

        judgement_ok = {
            "requests": ["yes"],
            "restrictions": ["no"],
            "unsupported_projection": "no",
            "fact_conflict": "no",
            "identity_assumption": "no",
        }
        st_ok = dim_status(judgement_ok, verified, "MCARE", "item1", None)
        self.assertEqual(need_status(st_ok), "ok")
        self.assertEqual(mar_status(st_ok), "ok")

        judgement_bad = dict(judgement_ok, restrictions=["yes"])
        st_bad = dim_status(judgement_bad, verified, "MCARE", "item1", None)
        self.assertEqual(need_status(st_bad), "err")

    def test_pair_selection_and_percentile(self) -> None:
        pct = percentile([float(i) for i in range(8)])
        self.assertAlmostEqual(pct[0], 0.0)
        self.assertAlmostEqual(pct[-1], 1.0)

        cs = [{"text": " ".join(["word"] * (18 + i)), "hit_cap": False} for i in range(8)]
        h = {
            "cs": cs,
            "valid": list(range(8)),
            "costs": [1, 0, 1, 0, 0, 1, 0, 0],
            "has_reference": False,
            "quality": [0.9, 0.85, 0.7, 0.5, 0.4, 0.8, 0.3, 0.2],
            "warmth": [0.0] * 8,
            "support": [0.0] * 8,
            "future": [0.0] * 8,
            "reference": [0.0] * 8,
        }
        w = {"quality": 1.0, "warmth": 0.0, "support": 0.0, "reference": 0.0}
        vals = values_for(h, w)
        chosen = 1  # compliant highest quality (0.85)
        rejected, reason = pick(h, vals, 0.25, chosen)
        self.assertIsNone(reason)
        self.assertEqual(rejected, 0)  # highest-value violating candidate within +-25% length window

    def test_lexical_metrics(self) -> None:
        refs = {"esconv:1": "i hear how hard this week has been"}
        resps = {"esconv:1": "i hear how hard this week has been for you"}
        res = compute_subset_metrics(["esconv:1"], refs, resps)
        self.assertEqual(res["n"], 1)
        self.assertAlmostEqual(res["rouge_l_recall"], 1.0)
        self.assertGreater(res["ead_2"], 0.0)
        self.assertGreater(res["ent_4"], 0.0)
        self.assertEqual(lcs_length(tokenize("a b c"), tokenize("a x b c")), 3)
        self.assertGreater(ead_n([tokenize("one two three four")], 2), 0.0)
        self.assertGreaterEqual(entropy_n([tokenize("one two three four five")], 4), 0.0)

    def test_mcnemar_and_ranking(self) -> None:
        self.assertAlmostEqual(exact_mcnemar_p(0, 4), 0.125)
        self.assertAlmostEqual(exact_mcnemar_p(2, 1), 1.0)

        mn, se, n = mean_and_se([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(n, 4)
        self.assertAlmostEqual(mn, 2.5)
        self.assertAlmostEqual(se, math.sqrt(5.0 / 3.0) / 2.0, places=6)

        rk = competitive_ranks({"A": 10.0, "B": 10.0, "C": 8.0}, "max")
        self.assertEqual(rk, {"A": 1, "B": 1, "C": 3})

        summary = compute_rank_summary(
            {
                "Base": {"M1_NFR": 0.70, "controlled_NVR": 0.05},
                "MCARE": {"M1_NFR": 0.95, "controlled_NVR": 0.01},
            },
            {"M1_NFR": "max", "controlled_NVR": "min"},
            ours_name="MCARE",
        )
        self.assertEqual(summary["MCARE"]["best_metrics"], 2)
        self.assertEqual(summary["MCARE"]["mean_rank"], 1.0)
        self.assertEqual(summary["Base"]["mcare_win_tie_loss"], {"win": 2, "tie": 0, "loss": 0})

    def test_losses_torch_optional(self) -> None:
        try:
            import torch
            from stage2.losses import distill, dpo, reversal
        except ImportError:
            self.skipTest("torch not installed in current Python environment")

        pol = torch.tensor([0.5, -0.2], requires_grad=True)
        ref = torch.tensor([0.1, -0.1])
        loss = dpo(pol, ref, beta=0.1) + reversal(pol[0], pol[1]) + distill(pol, ref)
        loss.backward()
        self.assertIsNotNone(pol.grad)
        self.assertTrue(torch.isfinite(pol.grad).all().item())

    def test_heldout_parent_isolation(self) -> None:
        from common.io_utils import load_heldout_parents

        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp) / "policy"
            write_jsonl(pdir / "esconv_dev.jsonl", [{"parent_id": "esc_dev_1"}])
            write_jsonl(pdir / "esconv_test.jsonl", [{"parent_id": "esc_test_1"}])
            write_jsonl(pdir / "ed_valid.jsonl", [{"parent_id": "ed_val_1"}])
            write_jsonl(pdir / "ed_test.jsonl", [{"parent_id": "ed_test_1"}])
            extra = Path(tmp) / "eval_inputs.jsonl"
            write_jsonl(extra, [{"parent_id": "extra_eval_1"}])
            held = load_heldout_parents(pdir, (extra,))
            self.assertEqual(held, {"esc_dev_1", "esc_test_1", "ed_val_1", "ed_test_1", "extra_eval_1"})

    def test_stage1_and_stage2_synthetic_builders(self) -> None:
        from common.io_utils import dump_json, key
        from stage1.build_base_pairs import build_base_pairs, score_candidate_pool
        from stage2.annotate_candidates import PROMPT as ANN_PROMPT, SCHEMA as ANN_SCHEMA
        from stage2.build_stage2_pairs import _build_split_pairs

        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "I feel overwhelmed."}]
        cands = [{"candidate_id": f"c{i}", "text": " ".join(["support"] * (20 + i)), "hit_cap": False} for i in range(8)]
        checks = [
            {
                "index": i,
                "warmth": 4 if i in (0, 2, 4, 6) else 2,
                "request_violation": i in (1, 3),
                "boundary_violation": False,
                "fact_violation": False,
                "identity_violation": False,
                "fabrication": False,
                "candidate_quote": "support" if i in (1, 3) else "",
                "user_quote": "overwhelmed" if i in (1, 3) else "",
            }
            for i in range(8)
        ]

        q_scores = [0.2, 0.9, 0.85, 0.4, 0.6, 0.1, 0.75, 0.3]
        s_scores = [0.3, 0.2, 0.8, 0.1, 0.5, 0.1, 0.7, 0.2]
        h_scores = [0.4, 0.5, 0.9, 0.2, 0.6, 0.1, 0.8, 0.3]

        vals, has_ref = score_candidate_pool(cands, checks, q_scores, s_scores, h_scores)
        self.assertTrue(has_ref)
        self.assertEqual(len(vals), 8)
        self.assertGreater(vals[2], vals[0])

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            cand_path = tdir / "candidates.jsonl"
            ann_dir = tdir / "annotations"
            q_path = tdir / "quality.jsonl"
            s_path = tdir / "support.jsonl"
            ref_path = tdir / "ref.jsonl"
            out_s1 = tdir / "out_s1"

            gid = "p1"
            write_jsonl(
                cand_path,
                [{"id": gid, "source": "esconv", "parent_id": "par1", "messages": msgs, "candidates": cands, "reference": "ok"}],
            )
            dump_json(ann_dir / f"{key(gid)}.json", {"id": gid, "protocol_hash": key([ANN_PROMPT, ANN_SCHEMA]), "checks": checks})
            write_jsonl(q_path, [{"id": gid, "scores": [{"candidate_id": f"c{i}", "score": q_scores[i]} for i in range(8)]}])
            write_jsonl(s_path, [{"id": gid, "scores": [{"candidate_id": f"c{i}", "p_E": s_scores[i]} for i in range(8)]}])
            write_jsonl(ref_path, [{"id": gid, "scores": [{"candidate_id": f"c{i}", "score": h_scores[i]} for i in range(8)]}])

            summary = build_base_pairs(cand_path, ann_dir, q_path, s_path, out_s1, reference_scores_path=ref_path)
            self.assertEqual(summary["pairs"], 1)
            s1_rows = read_jsonl(out_s1 / "stage1_base_pairs.jsonl")
            self.assertEqual(s1_rows[0]["rejected_response"], cands[1]["text"])

            s2_rows, s2_meta = _build_split_pairs(
                [{"id": gid, "source": "esconv", "parent_id": "par1", "kind": "natural", "messages": msgs, "candidates": cands}],
                {f"c{i}": float(i + 1) for i in range(8)},
                {f"c{i}": h_scores[i] for i in range(8)},
                {f"c{i}": s_scores[i] for i in range(8)},
                ann_dir,
                weights={"e": 0.50, "h": 0.25, "o": 0.25},
                apply_gate=True,
            )
            self.assertEqual(len(s2_rows), 1)
            self.assertEqual(s2_meta["rows"], 1)
            self.assertEqual(s2_rows[0]["rejected_response"], cands[1]["text"])

    def test_evaluation_and_analysis_imports(self) -> None:
        import analysis.compute_tables_and_se
        import evaluation.build_m2_empathy_table
        import evaluation.compute_boundary_breakdown
        import evaluation.compute_lexical_metrics
        import evaluation.compute_m2_joint_table
        import evaluation.compute_mechanism_e1_e3
        import evaluation.probes_fg.compute_fg_tables

        self.assertTrue(callable(analysis.compute_tables_and_se.compute_rank_summary))
        self.assertTrue(callable(evaluation.probes_fg.compute_fg_tables.compute_f))


if __name__ == "__main__":
    unittest.main()
