"""Replay protected and common-window quantile decisions and session oracles from trial CSVs."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
import frozen_io as frozen
from acdd.metrics import (TrialScore, balanced_accuracy_from_predictions,
                          oracle_logit_threshold, prior_quantile_logit_threshold, roc_auc_from_raw_logits)
from acdd.strengthening import leave_one_session_out_oracle


def decide(context, evaluation):
    """This interface accepts only scores. Labels never enter the fit or prediction."""
    cut = prior_quantile_logit_threshold(context, .5)
    return cut, [int(s >= cut) for s in evaluation]


def evaluate(context, scores, labels):
    threshold, predictions = decide(context, scores)
    baseline = balanced_accuracy_from_predictions(labels, [int(s >= 0) for s in scores])
    policy = balanced_accuracy_from_predictions(labels, predictions)
    return dict(threshold=threshold, n_trials=len(scores), default_ba=baseline, policy_ba=policy,
                gain=policy-baseline, decisions_sha256=hashlib.sha256(bytes(predictions)).hexdigest())


def grouped_rows(path, protocol):
    groups = defaultdict(list)
    with frozen.local_path(path).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            score = float(row["raw_logit"])
            diff = float(row["logit1"]) - float(row["logit0"])
            if not np.isfinite([diff,score]).all() or abs(diff-score) > 1e-6 or row["dataset_id"] != "Stieger2021":
                raise ValueError("Invalid dataset or logit pair")
            key = row["subject_key"]
            groups[key].append((int(row["trial_order"]),row["session_id"],row["run_id"],
                                int(row["trial_index"]),int(row["y_true"]),score))
    for person in groups:
        groups[person].sort(key=lambda entry: entry[0])
    return groups


def write_csv(out, name, rows):
    frozen.write_csv(out / name, rows)


def session_number(value):
    if not re.fullmatch(r"\d+", value):
        raise ValueError(f"Unexpected Stieger session ID {value}")
    return int(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, default=frozen.TARGET_ROOT)
    parser.add_argument("--manifest", type=Path, default=frozen.PROVENANCE)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--oracles", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((HERE / "common_suffix_protocol.json").read_text(encoding="utf-8"))
    assert plan["k"] == [5,10,20,50]
    units = json.loads(frozen.read_json(args.manifest))["units"]
    assert len(units) == len({u["unit_id"] for u in units}) == 30
    for backbone in ("eegnet","eeg_conformer"):
        selection = [u for u in units if u["backbone"] == backbone]
        conditions = {u["protocol_id"] for u in selection}
        assert len(conditions) == 5
        assert all(sorted(u["seed"] for u in selection if u["protocol_id"] == p) == [17,29,43] for p in conditions)
    identity, excluded, by_unit, protected, records, oracle_rows = {}, [], [], [], [], []
    for index, u in enumerate(units):
        path = args.target_root / u["target_predictions_relative_path"]
        if frozen.sha(path) != u["target_predictions_sha256"]:
            raise ValueError(str(path))
        groups = grouped_rows(path,u["protocol_id"])
        if index and set(groups) != set(identity):
            raise ValueError("Participant membership differs between frozen model units")
        for p, rows in sorted(groups.items()):
            order, sessions, runs, indices, labels, scores = map(list,zip(*rows))
            num = [session_number(s) for s in sessions]
            assert len(set(order)) == len(order) and num == sorted(num) and set(labels) == {0,1}
            keys = list(zip(order,sessions,runs,indices,labels))
            digest = frozen.content_hash(keys)
            suffix_digest = frozen.content_hash(keys[50:])
            if index == 0:
                identity[p] = dict(stream_sha256=digest,suffix_sha256=suffix_digest,n=len(keys),
                                   suffix_n=max(0,len(keys)-50),sessions=len(set(sessions)))
                if len(keys) <= 50 or len(set(labels[50:])) != 2:
                    excluded.append(p)
            if identity[p]["stream_sha256"] != digest or identity[p]["suffix_sha256"] != suffix_digest:
                raise ValueError(f"Ordered evaluation row keys/labels differ in {p} {u['unit_id']}")
            common = dict(unit_id=u["unit_id"],backbone=u["backbone"],protocol_id=u["protocol_id"],
                          seed=u["seed"],participant_id=p)
            primary = evaluate(scores[:10],scores[10:],labels[10:])
            _, ref_ba, _ = oracle_logit_threshold(labels[10:],scores[10:])
            primary.update(raw_logit_auc=roc_auc_from_raw_logits(labels[10:],scores[10:]),static_oracle_ba=ref_ba)
            protected.append({**common,**primary})
            if p not in excluded:
                for k in plan["k"]:
                    metric = evaluate(scores[:k],scores[50:],labels[50:])
                    records.append({**common,"k":k,"suffix_sha256":suffix_digest,**metric})
            if args.oracles:
                trials = [TrialScore(participant_id=p,dataset_id="Stieger2021",protocol_id=u["protocol_id"],
                                     y_true=y,raw_logit=s,order=o,session_id=sess)
                          for o,sess,_,_,y,s in rows]
                result, folds, exc = leave_one_session_out_oracle(trials)
                oracle_rows.append({**common,"valid_folds":len(folds),"excluded_folds":len(exc),
                                    "in_sample_decision_loss":result.in_sample_decision_loss,
                                    "cross_fitted_decision_loss":result.cross_fitted_decision_loss})
        by_unit.append({"unit_id":u["unit_id"],"sha256":u["target_predictions_sha256"],"participants":len(groups)})
        print(u["unit_id"], len(groups), flush=True)
    write_csv(args.output,"protected.csv",protected)
    write_csv(args.output,"common_suffix.csv",records)
    if args.oracles:
        write_csv(args.output,"oracles.csv",oracle_rows)
    persons = sorted(set(identity) - set(excluded))
    if not persons:
        raise ValueError("Empty common cohort")
    draws = np.random.default_rng(20261110).integers(0,len(persons),size=(10000,len(persons)))
    summaries = []
    for backbone in ("eegnet","eeg_conformer"):
        for k in plan["k"]:
            selected = [r for r in records if r["backbone"]==backbone and r["k"]==k]
            cells = defaultdict(list)
            for r in selected:
                cells[r["participant_id"]].append(r)
            if set(cells)!=set(persons) or any(len(cell)!=15 for cell in cells.values()):
                raise ValueError("Unbalanced participant/model grid")
            values = np.array([np.mean([r["gain"] for r in cells[p]]) for p in persons])
            lo,hi = np.quantile(values[draws].mean(axis=1),[.025,.975])
            summaries.append(dict(backbone=backbone,k=k,n=len(persons),models=15,
                                  distinct_suffix_trials=sum(identity[p]["suffix_n"] for p in persons),
                                  mean_gain=values.mean(),ci_lower=lo,ci_upper=hi))
    write_csv(args.output,"common_summary.csv",summaries)
    fixed = []
    for u in units:
        gains = [r["gain"] for r in protected if r["unit_id"] == u["unit_id"]]
        fixed.append(dict(unit_id=u["unit_id"],backbone=u["backbone"],seed=u["seed"],protocol_id=u["protocol_id"],
                          n=len(gains),negative=sum(g<0 for g in gains),decrease_gt_2pp=sum(g<-.02 for g in gains),
                          mean_gain=np.mean(gains)))
    write_csv(args.output,"fixed_models.csv",fixed)
    frozen.write_json(args.output / "audit.json",dict(status="PASS",plan_sha256=frozen.sha(HERE / "common_suffix_protocol.json"),
                      inputs=by_unit,participants=identity,excluded=excluded,
                      unique_total_trials=sum(v["n"] for v in identity.values()),
                      protected_rows=len(protected),common_rows=len(records),oracle_rows=len(oracle_rows),
                      chronological_session_check="recorded trial_order and numeric session IDs",
                      bootstrap_draws_sha256=hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest()))
    print("Replayed", len(protected), "protected,", len(records), "common-suffix and",len(oracle_rows),"oracle rows")


if __name__ == "__main__":
    main()
