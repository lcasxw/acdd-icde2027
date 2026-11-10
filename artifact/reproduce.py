"""Portable entry points for the ICDE EAB candidate. Run from an extracted package."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
import sys

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE / name) for name in ("src", "engine", "analysis")]
from public_paths import system_path

BASE = system_path(BASE)
VERSION = "platt-v2-centered-convex-box-20261015"
POLICIES = ("source_platt", "target_prior_quantile", "platt_em", "rlls_bbse")

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_bytes())

def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write(path, records):
    with path.open("w", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        out.writeheader()
        out.writerows(records)

def verify_members():
    index = read(BASE / "MANIFEST.json")
    if index["calibration_version"] != VERSION:
        raise ValueError("Calibration version differs from the candidate")
    for name, expected in index["files"].items():
        if digest(BASE / name) != expected:
            raise ValueError("Package member mismatch: " + name)

def verify_inputs(action, directory):
    absent = []
    for row in rows(BASE / "required_inputs.csv"):
        if action != "verify-inputs" and action not in row["commands"].split("|"):
            continue
        path = directory / row["path"]
        if not path.is_file():
            absent.append(row["path"])
        elif digest(path) != row["sha256"]:
            raise ValueError("Input hash mismatch: " + row["path"])
    if absent:
        print(f"NOT_RUN: {len(absent)} required inputs absent. See required_inputs.csv. First: {absent[0]}")
        raise SystemExit(2)

def policy_unit(task):
    from acdd.metrics import TrialScore
    from snapshot import load_model, shared_snapshot, metric_with_record
    from score_replay import grouped_rows
    unit, root = task
    uid = unit["unit_id"]
    src = rows(root / uid / "source.csv")
    x, y = [float(r["raw_logit"]) for r in src], [int(r["y_true"]) for r in src]
    snap = read(root / uid / "platt_v2.json")
    if snap["unit_id"] != uid or snap["source_csv_sha256"] != unit["source_validation_predictions_sha256"]:
        raise ValueError("Snapshot source mismatch: " + uid)
    model = load_model(snap)
    result = []
    with shared_snapshot(model,x,y):
        for person, group in grouped_rows(root / uid / "target.csv",unit["protocol_id"]).items():
            trials = [TrialScore(participant_id=person,dataset_id="Stieger2021",protocol_id=unit["protocol_id"],
                       order=t[0],session_id=t[1],y_true=t[4],raw_logit=t[5]) for t in group]
            for name in (*POLICIES,"source_platt_prior_cutoff"):
                prior_cutoff = name == "source_platt_prior_cutoff"
                method = "source_platt" if prior_cutoff else name
                metric = metric_with_record(trials,x,y,model,method,"prefix_suffix",.001,prior_cutoff=prior_cutoff)
                result.append(dict(unit_id=uid,participant_id=person,protocol_id=unit["protocol_id"],seed=unit["seed"],
                              backbone=unit["backbone"],method=name,gain=float(metric["gain"])))
    return result

def policies(inputs, out, jobs):
    from summaries import three_weight_bootstrap
    units = read(BASE / "units.json")["units"]
    tasks = [(u,inputs) for u in units]
    groups = defaultdict(list)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for unit in pool.map(policy_unit,tasks):
            for r in unit:
                groups[r["backbone"],r["method"]].append(r)
    result = []
    for exp in rows(BASE / "expected/table_iv.csv"):
        k=(exp["backbone"],exp["method"])
        group=groups[k]
        people=defaultdict(list)
        for r in group:
            people[r["participant_id"]].append(r)
        if len(people)!=62 or any(len(v)!=15 or len({(r["protocol_id"],r["seed"]) for r in v})!=15 for v in people.values()):
            raise ValueError("Incomplete participant/condition/seed grid")
        participant=[mean(r["gain"] for r in v) for v in people.values()]
        est=three_weight_bootstrap(group,"gain",10000,int(exp["seed"]))["participant_weighted"]
        got=dict(version=VERSION,backbone=k[0],method=k[1],seed=int(exp["seed"]),mean_gain=est["point"],
                 ci_lower=est["ci_lower"],ci_upper=est["ci_upper"],negative=sum(g<0 for g in participant),
                 below_minus_2pp=sum(g<-.02 for g in participant))
        for field in ("mean_gain","ci_lower","ci_upper","negative","below_minus_2pp"):
            if not math.isclose(float(exp[field]),got[field],abs_tol=1e-12,rel_tol=0):
                raise ValueError(f"{k} {field} differs from the candidate")
        result.append(got)
    utility=[]
    for r in rows(BASE/"expected/platt_v2_utility_alignment.csv"):
        g=groups[r["backbone"],r["method"]]
        value=mean(item["gain"] for item in g)
        check(value,r["mean_gain"])
        utility.append(dict(version=VERSION,backbone=r["backbone"],method=r["method"],mean_gain=value))
    write(out/"source_cutoffs.csv",utility)
    write(out/"table_iv.csv",result)
    print("PASS: eight protected Table IV rows from 30 trial files and 30 persisted source-only snapshots, zero refits.")

def check(value, expected):
    if not math.isclose(float(value),float(expected),rel_tol=0,abs_tol=1e-12):
        raise ValueError(f"Aggregate mismatch: {value} vs {expected}")

def known_prior_example(out):
    from fractions import Fraction
    from known_prior import witness

    computed = witness()
    expected = rows(BASE / "expected/revision_known_prior_witness.csv")
    def keyed(records):
        return {(r["world"], int(r["n_low_scores_predicted_negative"])):
                (Fraction(str(r["positive_prior"])), r["threshold_interval"],
                 Fraction(r["balanced_accuracy_exact"])) for r in records}
    if len(computed) != 18 or len(expected) != 18 or keyed(computed) != keyed(expected):
        raise ValueError("The 18 known-prior threshold splits differ from the paper")
    write(out / "known_prior_witness.csv", computed)
    print("PASS: both priors are 1/2. Optimal balanced accuracies are 3/4 and 13/20.")

def oracle_intervals(oracle_rows, full_defaults, out):
    from summaries import rebuild_crossfit

    # Leave-one-session-out and in-sample values both use complete eligible sessions.
    for r in oracle_rows:
        default = full_defaults[r["unit_id"], r["participant_id"]]
        for fit in ("in_sample", "cross_fitted"):
            r[fit + "_headroom"] = float(r[fit + "_decision_loss"]) + default - 0.5
    settings = read(BASE / "configs/q1_tnsre_strengthening_v1.json")["cross_fitted_oracle"]
    fitted = rebuild_crossfit(oracle_rows, settings["bootstrap_samples"], settings["bootstrap_seed"])
    estimates = {(r["backbone"], r["metric"]): r for r in fitted if r["weighting"] == "participant_weighted"}
    reference = rows(BASE / "expected/figure4_cross_fitted_oracle.csv")
    if len(estimates) != 8 or len(reference) != 8:
        raise ValueError("Expected eight backbone/metric combinations for Figure 5")
    output = []
    for r in reference:
        est = estimates[r["backbone"], r["metric"]]
        for field in ("point", "ci_lower", "ci_upper"):
            check(est[field], r[field])
        output.append(dict(backbone=r["backbone"], metric=r["metric"], replayed_point=est["point"],
                           replayed_ci_lower=est["ci_lower"], replayed_ci_upper=est["ci_upper"],
                           archived_ci_lower=r["ci_lower"], archived_ci_upper=r["ci_upper"]))
    write(out / "fig5_oracle.csv", output)

def model_degradation(protected, out):
    import sqlite3

    # Two orders of aggregation over the same 62 participants and 15 models per backbone.
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE gains (backbone TEXT, model TEXT, person TEXT, gain REAL, PRIMARY KEY (backbone, model, person))")
        db.executemany("INSERT INTO gains VALUES (?, ?, ?, ?)",
                       ((r["backbone"], r["unit_id"], r["participant_id"], float(r["gain"])) for r in protected))
        result = db.execute("""
            WITH decisions AS (
                SELECT backbone, model, person, gain FROM gains
                UNION ALL
                SELECT backbone, 'model_average', person, AVG(gain) FROM gains GROUP BY backbone, person
            )
            SELECT backbone, model, COUNT(*), SUM(gain < -0.02)
            FROM decisions GROUP BY backbone, model ORDER BY backbone, model
        """).fetchall()
    expected = {(r["backbone"], r["unit_id"]): int(r["decrease_gt_2pp"])
                for r in rows(out / "scores/fixed_models.csv")}
    for bb in ("eegnet", "eeg_conformer"):
        expected[bb, "model_average"] = 1
    if (len(result) != 32 or any(n != 62 for _, _, n, _ in result)
            or {(bb, model): count for bb, model, _, count in result} != expected):
        raise ValueError("The fixed-model and model-averaged degradation counts differ from the protected record")
    write(out / "degradation_by_aggregation.csv",
          [dict(backbone=bb, model=model, participants=n, below_minus_2pp=count) for bb, model, n, count in result])

def score_displays(inputs,out):
    import numpy as np
    from score_replay import grouped_rows
    from windows import past_decisions
    from intervals import interval
    from score_replay import decide
    from acdd.metrics import balanced_accuracy_from_predictions as ba
    from summaries import three_weight_bootstrap
    subprocess.run([sys.executable,"-B",str(BASE/"engine/score_replay.py"),"--oracles","--manifest",str(BASE/"units.json"),
                    "--target-root",str(inputs),"--output",str(out/"scores")],check=True)
    protected=rows(out/"scores/protected.csv")
    gains={(r["unit_id"],r["participant_id"]):float(r["gain"]) for r in protected}
    registry=rows(BASE/"expected/fig3.csv")
    units=read(BASE/"units.json")["units"]
    records=defaultdict(list)
    moving=defaultdict(list)
    full_defaults={}
    access_expected=rows(BASE/"expected/access.csv")
    for u in units:
        for p,stream in grouped_rows(inputs/u["unit_id"]/"target.csv",u["protocol_id"]).items():
            scores=[x[5] for x in stream]; labels=[x[4] for x in stream]
            raw=[s>=0 for s in scores]
            past=list(past_decisions(scores,5))
            for k in (5,10,20,50):
                base=ba(labels[k:],raw[k:])
                for ep,pred in (("prefix_suffix",decide(scores[:k],scores[k:])[1]),("prequential",past[k-5:])):
                    g=ba(labels[k:],pred)-base
                    moving[u["backbone"],ep,k,p].append(g)
                    if k==10:
                        records[u["backbone"],ep].append(dict(protocol_id=u["protocol_id"],participant_id=p,gain=g))
                        if ep=="prefix_suffix":
                            check(g,gains[u["unit_id"],p])
            full_defaults[u["unit_id"],p]=ba(labels,raw)
            end=next(i for i,trial in enumerate(stream) if trial[1]!=stream[0][1])
            for ep,ctx,first in (("full_prior",scores,0),("session1_to_later",scores[:end],end)):
                predictions=decide(ctx,scores[first:])[1]
                g=ba(labels[first:],predictions)-ba(labels[first:],raw[first:])
                records[u["backbone"],ep].append(dict(protocol_id=u["protocol_id"],participant_id=p,gain=g))
    access={}
    for exp in access_expected:
        bb,ep,weight=exp["backbone"],exp["endpoint"],exp["weighting"]
        fitted=three_weight_bootstrap(records[bb,ep],"gain",10000,int(exp["seed"]))[weight]
        for f,g in (("point","mean_gain"),("ci_lower","ci_lower"),("ci_upper","ci_upper")):check(fitted[f],exp[g])
        access[bb,ep,weight]=dict(backbone=bb,endpoint=ep,weighting=weight,**fitted)
    write(out/"access.csv",list(access.values()))
    fig=[]
    for exp in registry:
        if exp["cohort"]=="Stieger2021":
            ep={"prefix10_suffix":"prefix_suffix","prequential":"prequential"}[exp["endpoint"]]
            value=access[exp["backbone"],ep,"participant_weighted"]
            for f,g in (("point","mean_gain"),("ci_lower","ci_lower"),("ci_upper","ci_upper")):check(value[f],exp[g])
            fig.append(value)
    write(out/"fig3_stieger.csv",fig)
    response=[]
    for r in rows(BASE/"expected/review_k_response.csv"):
        bb,ep,k=r["backbone"],r["endpoint"],int(r["prefix_length"])
        vals=[mean(moving[key]) for key in sorted(moving) if key[:3]==(bb,ep,k)]
        assert len(vals)==62
        lo,hi=interval(vals,f"k/{bb}/{ep}/{k}")
        if k==10:
            est=access[bb,ep,"participant_weighted"];lo,hi=est["ci_lower"],est["ci_upper"]
        for v,f in ((mean(vals),"mean_gain"),(lo,"ci_lower"),(hi,"ci_upper")):check(v,r[f])
        response.append(dict(backbone=bb,endpoint=ep,k=k,mean_gain=mean(vals),ci_lower=lo,ci_upper=hi))
    write(out/"moving.csv",response)
    absolute=[]
    for r in rows(BASE/"expected/review_absolute_summary.csv"):
        bb,m=r["backbone"],r["metric"]; people=defaultdict(list)
        for item in protected:
            if item["backbone"]!=bb:continue
            ref=float(item["static_oracle_ba"])
            derived={"headroom":ref-.5,"decision_loss":ref-float(item["default_ba"]),"residual_loss":ref-float(item["policy_ba"])}
            people[item["participant_id"]].append(derived[m] if m in derived else float(item[m]))
        vals=[mean(people[p]) for p in sorted(people)]
        lo,hi=interval(vals,"absolute/"+bb)
        if m=="gain":
            est=access[bb,"prefix_suffix","participant_weighted"];lo,hi=est["ci_lower"],est["ci_upper"]
        for v,f in ((mean(vals),"mean"),(lo,"ci_lower"),(hi,"ci_upper")):check(v,r[f])
        absolute.append(dict(backbone=bb,metric=m,mean=mean(vals),ci_lower=lo,ci_upper=hi))
    write(out/"absolute.csv",absolute)
    oracle_intervals(rows(out / "scores/oracles.csv"), full_defaults, out)
    model_degradation(protected, out)
    radar=[]; seed_pairs=[]; quartiles=[]
    for bb in ("eegnet","eeg_conformer"):
        grid=[r for r in protected if r["backbone"]==bb]
        conds=sorted({r["protocol_id"] for r in grid})
        people=sorted({r["participant_id"] for r in grid})
        if len(grid)!=930 or len(conds)!=5 or len(people)!=62: raise ValueError("Incomplete protected grid")
        for c in conds:
            for seed in ("17","29","43","all"):
                g=[float(r["gain"]) for r in grid if r["protocol_id"]!=c and (seed=="all" or r["seed"]==seed)]
                radar.append(dict(backbone=bb,omitted=c,seed=seed,mean_gain=mean(g)))
        ranked={}
        for seed in ("17","29","43"):
            vals=[mean(float(r["gain"]) for r in grid if r["participant_id"]==p and r["seed"]==seed) for p in people]
            # Average ranks for ties, matching Spearman for this complete cohort.
            ranked[seed]=[1+sum(w<v for w in vals)+(sum(w==v for w in vals)-1)/2 for v in vals]
        for a,b in (("17","29"),("17","43"),("29","43")):
            seed_pairs.append(dict(backbone=bb,seed_a=a,seed_b=b,rho=float(np.corrcoef(ranked[a],ranked[b])[0,1])))
        for label in ("raw_logit_auc","decision_loss"):
            vals=[]; pg=[]
            for p in people:
                obs=[r for r in grid if r["participant_id"]==p]
                vals.append(mean(float(r["raw_logit_auc"]) if label=="raw_logit_auc" else float(r["static_oracle_ba"])-float(r["default_ba"]) for r in obs))
                pg.append(mean(float(r["gain"]) for r in obs))
            edges=[-np.inf,*np.quantile(vals,[.25,.5,.75]),np.inf]
            for i in range(4):
                chosen=[g for x,g in zip(vals,pg) if edges[i]<x<=edges[i+1]]
                strat="auc" if label=="raw_logit_auc" else label
                lo,hi=interval(chosen,f"quartile/{bb}/{strat}/{i+1}")
                quartiles.append(dict(backbone=bb,stratifier=strat,quartile=i+1,n_participants=len(chosen),mean_gain=mean(chosen),
                                      harm_any_negative=mean(x<0 for x in chosen),harm_gt_2pp=mean(x<-.02 for x in chosen),ci_lower=lo,ci_upper=hi))
    for r in rows(BASE/"expected/review_harm_quartiles.csv"):
        cell=next(item for item in quartiles if (item["backbone"],item["stratifier"],item["quartile"])==(r["backbone"],r["stratifier"],int(r["quartile"])))
        for f in ("n_participants","mean_gain","harm_any_negative","harm_gt_2pp","ci_lower","ci_upper"): check(cell[f],r[f])
    for r in rows(BASE/"expected/figure5_seed_consistency.csv"):
        cell=next(v for v in seed_pairs if (v["backbone"],v["seed_a"],v["seed_b"])==(r["backbone"],r["seed_left"],r["seed_right"]))
        check(cell["rho"],r["spearman_gain"])
    for r in rows(BASE/"expected/figure5_leave_one_protocol_out.csv"):
        cell=next(v for v in radar if (v["backbone"],v["omitted"],v["seed"])==(r["backbone"],r["omitted_protocol"],"all"))
        check(cell["mean_gain"],r["mean_gain"])
    write(out/"fig4_radar.csv",radar);write(out/"table_v.csv",seed_pairs);write(out/"fig6_quartiles.csv",quartiles)

def historical(inputs,out):
    import analyze_v4_dreyer_confirmation as dreyer
    import summarize_v5_historical_estimands as historical_stats
    plan=BASE/"configs/dreyer_plan.json"; cfg=BASE/"configs/v4_journal_protocol.json"
    metrics,complete=dreyer.collect_subject_metrics(plan,inputs/"dreyer",cfg)
    if not complete["is_complete"]: raise ValueError("Incomplete locked Dreyer grid")
    report,clustered=dreyer.analyze(metrics,read(cfg),10000,20260812)
    write(out/"dreyer_subject_metrics.csv",metrics)
    write(out/"dreyer_clustered_effects.csv",clustered)
    (out/"dreyer_confirmation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    historical_stats.run(inputs/"discovery_subject_metrics.csv",out/"dreyer_subject_metrics.csv",out/"historical",10000)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["verify","verify-inputs","known-prior","table-iv","scores","historical","construct","signal-controls"])
    ap.add_argument("--inputs",type=Path,default=BASE/"inputs")
    ap.add_argument("--out",type=Path)
    ap.add_argument("--jobs",type=int,default=1)
    args=ap.parse_args()
    args.inputs = system_path(args.inputs)
    if args.out is not None:
        args.out = system_path(args.out)
    verify_members()
    if args.action=="verify":
        print("PASS: package content hashes and declared calibration version.")
        return
    verify_inputs(args.action,args.inputs)
    if args.action=="verify-inputs":
        print(f"PASS: all {len(rows(BASE / 'required_inputs.csv'))} registered external input files match.")
        return
    if args.out is None or args.jobs<1:
        ap.error("supply --out NEW_DIRECTORY and a positive job count")
    args.out.mkdir(parents=True,exist_ok=False)
    if args.action=="known-prior":
        known_prior_example(args.out)
    elif args.action=="table-iv":
        policies(args.inputs,args.out,args.jobs)
    elif args.action=="scores":
        score_displays(args.inputs,args.out)
    elif args.action=="historical":
        historical(args.inputs,args.out)
    elif args.action=="construct":
        from acdd.construct_validity import run_grid
        cells,report=run_grid(read(BASE/"configs/q1_tnsre_strengthening_v1.json")["construct_validity"])
        write(args.out/"construct_validity_cells.csv",cells)
        (args.out/"construct_validity_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        known_prior_example(args.out)
    else:
        records=rows(args.inputs/"ttime_paired_subject_effects.csv")
        people=defaultdict(list)
        for r in records: people[r["subject_key"]].append(r)
        write(args.out/"ttime.csv",[dict(metric=m,participants=len(people),
              effect=mean(mean(float(r["difference_"+m]) for r in v) for v in people.values())) for m in ("balanced_accuracy","roc_auc")])
        grouped=defaultdict(list)
        for r in rows(args.inputs/"tta_clustered_subject_effects.csv"):grouped[r["method"],r["backbone"]].append(r)
        output=[]
        for (method,bb),g in sorted(grouped.items()):
            if len({r["subject_key"] for r in g})!=len(g):raise ValueError("Duplicated clustered subject in signal controls")
            output.append(dict(method=method,backbone=bb,n=len(g),ba_gain=mean(float(r["balanced_accuracy_gain"]) for r in g),
                               auc_gain=mean(float(r["roc_auc_gain"]) for r in g)))
        write(args.out/"tta.csv",output)
if __name__=="__main__":
    main()

