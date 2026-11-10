# Reproduction guide for acdd-public-v1

The repository includes 13 input volumes and a standard-library materialization program. From this directory, run the commands below before starting the four data-dependent analyses. The materializer fetches the small original BNCI cue archive, checks its SHA-256, restores the source-score rows locally and verifies every resulting input. Use --labels /path/to/true_labels.zip to provide the original archive offline. INPUTS in the commands below can be replaced with inputs or omitted.

~~~sh
python -B materialize_inputs.py
python -B reproduce.py verify-inputs
~~~

[Public access and commands](../README.md) · [Source terms](../DATA_SOURCES.md) · [Transport and checkpoint identity](checkpoint_provenance.csv)

## Verify and build the paper

Run from a fresh extraction. Python's standard library suffices for the first command. The paper was compiled with Tectonic 0.17.0+20260731, its LaTeX format cache and the included IEEEtran class and bibliography style. A first installation may require the matching Tectonic resource bundle. From the manuscript directory the entry is main.tex.

~~~sh
python -B reproduce.py verify
python -B reproduce.py known-prior --out run-witness
cd manuscript
tectonic -X compile --outfmt pdf --print --untrusted main.tex
~~~

The root candidate.pdf is the reviewed file. Source paths in the Fig. 3 registry identify inputs in the originating repository. The separate source digest and the row-level lineage can be checked without mistaking those paths for package members. Compilation can change PDF metadata. MANIFEST.json pins the candidate PDF, the complete manuscript tree, active entry, source ZIP and executable code tree. code_origins.json lists the original bytes for copied modules and the definitions used for the portable entry. Source versions are content digests. No project Git commit was recoverable in this checkout. expected/fig3.csv records each plotted mean and interval with its actual source digest. The historical Stieger JSON has its own field. The four comparisons, including the historical primary and hierarchical intervals and the plotted intervals, are in expected/stieger_interval_lineage.csv. The four Stieger means agree with its primary estimates. All four plotted intervals come from the post-confirmation participant bootstrap. Fig. 2(d) uses the two fixed-prefix rows, Fig. 3 uses all four rows and Table II uses the corresponding gain rows. The earlier confirmation intervals use order statistics of resampled participant means. The later intervals use interpolated percentiles after averaging seeds and training conditions for each participant. The original decision used the protected point estimates and the prespecified gate. MANIFEST.json records both interval sources and the Table IV snapshot/input/output version.

## Run a score audit

Use Python 3.10.11 and the package in requirements.txt. Use a different, previously unused output directory for each invocation. INPUTS denotes the directory produced by materialize_inputs.py. Each command verifies the shipped code and relevant frozen file hashes before reading trial data. One worker is sufficient for Table IV.

~~~sh
python -B reproduce.py verify-inputs --inputs INPUTS
python -B reproduce.py table-iv --inputs INPUTS --out run-policies --jobs 4
python -B reproduce.py scores --inputs INPUTS --out run-scores
python -B reproduce.py historical --inputs INPUTS --out run-cohorts
python -B reproduce.py signal-controls --inputs INPUTS --out run-controls
~~~

|Paper result|Lowest input and computation|Output or boundary|
|---|---|---|
|Fig. 1 and Table I|Protocol diagram and access definitions. The score route checks numeric session order, unique order keys, labels on evaluation windows and cross-model agreement on row identity.|Included vector and manuscript. End-to-end online EEG validity remains NOT_RUN.|
|Fig. 2a–c|The construct command uses the fixed simulation configuration and NumPy RNG. It performs 135,000 simulations when explicitly invoked.|construct_validity_cells.csv and construct_validity_report.json. Reference cell means in expected/review_construct_cells.csv.|
|§IV-A known-prior example|The known-prior command enumerates all nine thresholds for each of two monotone posterior distributions with the same uniform score marginal and prior 1/2. It uses exact rational arithmetic and checks 18 rows.|known_prior_witness.csv. The earlier simulation diagnostic in construct_validity_report.json uses different class priors. The fixed-prior example is also written by construct.|
|Fig. 2d and Table II|The scores command computes the protected default, quantile decision, AUC, static reference, decomposition and participant intervals from trial logits. Access summaries use the evaluation rows for each policy.|absolute.csv and access.csv|
|Fig. 3 discovery / Dreyer|The historical command reads the archived discovery participant metrics. For Dreyer it evaluates the recorded prediction tables for 90 registered units, verifies the checkpoint/summaries and joins within participant before estimating endpoints.|dreyer_confirmation.json and historical/historical_estimands.json. Discovery trial-score inference remains outside this replay.|
|Fig. 3 Stieger|Fit on the first 10 or on strictly earlier scores. Evaluate from trial 11. Average seeds and sources within participant, then resample the 62 participants.|fig3_stieger.csv. The corrected registry is expected/fig3.csv. KDE profiles in the supplied vector are a display of participant gains.|
|Table III|The scores command fits each budget on its own prefix. It evaluates the moving suffix and, separately, trial 51 onward for the common-suffix analysis.|moving.csv and scores/common_summary.csv|
|Table IV and cutoff sensitivity|The table-iv command loads 30 persisted source-only Platt-v2 snapshots, checks their source validation logits, scores the protected rows, then averages 15 models within each of 62 participants. The paired posterior cutoffs share the same snapshots.|table_iv.csv and source_cutoffs.csv. The expected table contains only Platt-v2. Eight interval signs agree with historical estimates. Two degradation counts changed.|
|Fig. 4 and Table V|The scores command omits one training condition from the same target cohort, computes seed-specific and pooled means, paired seed correlations, source-balanced intervals and the low-AUC stratum.|fig4_radar.csv, table_v.csv, access.csv and fig6_quartiles.csv|
|Fig. 5|The scores command refits each labeled oracle on other sessions and scores the held-out session. It recomputes eight point estimates and both bounds of each 95% interval from complete eligible sessions, using the historical participant bootstrap and seed schedule.|scores/oracles.csv and fig5_oracle.csv. Replayed columns are compared with archived bounds at absolute tolerance 1e-12. The scores route reads target trials only.|
|Fig. 6 and fixed-model text|The scores command ranks participant averages separately for AUC and decision loss and computes subgroup means, intervals and harm fractions. Fixed-model counts use the 62 people within each individual unit.|fig6_quartiles.csv, scores/fixed_models.csv and degradation_by_aggregation.csv. The last file applies the two aggregation orders to the same trial-derived participant gains.|
|Historical signal-access comparisons|The signal-controls command aggregates previously computed paired T-TIME effects within participant and the archived clustered AdaBN/TENT effects.|ttime.csv and tta.csv. Signal adapters and baseline inference are upstream of this calculation.|

The fixed-prior example and the verify command run with the Python standard library. For numerical routes install the pinned NumPy dependency in requirements.txt. Use a previously unused output directory name.

The simulation is a separate optional command. The preparation of this candidate relied on the prior simulation run and did not repeat it.

~~~sh
python -B reproduce.py construct --out run-simulation
~~~

## Raw EEG to scores

The original inference and dataset-export sources, both model definitions, recorded settings and provenance are under [upstream](../upstream/README.md). The original EEG recordings are obtained from their respective data providers. The historical runner expects the original checkpoint and lock identities. Public checkpoint copies have portable directory metadata. No EEG inference, raw-signal perturbation or historical Platt parameter recovery is asserted for this publication. The score-replay commands use the public inputs and do not execute the upstream training or inference code.

[Data providers and terms](../DATA_SOURCES.md) · [Figure sources](../figures/README.md) · [Publication and verification](../README.md)
