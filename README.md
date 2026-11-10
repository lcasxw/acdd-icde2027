# Access-Conditioned Decision Decomposition for Auditing Cross-Dataset Motor-Imagery EEG Decoders

Reproduction files for the ICDE 2027 Experiment, Analysis, and Benchmark submission. The score-replay release is **acdd-public-v1**. The paper, fitting, scoring and aggregation match the preserved submission-prep-4 candidate. The public launcher also handles long Windows extraction paths.

[Current paper PDF](submission/current/ACDD_ICDE2027_public.pdf) · [Matching LaTeX sources](submission/current/ACDD_ICDE2027_public_source.zip) · [Submission text and scope statement](submission/submission-text.txt) · [Per-result reproduction guide](artifact/README.md) · [Data sources and terms](DATA_SOURCES.md)

[Original figure scripts and drawing sources](figures/README.md) · [Upstream inference source and model mapping](upstream/README.md)

[Versioned release and verified downloads](https://github.com/lcasxw/acdd-icde2027/releases/tag/acdd-public-v1) · [Public replay receipt](submission/current/PUBLIC_REPLAY_RECEIPT.json) · [CMT artifact address](submission/current/CMT_ARTIFACT_URL.txt)

The current submission PDF places the public URL in Section VI-B. The frozen preparation paper remains in the analysis snapshot. The repository also contains the original drawing sources, both inference models and the historical-to-public checkpoint mapping.

## Reproduce the results

Download this repository using GitHub's **Code → Download ZIP**, or clone it. Extract the whole repository, including its 13 input volumes. Run the commands from the artifact directory. The original 7,197-byte BCI Competition IV label archive is downloaded from the organizers. For offline use, give materialize_inputs.py its local path with --labels. The two checks before installation need only Python's standard library.

~~~sh
cd artifact
python -B materialize_inputs.py
python -B reproduce.py verify-inputs
python -m pip install -r requirements.txt
python -B reproduce.py verify
python -B reproduce.py known-prior --out run-witness
python -B reproduce.py scores --out run-scores
python -B reproduce.py table-iv --out run-policies --jobs 4
python -B reproduce.py historical --out run-cohorts
python -B reproduce.py signal-controls --out run-controls
~~~

Numerical replay uses Python 3.10.11 and NumPy 2.2.6. Choose a new output directory for each calculation. The optional construct command runs the 135,000 simulations from their fixed design. It is independent of the EEG data.

### What is included

The public files materialize 363 inputs. All 273 score, prediction, participant, parameter and summary files match the preparation version byte for byte. Twelve source-score files obtain their BNCI cue column from the original archive. Ninety Dreyer checkpoints have portable directory placeholders. Their tensor buffers and all other configuration values agree with the frozen originals. The historic and public checkpoint digests, plus the per-model tensor storage digest, are in [checkpoint provenance](artifact/checkpoint_provenance.csv).

Discovery starts from published participant metrics, Dreyer from trial predictions, and the Stieger analyses from frozen trial scores and source calibration snapshots. The signal controls start from the recorded participant effects. The original EEG sites, attributions and the precise transformation of public inputs are listed in [data sources](DATA_SOURCES.md). Raw EEG inference and perturbation of future raw EEG were not executed. The original legacy Platt coefficients remain unavailable.

The [preparation archive](archives/ACDD_ICDE2027_release_candidate.zip) retains the manuscript, its historical manifests and the pre-publication access note as they were at that time. The current public access path and input checks are the ones on this page. Declarations particular to a CMT account are handled by the authors.
