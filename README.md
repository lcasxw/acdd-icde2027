# Access-Conditioned Decision Decomposition for Auditing Cross-Dataset Motor-Imagery EEG Decoders

Research artifact for the ICDE 2027 Experiment, Analysis, and Benchmark submission. The preserved preparation version is ICDE-2027-EAB-submission-prep-4.

- [Paper PDF](artifact/candidate.pdf) · [LaTeX source archive](submission/manuscript_source.zip) · [Submission text](submission/submission-text.txt)
- [Reproduction guide](artifact/README.md) · [Source and settings](artifact/) · [Input inventory](artifact/required_inputs.csv)
- [Original 72-file preparation archive](archives/ACDD_ICDE2027_release_candidate.zip) · [Original content hashes](artifact/MANIFEST.json)

## First checks

From the artifact directory run the following commands. The package-integrity check uses the standard library. The exact witness enumerates the 18 threshold splits in Section IV-A.

~~~sh
python -B reproduce.py verify
python -B reproduce.py known-prior --out run-witness
~~~

The preparation archive holds the paper, software, configurations and reference estimates. Its inventory lists the frozen inputs needed by the score, calibration, historical-cohort and signal-control paths. A hash list alone does not provide those inputs. The paper distinguishes replay on frozen scores from raw EEG inference. The historical Platt coefficient snapshots were unavailable. The preserved results specify which checks were actually run.
