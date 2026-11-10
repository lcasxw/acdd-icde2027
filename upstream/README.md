# Archived EEG-to-score path

This directory contains the historical Stieger export and participant-major inference code, both decoder definitions, the training and preprocessing settings, the Stieger evaluation plan and the recorded lock files that can be distributed without private host paths. Two deployment configuration copies replace host-specific directory strings. [Source identities and substitutions](source_digests.json) · [Dreyer-to-Stieger checkpoint mapping](checkpoint_mapping.csv) · [Original data providers](../DATA_SOURCES.md)

The checkpoint map covers the 90 registered Dreyer/Stieger model units. It identifies both the original byte digest and the public weight copy. The protected score replay uses 30 raw-BN streams. The original verification code requires the historical file digests. A public checkpoint with identical tensor buffers but different directory metadata will have a different whole-file digest. The original locks were retained. This directory documents the upstream implementation; gates were not relaxed and no raw inference, new training, or test on future raw EEG was run for this release.

The score-level commands that reproduce the results in the paper are in the [artifact guide](../artifact/README.md). Discovery is available at the participant-metric level and the signal controls start from historical participant effects.
