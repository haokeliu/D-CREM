# Generated cache directory

This directory is intentionally empty in the source release.

Tabular Protocol-v2 caches are generated from the ARFF/XML sources. Frozen
image experiments expect MATLAB files such as `voc2007_resnet50.mat` with:

- `X`: an `N x d` feature matrix;
- `Y`: an `N x Q` binary multi-label matrix;
- `label_names`: optional label names.

Use `scripts/run_m3_features.py --help` for the supported image feature
extraction workflow. Downloaded images, weights, and generated feature files
remain subject to their original licenses and should not be committed here.

