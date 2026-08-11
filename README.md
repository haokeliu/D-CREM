# D-CREM

D-CREM is a deep extension of Classifier-induced Reciprocal Points for
multi-label open-set recognition (MLOSR). This repository also contains a
leakage-free Python reproduction of CREM and Python implementations of the
SLAN and MUENL-F baselines.

The release follows Protocol v2:

- train / validation / test samples are split 40% / 10% / 50%;
- feature selection, standardization, fitting, and calibration use training
  data only;
- hyperparameters and top-K are selected on validation data;
- the locked model is evaluated on the test split once;
- every method reuses the same split for a given
  `(dataset, known_ratio, seed)`.

The mathematical definitions and full protocol are documented in
[docs/protocol-v2.zh-CN.md](docs/protocol-v2.zh-CN.md).

## Installation

The reference environment uses Python 3.9 and NumPy 1.26.x.

```bash
conda create -n dcrem python=3.9 -y
conda activate dcrem
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For image encoders and analysis figures:

```bash
python -m pip install -e ".[vision,analysis,dev]"
```

PyTorch wheels are platform-specific. If the default installation does not
match your CUDA runtime, install the appropriate PyTorch build first and then
run the editable install command.

## Verify the release

The complete test suite is data-free:

```bash
python -m pytest
```

Run a two-epoch synthetic D-CREM smoke test:

```bash
python examples/synthetic_smoke.py
```

## Data

Third-party datasets, pretrained weights, cached features, and experiment
outputs are intentionally not redistributed. See
[datasets_raw/README.md](datasets_raw/README.md) for the expected tabular data
layout and [cache/README.md](cache/README.md) for image-feature caches.

For tabular experiments, put each ARFF/XML pair in `datasets_raw/`, or point
the loader to another directory:

```bash
# Linux/macOS
export CREM_DATA_DIR=/path/to/mulan-datasets

# PowerShell
$env:CREM_DATA_DIR = "D:\path\to\mulan-datasets"
```

## Common commands

Run one CREM experiment:

```bash
python run_crem.py --dataset enron --known_ratio 0.5 --seed 0 --standardize
```

Run one paper-core D-CREM experiment in Mode B:

```bash
python dcrem/scripts/train.py --dataset enron --mode B --known-ratio 0.5 \
  --seed 0 --classifier-induced-reciprocal --no-correlation \
  --lamda2 0 --lamda3 0 --alpha 0 --gamma-div 0 --no-warmup
```

Run the formal experiment matrix or baselines:

```bash
python scripts/run_phase3.py --help
python scripts/run_slan.py --help
python scripts/run_muenl_f.py --help
```

Refresh the structured result summary after a batch:

```bash
python scripts/build_results_report.py
```

The summary command writes JSON by default. It creates a Markdown report only
when an explicit `--markdown` output path is supplied.

## Repository layout

```text
crem/       CREM kernel method and Protocol-v2 data pipeline
dcrem/      D-CREM models, optimization, evaluation, and tests
baselines/  SLAN and MUENL-F Python implementations
scripts/    Formal experiment, baseline, analysis, and reporting entry points
docs/       Locked Protocol-v2 mathematical and evaluation conventions
examples/   Data-free runnable examples
```

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). Please cite the
associated D-CREM paper when its final bibliographic record becomes available,
and cite the original CREM, SLAN, and MUENL-F papers when using those parts.

## License

Original code in this release is available under the MIT License. See
[LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

