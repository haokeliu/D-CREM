# Contributing

Contributions are welcome through issues and pull requests.

## Development setup

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Protocol invariants

Changes must preserve the Protocol-v2 boundaries:

- fit preprocessing, model parameters, and calibration on the training fold;
- select hyperparameters and K on validation data only;
- evaluate the locked configuration on test data once;
- reuse identical sample indices across methods for each
  `(dataset, known_ratio, seed)`;
- keep public targets in `Q x N` shape with values in `{-1, +1}`;
- keep features in `N x d` shape;
- preserve the stored `P` sign convention documented in the protocol.

Please add or update regression tests for behavior changes. Do not commit
datasets, model weights, caches, local paths, credentials, or generated
experiment artifacts.

