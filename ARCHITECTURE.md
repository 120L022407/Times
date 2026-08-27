# ARCHITECTURE.md

This repository uses Time-Series-Library as the common experimental framework for forecasting research.

## Main Flow

```text
scripts/long_term_forecast/
        ↓
      run.py
        ↓
exp/exp_long_term_forecasting.py
        ↓
 ┌──────────────┬──────────────┐
 ↓              ↓              ↓
data_provider/  models/       utils/
                  ↓
                layers/
```

The generic training/evaluation pipeline should normally remain unchanged when adding or modifying research methods.

## Main Locations

### `models/`

Main forecasting model implementations.

Use for:

- new paper models;
- new research models;
- experimental model variants.

### `layers/`

Reusable modeling components such as:

- attention
- patch modules
- frequency modules
- expert/routing modules
- decomposition
- embeddings

Prefer placing reusable research components here.

### `exp/exp_long_term_forecasting.py`

Shared long-term forecasting:

- train
- validation
- test
- loss/optimizer integration
- prediction evaluation

Avoid adding model-specific branches here when the behavior can live inside the model.

### `data_provider/`

Shared dataset and DataLoader logic.

Normally do not modify this when experimenting with model architectures.

### `run.py`

Shared CLI/configuration entrypoint.

Add new arguments only when an experiment requires them.

### `scripts/long_term_forecast/`

Reproducible experiment configurations.

Keep separate scripts/configurations for important baselines, proposed methods, and ablations.

## Typical Research Change

A model or method experiment normally changes:

```text
models/<Model>.py
layers/<optional_component>.py
run.py                         # only if new parameters are needed
scripts/long_term_forecast/... # experiment configuration
tests/...                      # when useful
```

Usually avoid changing:

```text
data_provider/
exp/exp_long_term_forecasting.py
existing unrelated models
```

## Forecast Interface

Typical encoder input:

```text
[B, seq_len, C]
```

Models should return predictions compatible with:

```text
[B, pred_len, c_out]
```

Follow the actual interface used by the current local repository.

## Experiment Types

The framework supports three common research patterns:

```text
1. Baseline
   Existing model with standard configuration.

2. Reproduction
   Implement a published method as faithfully as practical.

3. Research variant
   Modify/combine modules, losses, or ideas for a new experiment.
```

Do not overwrite a baseline when creating a research variant.

Use clear experiment/model names so results remain traceable.

## Benchmark Defaults

Primary datasets:

```text
ETTh1
ETTh2
Traffic
```

Common setting:

```text
96 -> 96
```

ETTh1 / ETTh2 single-target experiments commonly use:

```text
features=MS
target=OT
```

These are defaults, not hard constraints. The research question may require different settings.
