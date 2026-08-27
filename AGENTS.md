# AGENTS.md

This repository is a research workspace built on Time-Series-Library for time-series forecasting experiments.

Typical tasks include:

- integrating published models;
- transferring modules, losses, or ideas from papers into existing models;
- developing new methods;
- running baselines, ablations, and comparison experiments;
- adapting methods to custom forecasting tasks.

## Before Editing

Read only what is relevant:

- `ARCHITECTURE.md`
- the target model or related existing models
- relevant experiment scripts
- the paper and official code when a paper method is involved

Avoid scanning or refactoring unrelated parts of the repository.

## Framework Rules

Preserve the existing TSLib experiment pipeline.

Prefer:

- models in `models/`
- reusable components in `layers/`
- existing `run.py`
- existing `long_term_forecast` train/validation/test pipeline
- experiment scripts under `scripts/long_term_forecast/`

Do not create a separate Trainer, Dataset, Evaluator, or training entrypoint unless the research task truly requires a different data/training paradigm.

Avoid model-name-specific logic in generic training code.

Follow the model registration mechanism used by the current local repository.

## Model Contract

Forecasting models should normally remain compatible with:

```python
forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
```

and produce predictions compatible with:

```text
[B, pred_len, c_out]
```

Prefer adapting new methods inside the model/layers rather than changing the generic pipeline.

## Research Changes

When using a paper:

- distinguish the paper's core method from its original training infrastructure;
- use the paper and official code as references;
- preserve essential algorithmic behavior when claiming reproduction;
- record meaningful deviations from the original method.

Research exploration is allowed.

It is acceptable to:

- transplant only one module from a paper;
- combine ideas from multiple methods;
- change losses, routing, frequency modules, embeddings, or heads;
- modify an existing baseline for ablation or method development.

Such experiments must be clearly named and must not silently overwrite the original baseline.

## Experiment Fairness

For comparison experiments, keep the following consistent unless the experiment explicitly studies them:

- dataset split
- input/output length
- target definition
- preprocessing/scaling
- evaluation metrics
- evaluation mask
- training budget where practical

Do not use validation or test data to fit normalization or other training statistics.

Keep original baselines available.

## Main Benchmarks

Primary public datasets:

- ETTh1
- ETTh2
- Traffic

Default forecasting setting:

```text
seq_len   = 96
label_len = 48
pred_len  = 96
```

For ETTh1 / ETTh2 single-target experiments:

```text
features = MS
target   = OT
```

Other settings are allowed when required by the research question.

## Testing

Local development should use lightweight CPU or synthetic tests.

For new or substantially modified models, check at least:

- construction
- forward
- output shape
- backward
- batch size 1
- CPU execution
- important method-specific dimensions

Do not run long training locally unless explicitly requested.

Full experiments are normally run on the server.

## Change Discipline

Prefer the smallest change that answers the research question.

Do not:

- refactor unrelated code;
- modify unrelated baselines;
- duplicate common logic;
- hard-code dataset-specific behavior inside general models when avoidable.

New experimental variants should have clear names and reproducible configurations.

## Completion

After a task, report concisely:

- changed files;
- what method or experiment was implemented;
- important implementation choices;
- lightweight test results;
- commands/configurations for server experiments;
- remaining issues requiring real training.
