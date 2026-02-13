# DataPoint Contracts

DataPoint contracts define the minimum metadata quality needed for reliable DataPoint-driven answers.

## Why this exists

Without consistent metadata, retrieval quality degrades and semantic answers drift.  
Contracts make metadata quality explicit and lintable before runtime.

## Current Contract Rules

Severity model:

- `error`: blocks `datachat dp add`, `datachat dp sync`, and lint script success.
- `warning`: advisory by default; can be escalated via strict mode.

Cross-type expectations:

- `metadata.grain` (warning, strict -> error)
- `metadata.exclusions` (warning, strict -> error)
- `metadata.confidence_notes` (warning, strict -> error)

Type-specific expectations:

1. `Schema`
- must have freshness via top-level `freshness` or `metadata.freshness` (`error`)

2. `Business`
- must define units via `unit` or `metadata.unit`/`metadata.units` (`error`)
- should define freshness via `metadata.freshness` (warning, strict -> error)

3. `Process`
- must define freshness via `data_freshness` or `metadata.freshness` (`error`)

## CLI enforcement

`datachat dp add` and `datachat dp sync` now run contract lint checks after model validation.
`datachat dp lint` runs the same checks without mutating vector store/graph.

Options:

- `--strict-contracts`: escalate advisory gaps to errors
- `--fail-on-contract-warnings`: fail even when only warnings exist

## Standalone lint script

Use:

```bash
python scripts/lint_datapoints.py --path datapoints --recursive
```

Strict mode:

```bash
python scripts/lint_datapoints.py --path datapoints --recursive --strict
```

Fail on warnings:

```bash
python scripts/lint_datapoints.py --path datapoints --recursive --fail-on-warnings
```

## Roadmap

Next increment will apply contract checks to pending DataPoint approval paths and wire thresholded contract metrics into CI gates.
