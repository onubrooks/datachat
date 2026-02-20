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

Current defaults:

- `datachat dp add`: strict contracts enabled by default
- `datachat dp sync`: strict contracts enabled by default
- `datachat dp lint`: strict contracts enabled by default

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

## API and sync enforcement

- `POST /api/v1/datapoints` and `PUT /api/v1/datapoints/{id}` reject contract violations.
- Pending approval endpoints (`approve`, `bulk-approve`) apply strict contract validation.
- Background sync orchestrator runs with strict contract checks and fails the sync job when files violate contracts.
- Bundled `datapoints/demo/` files are exempt from strict advisory-field escalation in sync mode
  so default demo deployments do not fail on non-critical metadata gaps.

## Roadmap

Next increment wires thresholded contract quality metrics into CI gates and release dashboards.
