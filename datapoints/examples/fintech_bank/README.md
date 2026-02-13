# Fintech Bank DataPoints

This folder contains a realistic sample DataPoint bundle for banking/fintech analytics.

Includes:

- Schema DataPoints for customers, accounts, transactions, cards, loans, and FX rates
- Business metric DataPoints for deposits, interest income, default rate, and failed transactions
- Process DataPoints for daily transaction rollups and nightly risk snapshots

Use with:

```bash
datachat dp sync --datapoints-dir datapoints/examples/fintech_bank
```

or load demo end-to-end:

```bash
datachat demo --dataset fintech --reset
```
