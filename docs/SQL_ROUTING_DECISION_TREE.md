# SQL Routing and Visualization Decision Tree

This document summarizes how DataChat routes SQL generation and visualization selection after the latest runtime updates.

## SQL Routing Overview

```mermaid
flowchart TD
  A["User Query"] --> B{"Catalog operation?<br/>(list tables/columns/row count/sample)"}
  B -->|Yes| C["Run deterministic catalog SQL"]
  B -->|No| D{"Deterministic metric fallback match?<br/>(revenue-margin-waste, stockout-risk)"}
  D -->|Yes| E["Generate deterministic SQL (no LLM)"]
  D -->|No| F{"Table Resolver enabled?"}
  F -->|No| G["Build SQL prompt from schema/business/live context"]
  F -->|Yes| H["Mini LLM resolves likely tables + columns"]
  H --> I{"Low confidence + ambiguity?"}
  I -->|Yes| J["Return targeted clarification question"]
  I -->|No| K{"Fallback match with resolved tables?"}
  K -->|Yes| E
  K -->|No| G
  G --> L["Main SQL generation LLM"]
  L --> M{"Malformed SQL JSON?"}
  M -->|Yes| N["Formatter fallback model repairs JSON"]
  M -->|No| O["Validate SQL"]
  N --> O
  O --> P{"Validation errors?"}
  P -->|Yes| Q["SQL self-correction loop"]
  P -->|No| R["Execute SQL"]
  Q --> R
```

## Visualization Hint Overview

```mermaid
flowchart TD
  V0["Query Result"] --> V1{"No rows / single scalar?"}
  V1 -->|Yes| V2["Visualization: none"]
  V1 -->|No| V3{"User explicitly requested chart type?"}
  V3 -->|Yes| V4["Respect request if shape is valid"]
  V3 -->|No| V5{"Temporal axis + numeric metric?"}
  V5 -->|Yes| V6["Visualization: line_chart"]
  V5 -->|No| V7{"Category + metric?"}
  V7 -->|Yes| V8{"Share/distribution query + small cardinality + positive values?"}
  V8 -->|Yes| V9["Visualization: pie_chart"]
  V8 -->|No| V10["Visualization: bar_chart"]
  V7 -->|No| V11{"Numeric pair analysis + explicit scatter intent?"}
  V11 -->|Yes| V12["Visualization: scatter"]
  V11 -->|No| V13["Visualization: table"]
```
