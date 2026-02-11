# DataChat Demo Playbook (Persona-Based)

This playbook defines repeatable demo environments per persona. Each persona has a
curated dataset, DataPoints, and optional workspace context to keep demos focused
and reliable for clients, execs, or funding sessions.

---

## Personas and Demo Goals

### Sarah (Analyst)

- **Goal:** Fast answers, minimal setup, clear tables.
- **Story:** "Ask in plain English and get a correct, quick result."
- **Level coverage:** Level 1 (schema-aware) + Level 2 (business context).

### Marcus (Data Engineer)

- **Goal:** Consistency and control with executable metrics.
- **Story:** "Metric definitions are standardized and reusable."
- **Level coverage:** Level 3 (SQL templates).

### Priya (Platform Lead)

- **Goal:** Traceability, governance, and operational clarity.
- **Story:** "Every answer is auditable, and sync is explicit."
- **Level coverage:** Level 4/5 signals (audit trail, dependencies).

### James (Executive)

- **Goal:** Trustworthy numbers with evidence and confidence cues.
- **Story:** "Answers are board-ready with clear provenance."
- **Level coverage:** Level 2+ with confidence + sources.

---

## Demo Assets Layout (Recommended)

```text
datapoints/
  demo/
    analyst/        # Level 1-2 DataPoints
    engineer/       # Level 3 templates
    platform/       # Audit/ops focused metadata
    executive/      # Executive-friendly naming

scripts/
  demo_seed.sql     # Base demo database
  demo_seed_*.sql   # Persona-specific variants (optional)

workspace_demo/
  analyst/          # Small docs + queries
  engineer/         # dbt models + SQL
  platform/         # ops docs + runbooks
  executive/        # metrics summaries
```

---

## Demo Setup Command (Spec)

Use the CLI entrypoint to prepare a persona-specific demo:

```text
datachat demo --persona analyst --reset
```

### Expected behavior

- Seeds demo database using `SYSTEM_DATABASE_URL` (`scripts/demo_seed.sql` + persona variant if present).
- Loads DataPoints from `datapoints/demo/<persona>/`.
- Optionally indexes workspace from `workspace_demo/<persona>/`.
- Prints a short “demo script” of suggested questions.

### Flags

```text
--persona <analyst|engineer|platform|executive>
--reset          # Drops and re-seeds demo data
--no-workspace   # Skip workspace indexing
--open           # (Optional) open UI after setup
```

---

## Suggested Demo Questions

### Analyst

- "What was revenue last month?"
- "Top 5 customers by orders"
- "Orders by day last week"

### Engineer

- "Show monthly recurring revenue"
- "Revenue by plan tier"
- "Which template was used for this metric?"

### Platform Lead

- "Show the audit trail for this answer"
- "What was indexed in the last sync?"
- "Which files were skipped and why?"

### Executive

- "What was Q4 revenue and how confident are we?"
- "Compare revenue vs last quarter"
- "What’s the source for this number?"

---

## Demo Session Flow (15-20 minutes)

1. Run demo setup for the persona.
2. Start backend + frontend (if not already running).
3. Ask 2-3 scripted questions.
4. Take 1-2 live questions.
5. Close with a summary + next steps.

---

## Implementation Notes

- Keep demos deterministic: limit randomness and use seeded data.
- Ensure demo data is small but realistic (10-100 rows per table).
- Favor queries that highlight confidence + sources.
- For execs, avoid raw SQL unless asked.
