# Intent Gathering Plan (PR-003)

## Goals

- Detect exit requests reliably in CLI/chat.
- Detect out-of-scope prompts and respond politely.
- Reduce clarification loops by preserving intent across turns.
- Keep the system safe (no unintended actions).

## Scope

- Applies to CLI, REST, and Web UI.
- No new tools or model providers required.

## Phase 1: Intent Gate (Fast, Cheap)

- Add a lightweight intent classifier before the main pipeline.
- Categories: data_query, setup_help, exit, out_of_scope, small_talk.
- If exit: terminate session immediately.
- If out_of_scope: respond with a short boundary message + examples.

## Phase 2: Conversation State

- Track a minimal "intent summary" in memory:
  - last user goal
  - last clarifying question asked
  - table/column hints
- Merge short follow-up answers into the summary.

## Phase 3: Clarification Strategy

- Limit clarifications to 2-3 turns.
- If still ambiguous, provide a short list of likely tables.
- Accept "any table" by selecting the most relevant table from schema tokens.

## Phase 4: UI/CLI Improvements

- Show clarifying questions as quick actions.
- In CLI, provide a one-line prompt to answer clarifications.
- Add a gentle reminder of how to exit the chat.

## Acceptance Criteria

- "Ok I'm done" exits the CLI without LLM calls.
- Out-of-scope requests get a friendly boundary response.
- Clarification loops stop within 3 turns.
- Intent summary consistently preserves table/column hints.

## Risks and Mitigations

- False positives on exit intent.
  - Mitigation: require explicit intent and confirm once if ambiguous.
- Over-filtering legitimate queries.
  - Mitigation: prefer data_query when unsure.
