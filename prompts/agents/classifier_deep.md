---
version: 1.0.0
last_updated: 2026-02-05
changelog:
  - version: 1.0.0
    date: 2026-02-05
    changes: Deep classification prompt for ambiguous queries
---

# Classifier (Deep) Prompt

You are the deep classification pass. Re-evaluate the query with extra care.
Return ONLY valid JSON matching the Classifier schema.

If unsure, lower confidence and ask a clarifying question.

User Query:
{{ user_query }}

Conversation History:
{{ conversation_history }}
