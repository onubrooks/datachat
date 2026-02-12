# Progressive Latency Benchmark

- Timestamp (UTC): 2026-02-12T12:03:51.143865+00:00
- Iterations: 1
- Queries per iteration: 3

## Query Suite
- `List all grocery stores`
- `What is total grocery revenue?`
- `Show gross margin by category`

## Stage Summary

| Stage | Mean Latency (ms) | Median (ms) | P95 (ms) | Mean LLM Calls | Success Rate | Clarification Rate | SQL Answer Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 9547.1 | 9837.5 | 13714.7 | 2.33 | 66.67% | 66.67% | 0.00% |
| stage1_sql_two_stage | 7409.6 | 8332.5 | 8655.7 | 2.00 | 66.67% | 66.67% | 0.00% |
| stage2_prompt_budget | 39924.4 | 47688.8 | 62774.0 | 4.00 | 66.67% | 0.00% | 66.67% |
| stage3_simple_sql_synthesis_off | 20434.4 | 7937.1 | 44536.4 | 2.67 | 100.00% | 66.67% | 33.33% |
| stage4_classifier_deep_gate | 8598.7 | 7907.2 | 12115.6 | 2.33 | 33.33% | 33.33% | 0.00% |
| stage5_selective_tool_planner | 25038.2 | 6193.2 | 60280.4 | 2.00 | 100.00% | 66.67% | 33.33% |
| stage6_schema_snapshot_cache | 34849.6 | 43607.9 | 54903.6 | 2.33 | 100.00% | 33.33% | 66.67% |

## Per-Agent Mean Latency (ms)

### baseline
- `classifier`: 6283.2ms
- `tool_planner`: 2061.1ms
- `context`: 1202.5ms
- `intent_gate`: 0.3ms

### stage1_sql_two_stage
- `classifier`: 4554.7ms
- `tool_planner`: 2113.6ms
- `context`: 741.1ms
- `intent_gate`: 0.2ms

### stage2_prompt_budget
- `sql`: 33315.3ms
- `classifier`: 8571.7ms
- `executor`: 4058.5ms
- `response_synthesis`: 3045.5ms
- `tool_planner`: 2387.6ms
- `context`: 2016.5ms
- `validator`: 2.8ms
- `intent_gate`: 0.6ms

### stage3_simple_sql_synthesis_off
- `sql`: 35504.3ms
- `executor`: 4929.6ms
- `response_synthesis`: 4028.6ms
- `classifier`: 2435.6ms
- `tool_planner`: 2134.2ms
- `context`: 1042.3ms
- `validator`: 3.3ms
- `intent_gate`: 0.4ms

### stage4_classifier_deep_gate
- `classifier`: 4862.8ms
- `context`: 1972.3ms
- `tool_planner`: 1763.2ms
- `intent_gate`: 0.4ms

### stage5_selective_tool_planner
- `sql`: 40619.4ms
- `executor`: 9452.1ms
- `classifier`: 5632.5ms
- `response_synthesis`: 3437.3ms
- `context`: 1567.4ms
- `validator`: 5.0ms
- `intent_gate`: 0.3ms

### stage6_schema_snapshot_cache
- `sql`: 34836.7ms
- `classifier`: 5816.8ms
- `executor`: 5691.8ms
- `response_synthesis`: 3444.4ms
- `context`: 863.7ms
- `validator`: 2.5ms
- `intent_gate`: 0.3ms
