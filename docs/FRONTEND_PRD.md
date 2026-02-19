# Frontend Product Requirements Document

**Version:** 1.1  
**Last Updated:** February 19, 2026

This document describes the frontend architecture, current state, and roadmap for the DataChat web UI.

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Framework | Next.js 14 (App Router) |
| Styling | Tailwind CSS |
| State Management | Zustand |
| API Layer | REST + WebSocket |
| Testing | Jest + React Testing Library |
| Charts | Native SVG (no external library) |

---

## Current State: What Has Been Done ✅

### Core Chat Experience

| Feature | Description | Status |
|---------|-------------|--------|
| Chat Interface | Full-height chat with message list and input | ✅ Implemented |
| Real-time Streaming | WebSocket integration for agent updates | ✅ Implemented |
| Message Display | User/assistant messages with formatting | ✅ Implemented |
| SQL Code Blocks | Syntax display with copy button | ✅ Implemented |
| Data Tables | Result tables with expand/collapse | ✅ Implemented |
| Table Pagination | 50 rows/page with navigation controls | ✅ Implemented |
| Visualizations | Bar, line, scatter, pie charts (SVG) | ✅ Implemented |
| Clarifying Questions | Interactive question prompts | ✅ Implemented |
| Multi-Question Support | Sub-answers with Q1/Q2 selector | ✅ Implemented |
| Conversation Persistence | localStorage backup with data recovery | ✅ Implemented |
| Error Recovery | Retry button with error categorization | ✅ Implemented |
| Conversation History Sidebar | Resume prior local sessions | ✅ Implemented |
| Schema Explorer Sidebar | Browse tables/columns with search | ✅ Implemented |
| Query Templates | Quick-action buttons for common prompts | ✅ Implemented |
| Chart Interaction | Tooltips, zoom controls, legend toggles | ✅ Implemented |
| Chart Configuration | Per-chart axis + display settings panel | ✅ Implemented |
| Accessibility Labels | ARIA labels, dialog semantics, live regions | ✅ Implemented |
| Keyboard Navigation | Tabs + global shortcuts + modal focus handling | ✅ Implemented |

### Database Management

| Feature | Description | Status |
|---------|-------------|--------|
| Connection CRUD | Add, edit, delete database connections | ✅ Implemented |
| Connection Selector | Dropdown to switch databases | ✅ Implemented |
| Profiling Workflow | Profile database with progress tracking | ✅ Implemented |
| DataPoint Approval | Review pending DataPoints | ✅ Implemented |
| Bulk Approve | Approve all pending DataPoints | ✅ Implemented |

### Observability

| Feature | Description | Status |
|---------|-------------|--------|
| Agent Status | Real-time agent progress display | ✅ Implemented |
| Agent Timing Breakdown | Per-agent latency metrics | ✅ Implemented |
| Thinking Notes | Live reasoning stream | ✅ Implemented |
| LLM Call Counter | Track LLM usage per query | ✅ Implemented |
| Decision Trace | Query routing decisions | ✅ Implemented |

### Tool Integration

| Feature | Description | Status |
|---------|-------------|--------|
| Tool Approval Modal | Approve/reject tool executions | ✅ Implemented |
| Cost Estimates | Show expected LLM calls for tools | ✅ Implemented |

### Settings

| Feature | Description | Status |
|---------|-------------|--------|
| Result Layout Mode | Stacked vs tabbed view | ✅ Implemented |
| Agent Timing Toggle | Show/hide timing breakdown | ✅ Implemented |
| Live Reasoning Toggle | Show/hide thinking notes | ✅ Implemented |
| Simple SQL Synthesis | Toggle for simple SQL responses | ✅ Implemented |

---

## Needs Improvement ⚠️

### P2: Discovery Friction

| Issue | Impact | Recommendation |
|-------|--------|----------------|
| ✅ **Schema Explorer Added** | Users can inspect tables/columns directly | Collapsible schema browser sidebar with search |
| ✅ **Conversation History Added** | Users can resume past sessions | Collapsible conversation list sidebar with local restore |
| ✅ **Query Templates Added** | Faster repeated workflows | Quick-action buttons for common query patterns |

### P3: Visualization Polish

| Issue | Impact | Recommendation |
|-------|--------|----------------|
| ✅ **Chart Interaction Added** | Users can inspect datapoints and control visual density | Tooltips + zoom + legend toggles across chart types |
| ✅ **Chart Configuration Added** | Users can adjust chart behavior without re-querying | Per-chart settings panel (axes, max points/slices, grid/legend) |

### P4: Accessibility

| Issue | Impact | Recommendation |
|-------|--------|----------------|
| ✅ **ARIA Coverage Expanded** | Better screen reader support across chat/sidebar/modal/chart surfaces | Region labels, control labels, dialog roles, status/live semantics |
| ✅ **Keyboard Navigation Added** | Faster non-pointer workflows | Tab keyboard navigation + global shortcuts + focus restoration |

**Implemented keyboard shortcuts**
- `Ctrl/Cmd + K`: Focus chat input
- `Ctrl/Cmd + H`: Toggle conversation history sidebar
- `Ctrl/Cmd + /`: Open/close shortcut reference modal
- `/`: Focus chat input (when not typing in an input)
- `Esc`: Close open modal and restore chat input focus

---

## Should Add ➕

### P1: Essential Features

| Feature | Description | Effort |
|---------|-------------|--------|
| **Conversation Sidebar** | List of past conversations with search | 16h |
| **Schema Browser** | Interactive table/column explorer | 12h |

### P2: Productivity Features

| Feature | Description | Effort |
|---------|-------------|--------|
| **Query Templates** | Pre-defined patterns (Top N, Trends) | 8h |
| **SQL Editor Mode** | Edit SQL before execution | 12h |
| **Keyboard Shortcuts** | Cmd+K, Cmd+H, Cmd+/, Esc | 4h |
| **Dark Mode Toggle** | Manual theme override | 2h |

### P3: Export & Sharing

| Feature | Description | Effort |
|---------|-------------|--------|
| **Export CSV** | Download result data | 2h (already implemented) |
| **Export JSON** | JSON format download | 1h |
| **Export Markdown** | Copy table as markdown | 2h |
| **Share Link** | Deep link to query result | 8h |

### P4: Feedback Loop

| Feature | Description | Effort |
|---------|-------------|--------|
| **Answer Feedback** | Thumbs up/down on responses | 4h |
| **Issue Reporting** | Report problems with context | 6h |
| **Improvement Suggestions** | UI to suggest DataPoint improvements | 8h |

---

## Should Remove 🗑️

### Dead Code

| Location | Issue | Action |
|----------|-------|--------|
| `loadingUx.ts` | Multiple modes unused | Consolidate to single mode |
| Redundant job state | Multiple similar state variables | Consolidate into single `jobs` object |

### Technical Debt

| Issue | Impact | Action |
|-------|--------|--------|
| No React Query | Manual loading/error states | Migrate to React Query for API state |
| No error boundaries | Crashes kill whole app | Add error boundaries with recovery UI |
| Inline chart rendering | Hard to maintain | Extract to separate components |

---

## Architecture Recommendations

### State Management

Current: Zustand with manual API calls

**Recommended: Add React Query**

```typescript
// Before
const [connections, setConnections] = useState([]);
const [isLoading, setIsLoading] = useState(false);
const [error, setError] = useState(null);

useEffect(() => {
  setIsLoading(true);
  api.listDatabases()
    .then(setConnections)
    .catch(setError)
    .finally(() => setIsLoading(false));
}, []);

// After
const { data: connections, isLoading, error } = useQuery({
  queryKey: ['connections'],
  queryFn: () => api.listDatabases(),
});
```

### Conversation Persistence

✅ **Implemented** - Chat store uses Zustand's persist middleware with localStorage.

```typescript
// In chat store - currently implemented
export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      messages: [],
      conversationId: null,
      // ... other state
    }),
    {
      name: 'datachat.chat.session.v1',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        messages: state.messages.slice(-60), // Keep last 60 messages
        conversationId: state.conversationId,
        sessionSummary: state.sessionSummary,
        sessionState: state.sessionState,
      }),
    }
  )
);
```

**What's persisted:**
- Last 60 messages (compacted)
- SQL queries and results (up to 50 rows)
- Visualization hints and metadata
- Sources and evidence
- Agent timing metrics
- Sub-answers (up to 5)
- Conversation ID and session state

**Future enhancement:** Backend persistence for cross-device access.

### Error Recovery

✅ **Implemented** - Retry button with error categorization for failed queries.

**Error Categories:**
| Category | Icon | Triggers | Suggestion |
|----------|------|----------|------------|
| Network | Wifi | connection, econnrefused, enotfound, fetch failed | Check internet connection |
| Timeout | Clock | timeout, timed out, deadline exceeded | Simplify query |
| Validation | AlertTriangle | invalid, syntax, required | Check input |
| Database | Database | sql, table, column, schema, query | Rephrase query |
| Unknown | AlertCircle | All other errors | Try again |

**Features:**
- Retry button re-populates input with failed query
- Attempt counter shows retry count
- Contextual suggestions based on error type
- Dismiss button to clear error state
- Error state stored for retry functionality

**Implementation:**
```typescript
const categorizeError = (errorMessage: string) => {
  const lower = errorMessage.toLowerCase();
  if (lower.includes("network") || lower.includes("connection")) {
    return "network";
  }
  if (lower.includes("timeout")) {
    return "timeout";
  }
  // ... more categories
  return "unknown";
};
```

### Component Extraction

Move visualization rendering to dedicated components:

```
frontend/src/components/visualizations/
├── BarChart.tsx
├── LineChart.tsx
├── ScatterChart.tsx
├── PieChart.tsx
├── ChartContainer.tsx
└── types.ts
```

---

## Roadmap

### Sprint 1: Persistence & Discovery (P1)

| Task | Effort | Priority | Status |
|------|--------|----------|--------|
| Add conversation persistence (localStorage) | 8h | P1 | ✅ Done |
| Add table pagination (50 rows/page) | 4h | P1 | ✅ Done |
| Add retry button for errors | 4h | P1 | ✅ Done |
| Add schema browser sidebar | 12h | P1 | ✅ Done |

**Total Remaining: 0h**

### Sprint 2: Productivity (P2)

| Task | Effort | Priority | Status |
|------|--------|----------|--------|
| Add conversation history sidebar | 16h | P2 | ✅ Done |
| Add query templates | 8h | P2 | ✅ Done |
| Add keyboard shortcuts | 4h | P2 | Pending |
| Add dark mode toggle | 2h | P2 | Pending |

**Total Remaining: 6h**

### Sprint 3: Polish (P3)

| Task | Effort | Priority |
|------|--------|----------|
| Add chart tooltips | 4h | P3 |
| Add chart configuration | 6h | P3 |
| Add export JSON/markdown | 3h | P3 |
| Add answer feedback | 4h | P3 |
| Extract chart components | 4h | P3 |
| Add error boundaries | 4h | P3 |

**Total: 25h**

---

## UI/UX Specifications

### Schema Browser

```
┌─────────────────────────────────────────────────────────────┐
│ [Schema Browser ▼]                              [Collapse] │
├─────────────────────────────────────────────────────────────┤
│ 📊 fact_sales                                    1.2M rows │
│    ├── id (BIGINT) - Surrogate key                         │
│    ├── customer_id (BIGINT) - FK to dim_customer           │
│    ├── amount (DECIMAL) - Transaction value                │
│    └── transaction_time (TIMESTAMP) - When occurred         │
│                                                             │
│ 📊 dim_customer                                  50K rows   │
│    ├── customer_id (BIGINT) - PK                           │
│    ├── name (VARCHAR) - Full name                          │
│    └── segment (VARCHAR) - Customer segment                │
│                                                             │
│ 📈 metric_revenue                                          │
│    Calculation: SUM(amount) WHERE status='completed'       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Conversation Sidebar

```
┌─────────────────────────────────────────┐
│ Conversations              [+ New Chat] │
├─────────────────────────────────────────┤
│ 📊 Sales analysis yesterday             │
│    3 queries · 2 min ago                │
│                                         │
│ 📈 Revenue trends last quarter          │
│    5 queries · 2 hours ago              │
│                                         │
│ 🔍 Customer lookup                      │
│    2 queries · yesterday                │
│                                         │
└─────────────────────────────────────────┘
```

### Keyboard Shortcuts

| Shortcut | Action | Context |
|----------|--------|---------|
| `Cmd+K` / `Ctrl+K` | Focus query input | Global |
| `Cmd+H` / `Ctrl+H` | Toggle history sidebar | Global |
| `Cmd+/` / `Ctrl+/` | Toggle schema browser | Global |
| `Cmd+Enter` | Send message | Input focused |
| `Esc` | Cancel streaming / close modal | Contextual |
| `?` | Show keyboard shortcuts | Global |

---

## Metrics & Success Criteria

| Metric | Before | After | Status |
|--------|--------|-------|--------|
| Time to first query | ~30s (setup) | <10s (with saved connection) | ✅ |
| Query recovery rate | 0% (no retry) | 80% (with retry button) | ✅ Implemented |
| Session continuation | 0% (no persistence) | 60% (with localStorage) | ✅ Implemented |
| Schema discovery time | Ask → Wait → Answer | Browse sidebar → Instant | Pending |

---

## Appendix: File Structure

```
frontend/src/
├── app/
│   ├── page.tsx              # Main chat page
│   ├── layout.tsx            # Root layout
│   ├── settings/page.tsx     # Settings page
│   └── databases/page.tsx    # Database manager
├── components/
│   ├── chat/
│   │   ├── ChatInterface.tsx # Main chat component
│   │   ├── Message.tsx       # Message display
│   │   ├── loadingUx.ts      # Loading states
│   │   └── AgentStatus.tsx   # Agent progress
│   ├── system/
│   │   ├── DatabaseManager.tsx
│   │   └── SystemSetup.tsx
│   ├── agents/
│   │   └── AgentStatus.tsx
│   └── ui/
│       ├── button.tsx
│       ├── input.tsx
│       └── card.tsx
├── lib/
│   ├── api.ts                # REST + WebSocket client
│   ├── stores/chat.ts        # Zustand store
│   ├── settings.ts           # User preferences
│   └── utils.ts              # Utilities
└── test/
    └── setup.ts              # Test configuration
```

---

*This document should be updated as features are implemented and requirements evolve.*
