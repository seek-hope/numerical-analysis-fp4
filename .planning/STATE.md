# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-17)

**Core value:** Use numerical analysis to predict, measure, and explain where quantization error goes in a Transformer -- and redesign the measurement protocol when the theory and experiments diverge.

**Current focus:** Phase 1 (Clean Data Split)

## Current Position

Phase: 1 of 5 (Clean Data Split)
Plan: TBD
Status: Ready to plan
Last activity: 2026-05-17 -- Roadmap created

Progress: [                    ] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: TBD
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: (none)
- Trend: N/A

## Accumulated Context

### Decisions

- [Key]: PPL is the wrong metric for testing Theorem 1 -- measured error at matrix output, not after RMSNorm/attention/FFN cascade (per PROJECT.md)
- [Protocol]: Single-pass capture (save clean activations from one forward pass, compute per-matrix ||dy||/||y|| offline) avoids cascading confound from two-pass approach (per PITFALLS.md 3.3)
- [Protocol]: Use round-to-nearest for Theorem 1 test; GPTQ compensation invalidates per-matrix independence assumption (per PITFALLS.md 5.4)
- [Stats]: Bonferroni correction mandatory for 72-matrix correlation test; alpha = 0.05/72 = 0.00069

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 1 blocks all phases (data split is critical prerequisite)
- Phase 5 depends on all prior phases (uses both measurement infrastructure and theorem validation results)

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| (none) | | | |

## Session Continuity

Last session: 2026-05-17 00:00
Stopped at: Roadmap created for 5-phase Measurement Protocol project
Resume file: None
