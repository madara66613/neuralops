# Delivery milestones

| Milestone | Branch scope | Exit evidence |
| --- | --- | --- |
| M0 | Research, architecture, scaffold | Dataset decision, leakage contract, configs, CI smoke check |
| M1 | Data and baseline | Deterministic preparation, data-quality report, baseline artifact and tests |
| M2 | GRU modeling | Reproducible trainer, validation policy, evaluation and checkpoint tests |
| M3 | Synthetic multi-task | OpsForge generator, isolated category/severity heads and report |
| M4 | Inference API | Predictor, safe FastAPI contract, structured logging and API tests |
| M5 | Operator console | React UI, accessible states, frontend tests, Playwright evidence |
| M6 | Release engineering | Docker Compose, benchmarks, final reports, CI, release tag |

Each milestone is developed on an `agent/m*` branch, pushed, reviewed through a pull request, and merged before the next milestone begins. Machine-generated metrics are committed only when their exact provenance is retained.

