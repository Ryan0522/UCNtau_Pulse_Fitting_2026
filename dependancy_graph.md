Program pipeline:
**config JSON → run list / paths / settings → ROOT file loader → per-run per-segment generate `vector<Hit>` → analysis window selection → histogram building → seed finding → fitting → result packaging → CSV output**

-----
Dependency:
```
AnalysisConfig
    ↓
RootRunLoader
    ↓
BatchAnalysisRunner
    ↓
Analysispipeline
    ↓
WindowedPulseProcessor
    ↓
GreedyLRTFitter
    ↓
Likelihood + Pulse Template
```
And with types `SignalTypes` and `FitTypes` accessible everywhere.