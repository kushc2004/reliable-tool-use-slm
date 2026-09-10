# CV metrics (computed from `results/`)

## Headline

| Metric | Value |
|---|---:|
| Base exact tool-call accuracy | 30.0% |
| Reliable Tool-SFT exact tool-call accuracy | 81.0% |
| Absolute improvement | 51.0% |
| Relative improvement | +170.0% |
| Held-out (unseen) function accuracy | 78.0% |
| Base false tool-call rate | 22.5% |
| Reliable Tool-SFT false tool-call rate | 57.5% |
| Absolute reduction in false tool calls | 35.0% |

## Full comparison

| Metric | Base | Tool-SFT | Reliable Tool-SFT |
|---|---:|---:|---:|
| Exact tool-call match | 30.0% | 100.0% | 81.0% |
| Function-name accuracy | 30.0% | 100.0% | 81.0% |
| Argument accuracy | 30.0% | 100.0% | 81.0% |
| JSON validity | 100.0% | 100.0% | 100.0% |
| Held-out function EM | 32.0% | 100.0% | 78.0% |
| When2Call decision accuracy | 72.5% | 25.0% | 35.6% |
| When2Call false tool-call rate | 22.5% | 66.7% | 57.5% |
| Missing-info accuracy | 57.5% | 0.0% | 17.5% |

| Count | Base | Tool-SFT | Reliable Tool-SFT |
|---|---:|---:|---:|
| Tool-call eval examples | 150 | 150 | 150 |
| When2Call eval examples | 160 | 160 | 160 |

---

Numbers are read directly from `results/*.json`. Fill in GPU model, wall-clock training time and peak memory from your own run log — those are not recorded by these scripts.
