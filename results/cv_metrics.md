# CV metrics (computed from `results/`)

## Headline

| Metric | Value |
|---|---:|
| Official When2Call normalized accuracy | 71.03% |
| Official When2Call macro F1 | 52.01% |
| Official hallucination rate | 8.14% |
| Official hallucination reduction (Tool-SFT → Reliable) | 32.56% |
| Official relative hallucination reduction | 80.0% |
| Base exact tool-call accuracy | 0.0% |
| Reliable Tool-SFT exact tool-call accuracy | 86.1% |
| Absolute improvement | 86.1% |
| Relative improvement | n/a |
| Held-out (unseen) function accuracy | 78.3% |
| Tool-SFT false tool-call rate | 89.9% |
| Reliable Tool-SFT false tool-call rate | 11.2% |
| False tool-call reduction (Tool-SFT → Reliable) | 78.7% |
| Relative false tool-call reduction | 87.5% |
| Decision-accuracy gain (Tool-SFT → Reliable) | 24.8% |

## Official When2Call MCQ (3,652 examples)

| Metric | Base | Tool-SFT | Reliable Tool-SFT |
|---|---:|---:|---:|
| Accuracy | 47.21% | 43.81% | 69.17% |
| Length-normalized accuracy | 52.79% | 49.81% | 71.03% |
| Macro F1 | 30.63% | 26.26% | 52.01% |
| Hallucination rate | 24.42% | 40.70% | 8.14% |
| Tool-call precision | 49.36% | 46.37% | 68.29% |
| Tool-call recall | 94.83% | 95.29% | 79.00% |

## Generation-based comparison

| Metric | Base | Tool-SFT | Reliable Tool-SFT |
|---|---:|---:|---:|
| Exact tool-call match | 0.0% | 94.6% | 86.1% |
| Function-name accuracy | 0.0% | 95.2% | 87.3% |
| Argument accuracy | 0.0% | 94.9% | 86.7% |
| JSON validity | 100.0% | 99.2% | 100.0% |
| Held-out function EM | 0.0% | 96.4% | 78.3% |
| When2Call decision accuracy | 23.2% | 37.3% | 62.1% |
| When2Call macro F1 | 16.7% | 17.9% | 47.4% |
| Tool-call precision | n/a | 37.2% | 70.2% |
| Tool-call recall | 0.0% | 97.2% | 48.2% |
| When2Call false tool-call rate | 0.0% | 89.9% | 11.2% |
| Missing-info accuracy | 66.2% | 10.0% | 73.4% |
| Cannot-answer accuracy | 11.3% | 0.0% | 66.7% |

| Count | Base | Tool-SFT | Reliable Tool-SFT |
|---|---:|---:|---:|
| Tool-call eval examples | 249 | 249 | 249 |
| When2Call eval examples | 1200 | 1200 | 1200 |

---

Numbers are read directly from `results/*.json`. Base's 0% false-call rate is intentionally not used as a reliability headline because Base almost never calls tools; the meaningful negative-supervision comparison is Tool-SFT → Reliable Tool-SFT. GPU model, wall-clock training time and peak memory should be taken from the run log when not present in run_config.json.
