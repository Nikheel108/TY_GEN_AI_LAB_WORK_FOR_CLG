# Adaptive Parallel LLM Inference Engine using Workload-Aware Scheduling and ML-Based Latency Prediction

**Course:** Generative AI Lab (22311312L) – Practical ESE, Exam Reform Project
**Pathway:** 2 – Research-based project (extends *Efficient LLM Scheduling by Learning to Rank*, NeurIPS 2024)
**Institution:** MIT Academy of Engineering, Alandi (D), Pune – Dept. of CSE (AI & ML)

---

## 1. Overview

When many users send requests to an LLM at the same time, the inference engine must decide **which waiting request runs next** and **how many run in parallel**. FIFO and fixed-concurrency policies ignore how much work each request carries, so queue waiting often dominates the latency users see.

This project trains a **Random Forest latency predictor** on GPU-measured request records and uses its predictions in two schedulers:

- **CAMP** (Context-Aware ML-Predictive scheduling): dispatches the waiting request with the shortest predicted latency.
- **Fair-CAMP**: CAMP plus an aging rule. If the oldest waiting request has waited at least 2 s, it goes first.

Five policies (FIFO, Fixed-K, Adaptive, CAMP, Fair-CAMP) are compared on steady, mixed and bursty workloads.

## 2. Problem Statement

Simple scheduling policies do not use request size or changing load. This causes inefficient waiting and unfair delays. The problem is to build a scheduling approach that uses ML-based latency prediction for workload-aware decisions, while balancing **average latency** against **waiting-time fairness**.

**Target users:** engineers and researchers who operate or study LLM serving systems.
**Input:** a stream of LLM requests (prompt, output-token limit, arrival time, workload type).
**Output:** a predicted latency per request, an execution order, and waiting-time / latency / throughput metrics per policy.

## 3. Objectives

1. Study FIFO and fixed-concurrency scheduling and their limits on variable LLM workloads.
2. Train a Random Forest model that predicts `engine_latency_s` from admission-time features.
3. Design CAMP, a scheduler driven by predicted latency.
4. Design Fair-CAMP, an aging-based extension that bounds the waiting of long requests.
5. Compare all five policies using average and P95 waiting time, end-to-end latency, throughput and fairness.

## 4. System Workflow

```
Requests (prompt, output limit, arrival time, workload type)
        │
        ▼
Feature extraction  (prompt tokens, max output tokens, arrival rate, queue depth,
        │            active requests, concurrency limit, workload type)
        ▼
Random Forest latency predictor  ──►  predicted engine latency per request
        │
        ▼
Scheduling module:  FIFO | Fixed-K | Adaptive | CAMP | Fair-CAMP
        │
        ▼
Request execution (admission, dispatch, service time, completion)
        │
        ▼
Evaluation: avg / P95 / max wait, end-to-end latency, throughput, Jain's fairness index
```

## 5. Dataset

| File | Purpose | Size |
|---|---|---|
| `latency_dataset_tokenfixed.csv` | Latency-prediction dataset (measured on the GPU) | 180 requests: 60 steady, 60 mixed, 60 bursty |
| `benchmark_traces_tokenfixed.csv` | Scheduler benchmark trace | 36 requests: 12 per workload type |

**Collection.** The 180 records were produced by serving **Qwen2.5-1.5B-Instruct on an NVIDIA T4 GPU (Google Colab)**: 3 workload patterns × 3 runs × 20 requests, with concurrency limits 1, 2 and 4 (60 requests each). Each record logs the request features, measured engine latency, time to first token, generated tokens, GPU utilization and GPU memory. All 180 requests completed.

| Quantity | Mean ± SD | Range |
|---|---|---|
| Engine latency (s) | 1.026 ± 0.388 | 0.483–1.826 |
| Queue wait (s) | 1.542 ± 2.345 | 0.000–9.902 |
| Time to first token (s) | 0.093 ± 0.046 | 0.038–0.231 |
| Prompt tokens | 250 ± 178 | 49–529 |
| Max output tokens | 63.8 ± 25.6 | 32–96 |

**Preparation.** The workload type is one-hot encoded (9 input columns). The split is **by run**: runs 1–2 train (120 records) and run 3 is held out (60 records), so requests from one run never appear in both sets.

**Permissions.** Prompts were written for this project. No personal or confidential data is used. No API keys or credentials are stored in this repository.

## 6. Model and Tools

| Item | Detail |
|---|---|
| Latency predictor | `sklearn.ensemble.RandomForestRegressor` – 100 trees, max depth 5, min samples split 4, min samples leaf 2, `random_state=42` |
| Target | `engine_latency_s` |
| Inference model (data collection) | Qwen2.5-1.5B-Instruct (Apache-2.0 per its model card – please verify) |
| Hardware (data collection) | NVIDIA T4 GPU, Google Colab |
| Saved predictor | `camp_latency_predictor.joblib` (load with scikit-learn 1.6.1) |
| Language / libraries | Python, pandas, NumPy, scikit-learn, matplotlib |

## 7. Scheduling Policies

| Policy | Concurrency | Selection rule |
|---|---|---|
| FIFO | K = 4 | Arrival order |
| Fixed-K | K = 4 | Arrival order (identical to FIFO in this study) |
| Adaptive | K = 2 if arrival rate ≥ 2 req/s, else K = 4 | Arrival order |
| **CAMP** | K = 4 | Smallest predicted latency |
| **Fair-CAMP** | K = 4 | Oldest request if it has waited ≥ 2 s, else smallest predicted latency |

## 8. Evaluation Method

Scheduling is evaluated with an **event-driven, trace-driven procedure**. Requests enter at their trace arrival times and each takes one of K slots for its **predicted service time**. All five policies use the identical 36-request trace. Metrics: average / P95 / maximum queue wait, average / P95 end-to-end latency, throughput, class-wise wait (short / medium / long as terciles of service time) and Jain's fairness index.

> **Scope:** this is **not** live GPU scheduling. See Section 12.

## 9. Results

### 9.1 Latency prediction (60 held-out requests)

| Model | MAE (s) | RMSE (s) | R² |
|---|---|---|---|
| Mean value | 0.326 | 0.385 | −0.198 |
| Linear regression | 0.165 | 0.233 | 0.560 |
| **Random Forest** | **0.135** | **0.186** | **0.721** |

Spearman rank correlation (predicted vs measured): 0.787.
Feature importance: max output tokens 83.68%, arrival rate 6.59%, queue depth 4.90%, prompt tokens 2.52%.

### 9.2 Scheduler comparison (36 requests)

| Policy | Avg wait (s) | P95 wait (s) | Max wait (s) | Avg E2E (s) | P95 E2E (s) | Throughput (req/s) |
|---|---|---|---|---|---|---|
| FIFO | 1.9820 | 3.5751 | 3.8535 | 3.1401 | 4.9982 | 3.3179 |
| Fixed-K | 1.9820 | 3.5751 | 3.8535 | 3.1401 | 4.9982 | 3.3179 |
| Adaptive | 5.3444 | 10.8564 | 11.7712 | 6.5025 | 11.8957 | 1.8649 |
| **CAMP** | **1.7203** | 6.9367 | 7.5285 | **2.8784** | 8.4436 | 3.3132 |
| **Fair-CAMP** | 1.8972 | 3.6577 | 3.7676 | 3.0553 | 4.9447 | 3.3100 |

**Headline numbers**
- CAMP vs FIFO: average wait **−13.20%**, average end-to-end latency **−8.33%**, throughput unchanged (−0.14%).
- Fair-CAMP vs CAMP: P95 wait **−47.27%**.
- Fair-CAMP vs FIFO: average wait −4.28%, average end-to-end −2.70%.
- Adaptive is the worst policy (throughput −43.8% vs FIFO).

### 9.3 Fairness by request class (average queue wait, s)

| Policy | Short | Medium | Long |
|---|---|---|---|
| FIFO | 1.692 | 1.791 | 2.463 |
| CAMP | 0.266 | 1.170 | 3.724 |
| Fair-CAMP | 1.055 | 1.942 | 2.695 |

Fair-CAMP cuts the average wait of long requests by 27.65% (3.72 s → 2.69 s) and their maximum wait by 50.3% (7.53 s → 3.74 s). Short and medium requests pay for this. Jain's index over waiting times: FIFO 0.741, CAMP 0.390, Fair-CAMP 0.677.

### 9.4 Workload-wise average / P95 queue wait (s)

| Policy | Steady | Mixed | Bursty |
|---|---|---|---|
| FIFO / Fixed-K | 2.326 / 3.499 | 1.724 / 3.790 | 1.895 / 3.293 |
| Adaptive | 6.811 / 10.971 | 4.012 / 9.673 | 5.210 / 11.046 |
| CAMP | 1.810 / 7.429 | 1.695 / 4.305 | 1.656 / 5.410 |
| Fair-CAMP | 2.170 / 3.691 | 1.814 / 3.629 | 1.708 / 3.256 |

## 10. Output Quality and Reliability

- CAMP gives the lowest mean latency but a heavy tail: its P95 wait is 94% above FIFO's.
- Fair-CAMP restores P95 to within 2.3% of FIFO, but on the mixed workload its average wait is 5.2% worse than FIFO's.
- The predictor under-predicts the longest requests (predicted maximum 1.49 s against a measured maximum of 1.79 s on the held-out set). Inspect the largest errors in the notebook before presenting.

## 11. Efficiency

- Mean measured engine latency 1.03 s; mean time to first token 0.093 s.
- GPU memory during data collection about 12.4 GB (T4).
- The scheduler's selection step scans the waiting queue once per dispatch, which is linear in queue length.

## 12. Limitations

1. The scheduler comparison is event-driven and trace-driven, **not live GPU scheduling**. GPU contention between concurrent requests is not re-estimated during the replay.
2. Service times in the replay are the predictor's outputs, produced with queue depth and active requests set to zero. The effect of prediction error on dispatch order is therefore not isolated.
3. One 36-request trace, so results are point values without confidence intervals.
4. The predictor was trained on 120 records from one model and one GPU type.
5. Adaptive's threshold (2 req/s) and the 2-s aging threshold are fixed settings, not tuned optima.

## 13. Responsible AI

- **Honest reporting:** all scheduling numbers are labelled as trace-driven evaluation results, not live GPU performance.
- **Reliability:** prediction errors can change scheduling order, and the held-out error is reported above.
- **Fairness:** shortest-first scheduling disadvantages long requests. Fair-CAMP is the mitigation and its cost is shown.
- **Privacy and licensing:** self-written prompts, no personal data. Check the Qwen2.5 and library licenses before redistributing.
- **Reproducibility:** datasets, model settings and scripts are in this repository.

## 14. Repository Structure

```
.
├── README.md
├── requirements.txt
├── Adaptive_LLM_Inference_Engine_Final.ipynb   # analysis, scheduler results, plots
├── generate_paper_figures.py                   # regenerates the five paper figures
├── IEEE_Adaptive_Parallel_LLM_Inference_Engine.docx   # research paper
├── data/
│   ├── latency_dataset_tokenfixed.csv
│   ├── benchmark_traces_tokenfixed.csv
│   ├── all_scheduler_request_results.csv
│   ├── scheduler_overall_comparison.csv
│   ├── workload_wise_comparison.csv
│   └── final_research_findings.csv
├── models/
│   └── camp_latency_predictor.joblib
└── figures/                                    # output of generate_paper_figures.py
```

Tip: remove duplicate files (the `_1` copies) so the repository has one clear version of each result.

## 15. Setup and Run

**Software:** Python 3.10 or newer (the notebook was last run on 3.13). Tested with pandas 2.2.3, scikit-learn 1.6.1 and matplotlib 3.10.0. Use scikit-learn 1.6.1 to load the saved `.joblib` model.

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt

# 1. Regenerate the paper figures and print the numbers behind them
python generate_paper_figures.py data figures

# 2. Run the full analysis
jupyter notebook Adaptive_LLM_Inference_Engine_Final.ipynb
```

**Google Colab:** open the notebook, upload the CSV files from `data/` when prompted, then run all cells. The analysis and scheduler evaluation run on CPU. A GPU was needed only for the original latency data collection.

**Expected output:** `Train=120 Test=60 MAE=0.1351 R2=0.7209`, followed by the scheduler tables in Section 9.

## 16. Team Contributions

| Name | PRN | Contribution |
|---|---|---|
| Nikheel C. Khadakabhavi | 202401110020 | [fill in: e.g. predictor, scheduling policies, evaluation, paper] |
| [Member 2] | [PRN] | [fill in] |

## 17. Future Work

Integrate CAMP and Fair-CAMP with a live inference engine such as vLLM; update predictions as queue depth and active requests change; add GPU utilization and memory as predictor inputs; collect larger and more varied workloads; tune the aging threshold against explicit latency targets.

## 18. References

1. G.-I. Yu et al., "Orca: A distributed serving system for transformer-based generative models," OSDI, 2022.
2. W. Kwon et al., "Efficient memory management for large language model serving with PagedAttention," SOSP, 2023.
3. B. Wu et al., "Fast distributed inference serving for large language models," arXiv:2305.05920, 2023.
4. Y. Jin et al., "S³: Increasing GPU utilization during generative inference for higher throughput," NeurIPS, 2023.
5. Z. Zheng et al., "Response length perception and sequence scheduling," NeurIPS, 2023.
6. H. Qiu et al., "Efficient interactive LLM serving with proxy model-based sequence length prediction," arXiv:2404.08509, 2024.
7. Y. Fu et al., "Efficient LLM scheduling by learning to rank," NeurIPS, 2024 (arXiv:2408.15792). **Main reference paper.**
8. A. Agrawal et al., "Taming throughput-latency tradeoff in LLM inference with Sarathi-Serve," OSDI, 2024.
9. L. E. Schrage, "A proof of the optimality of the shortest remaining processing time discipline," Operations Research, 1968.
10. L. Breiman, "Random forests," Machine Learning, 2001.
11. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," JMLR, 2011.
12. A. Yang et al., "Qwen2.5 technical report," arXiv:2412.15115, 2024.
13. R. Jain, D.-M. Chiu, W. R. Hawe, "A quantitative measure of fairness and discrimination for resource allocation in shared computer systems," DEC-TR-301, 1984.
14. J. Dean and L. A. Barroso, "The tail at scale," Communications of the ACM, 2013.

## 19. Acknowledgements

Course instructor: [name]. Thanks to the authors of the papers above and to the Qwen, scikit-learn and Google Colab teams for the tools used.
