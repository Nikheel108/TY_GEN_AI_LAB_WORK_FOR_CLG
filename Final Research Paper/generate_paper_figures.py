"""
Regenerates all five figures used in the IEEE paper.
Inputs (put in DATA_DIR):  latency_dataset_tokenfixed.csv, all_scheduler_request_results.csv
Outputs (in OUT_DIR):      f_arch.png, f_pred.png, f_imp.png, f_sched.png, f_class.png
Usage:  python generate_paper_figures.py [DATA_DIR] [OUT_DIR]
Requires: pandas numpy matplotlib scikit-learn
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
OUT  = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
BLUE, LIGHT, GREY, RED = "#1f4e79", "#7fa6c9", "#b0b0b0", "#c0504d"

# ---------- 1. train predictor exactly as in the paper (run-based split) ----------
d = pd.read_csv(os.path.join(DATA, "latency_dataset_tokenfixed.csv"))
feats = ["prompt_tokens", "max_output_tokens", "arrival_rate_5s_rps", "queue_depth_at_admission",
         "active_requests_at_admission", "concurrency_limit", "workload_type"]
X = pd.get_dummies(d[feats], columns=["workload_type"], dtype=int)
y = d["engine_latency_s"]
test = d["run_id"].isin(["steady_run_3", "mixed_run_3", "bursty_run_3"])      # run 3 held out
rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=4,
                           min_samples_leaf=2, random_state=42).fit(X[~test], y[~test])
pred, actual = rf.predict(X[test]), y[test]
mae, r2 = mean_absolute_error(actual, pred), r2_score(actual, pred)
print(f"Train={int((~test).sum())} Test={int(test.sum())}  MAE={mae:.4f}  R2={r2:.4f}")

# ---------- Fig 1: architecture ----------
fig, ax = plt.subplots(figsize=(3.5, 4.1)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 12.6)
boxes = [("Incoming requests\n(prompt, output limit, arrival time)", 11.2),
         ("Feature extraction\n(tokens, arrival rate, queue depth,\nactive requests, concurrency, workload)", 9.3),
         ("Random Forest latency predictor\npredicted engine latency per request", 7.4),
         ("Scheduling module\nFIFO | Fixed-K | Adaptive | CAMP | Fair-CAMP", 5.5),
         ("Execution on LLM backend\n(Qwen2.5-1.5B-Instruct, NVIDIA T4 GPU)", 3.6),
         ("Metrics: avg/P95 wait, end-to-end,\nthroughput, fairness", 1.7)]
for t, yy in boxes:
    ax.add_patch(FancyBboxPatch((0.6, yy-0.7), 8.8, 1.5, boxstyle="round,pad=0.05", fc="#e8eef5", ec=BLUE, lw=.8))
    ax.text(5, yy+0.05, t, ha="center", va="center", fontsize=6.6)
for yy in [11.2, 9.3, 7.4, 5.5, 3.6]:
    ax.add_patch(FancyArrowPatch((5, yy-0.78), (5, yy-1.12), arrowstyle="-|>", mutation_scale=7, color="k", lw=.7))
fig.tight_layout(); fig.savefig(f"{OUT}/f_arch.png", dpi=300); plt.close(fig)

# ---------- Fig 2: measured vs predicted ----------
fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.scatter(actual, pred, s=12, c=BLUE, alpha=.75, edgecolor="none")
lo, hi = min(actual.min(), pred.min()) - .05, max(actual.max(), pred.max()) + .05
ax.plot([lo, hi], [lo, hi], "k--", lw=.8)
ax.set_xlabel("Measured engine latency (s)"); ax.set_ylabel("Predicted engine latency (s)")
ax.text(.05, .92, f"MAE = {mae:.3f} s\n$R^2$ = {r2:.3f}", transform=ax.transAxes, va="top")
ax.grid(alpha=.3, lw=.4); fig.tight_layout(); fig.savefig(f"{OUT}/f_pred.png", dpi=300); plt.close(fig)

# ---------- Fig 3: feature importance ----------
names = {"max_output_tokens": "max output tokens", "arrival_rate_5s_rps": "arrival rate (5 s)",
         "queue_depth_at_admission": "queue depth", "prompt_tokens": "prompt tokens",
         "active_requests_at_admission": "active requests", "workload_type_mixed": "type: mixed",
         "workload_type_bursty": "type: bursty", "workload_type_steady": "type: steady",
         "concurrency_limit": "concurrency limit"}
imp = pd.Series(rf.feature_importances_, X.columns).sort_values()
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.barh([names[i] for i in imp.index], imp.values, color=BLUE)
ax.set_xlabel("Impurity-based importance"); ax.grid(axis="x", alpha=.3, lw=.4)
fig.tight_layout(); fig.savefig(f"{OUT}/f_imp.png", dpi=300); plt.close(fig)

# ---------- scheduler results ----------
r = pd.read_csv(os.path.join(DATA, "all_scheduler_request_results.csv"))
order = ["FIFO", "Fixed-K", "Adaptive", "CAMP", "Fair-CAMP"]
g = r.groupby("Scheduler").agg(w=("queue_wait_s", "mean"),
                               p=("queue_wait_s", lambda x: np.percentile(x, 95)),
                               e=("end_to_end_s", "mean")).loc[order]
print(g.round(4))

# ---------- Fig 4: scheduler comparison ----------
fig, ax = plt.subplots(figsize=(3.5, 2.6)); x = np.arange(5); w = .27
ax.bar(x-w, g.w, w, label="Avg wait", color=BLUE)
ax.bar(x,   g.p, w, label="P95 wait", color=LIGHT)
ax.bar(x+w, g.e, w, label="Avg end-to-end", color=GREY)
ax.set_xticks(x); ax.set_xticklabels(order, fontsize=7); ax.set_ylabel("Seconds")
ax.legend(fontsize=6.5, frameon=False); ax.grid(axis="y", alpha=.3, lw=.4)
fig.tight_layout(); fig.savefig(f"{OUT}/f_sched.png", dpi=300); plt.close(fig)

# ---------- Fig 5: class-wise waiting (terciles of service time) ----------
lo_, hi_ = np.percentile(r[r.Scheduler == "FIFO"].service_time_s, [33.3, 66.7])
r["cls"] = pd.cut(r.service_time_s, [-1, lo_, hi_, 99], labels=["Short", "Medium", "Long"])
m = r.groupby(["Scheduler", "cls"], observed=True).queue_wait_s.mean().unstack().loc[["FIFO", "CAMP", "Fair-CAMP"]]
print(m.round(4))
fig, ax = plt.subplots(figsize=(3.5, 2.5)); x = np.arange(3); w = .26
for i, (s, c) in enumerate(zip(m.index, [GREY, BLUE, RED])):
    ax.bar(x+(i-1)*w, m.loc[s].values, w, label=s, color=c)
ax.set_xticks(x); ax.set_xticklabels(["Short", "Medium", "Long"])
ax.set_xlabel("Request class (by service time)"); ax.set_ylabel("Average queue wait (s)")
ax.legend(fontsize=7, frameon=False); ax.grid(axis="y", alpha=.3, lw=.4)
fig.tight_layout(); fig.savefig(f"{OUT}/f_class.png", dpi=300); plt.close(fig)
print("Saved figures to", OUT)
