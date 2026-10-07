"""Builds every table and figure of the paper from the raw CSV files.

Inputs : results/raw_main.csv, results/raw_ksweep.csv
Outputs: results/summary_main.csv, results/summary_ksweep.csv,
         results/summary_lazy.csv, results/summary_dnak_diagnostics.csv,
         results/tables/*.tex, results/figures/*.pdf

Speed-up and distance-computation (DC) reduction are computed per seed
against Lloyd's run on the same seed, then averaged (mean and standard
deviation over seeds)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _setup import RESULTS

TAB = os.path.join(RESULTS, "tables"); FIG = os.path.join(RESULTS, "figures")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)

METHODS = ["Lloyd", "Hamerly", "Elkan", "Yinyang", "Exponion", "Ball", "DNAK", "DNAK-6"]
LABEL = {"Ball": "Ball $k$-means", "DNAK": "DNAK-B", "DNAK-6": "DNAK"}
CONFIG_ORDER = ["synth-2D-k64", "synth-2D-k128", "synth-3D-k64",
                "synth-4D-k64", "uber-k64", "uber-k256", "road-k64",
                "china-k64", "flower-k32"]
CONFIG_TEX = {
    "synth-2D-k64": r"Synth., $d{=}2$, $k{=}64$",
    "synth-2D-k128": r"Synth., $d{=}2$, $k{=}128$",
    "synth-3D-k64": r"Synth., $d{=}3$, $k{=}64$",
    "synth-4D-k64": r"Synth., $d{=}4$, $k{=}64$",
    "uber-k64": r"Uber NYC, $d{=}2$, $k{=}64$",
    "uber-k256": r"Uber NYC, $d{=}2$, $k{=}256$",
    "road-k64": r"3D Road Network, $d{=}3$, $k{=}64$",
    "china-k64": r"china RGB, $d{=}3$, $k{=}64$",
    "flower-k32": r"flower RGB, $d{=}3$, $k{=}32$",
}
SHORT = {"synth-2D-k64": "Syn d=2\nk=64", "synth-2D-k128": "Syn d=2\nk=128",
         "synth-3D-k64": "Syn d=3\nk=64", "synth-4D-k64": "Syn d=4\nk=64",
         "uber-k64": "Uber\nk=64", "uber-k256": "Uber\nk=256",
         "road-k64": "Road\nk=64", "china-k64": "china\nk=64",
         "flower-k32": "flower\nk=32"}
COLORS = {"Lloyd": "#9aa0a6", "Hamerly": "#5b8bd6", "Elkan": "#d9785a",
          "Yinyang": "#7fbf8e", "Exponion": "#c68a35", "Ball": "#9b6fb8",
          "DNAK": "#6aa5c9", "DNAK-6": "#173f6e", "DNAK-Lazy": "#9ec5d8"}


def per_seed(df):
    """Attach per-seed speed-up and DC reduction relative to Lloyd."""
    ref = df[df.algo == "Lloyd"][["config", "seed", "time_s", "comps"]]
    ref = ref.rename(columns={"time_s": "t_lloyd", "comps": "c_lloyd"})
    df = df.merge(ref, on=["config", "seed"])
    df["speedup"] = df.t_lloyd / df.time_s
    df["dcred"] = df.c_lloyd / df.comps
    return df


def summarise(df):
    g = df.groupby(["config", "algo"])
    s = g.agg(n=("n", "first"), d=("d", "first"), k=("k", "first"),
              seeds=("seed", "nunique"), iters=("iters", "mean"),
              time_mean=("time_s", "mean"), time_std=("time_s", "std"),
              speedup_mean=("speedup", "mean"), speedup_std=("speedup", "std"),
              dcred_mean=("dcred", "mean"), dcred_std=("dcred", "std"),
              all_exact=("exact", "all")).reset_index()
    return s


def fmt(m, s, nd=1):
    return f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"


# ------------------------------------------------------------- main table
raw = pd.read_csv(os.path.join(RESULTS, "raw_main.csv"))
assert raw.exact.all(), "non-exact run in raw_main.csv"
raw = per_seed(raw)
S = summarise(raw)
S.to_csv(os.path.join(RESULTS, "summary_main.csv"), index=False,
         float_format="%.4f")

configs = [c for c in CONFIG_ORDER if c in set(S.config)]


SHORT_TEX = {
    "synth-2D-k64": r"Synth., $d{=}2$, $k{=}64$",
    "synth-2D-k128": r"Synth., $d{=}2$, $k{=}128$",
    "synth-3D-k64": r"Synth., $d{=}3$, $k{=}64$",
    "synth-4D-k64": r"Synth., $d{=}4$, $k{=}64$",
    "uber-k64": r"Uber, $d{=}2$, $k{=}64$",
    "uber-k256": r"Uber, $d{=}2$, $k{=}256$",
    "road-k64": r"Road, $d{=}3$, $k{=}64$",
    "china-k64": r"china, $d{=}3$, $k{=}64$",
    "flower-k32": r"flower, $d{=}3$, $k{=}32$",
}


def matrix_table(col, fname, lloyd_col):
    """Rows: configurations; columns: methods; cells: mean (std).
    DNAK-6 was timed in a later session, so it appears in the
    distance-computation table only."""
    meths = [m for m in METHODS[1:] if not (lloyd_col and m == "DNAK-6")]
    head = ["Configuration"] + ([r"\shortstack[r]{Lloyd\\ time (s)}"] if lloyd_col else []) + \
           [r"\shortstack[r]{" + LABEL.get(m, m).replace(" ", r"\\ ") + "}" if m == "Ball" else LABEL.get(m, m) for m in meths]
    lines = [r"\begin{tabular}{l" + ("r" if lloyd_col else "") + "r" * len(meths) + "}",
             r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for c in configs:
        sub = S[S.config == c].set_index("algo")
        best = sub.loc[meths, col + "_mean"].max()
        cells = [SHORT_TEX[c]]
        if lloyd_col:
            cells.append(f"{sub.loc['Lloyd', 'time_mean']:.2f}")
        for m in meths:
            mval = sub.loc[m, col + "_mean"]; sval = sub.loc[m, col + "_std"]
            cell = f"{mval:.1f}" + r"\,{\scriptsize(" + f"{sval:.1f}" + ")}"
            if np.isclose(mval, best):
                cell = r"\textbf{" + f"{mval:.1f}" + "}" + r"\,{\scriptsize(" + f"{sval:.1f}" + ")}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, fname), "w").write("\n".join(lines) + "\n")


matrix_table("speedup", "table_speedup.tex", True)
matrix_table("dcred", "table_dcred.tex", False)

# ------------------------------------------------------------- dataset table
ds = raw.groupby("config").agg(dataset=("dataset", "first"), n=("n", "first"),
                               d=("d", "first"), k=("k", "first"),
                               seeds=("seed", "nunique")).reindex(configs)
lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Data set & $n$ & $d$ & $k$ & Seeds \\", r"\midrule"]
for c, r in ds.iterrows():
    lines.append(f"{r.dataset} & {r.n:,} & {r.d} & {r.k} & {r.seeds} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "table_datasets.tex"), "w").write(
    "\n".join(lines).replace(",", "{,}") + "\n")

# ------------------------------------------------------------- lazy table
L = S[S.algo.isin(["DNAK", "DNAK-Lazy"])].pivot(index="config", columns="algo",
                                                 values="speedup_mean")
Lsd = S[S.algo.isin(["DNAK", "DNAK-Lazy"])].pivot(index="config", columns="algo",
                                                   values="speedup_std")
lz = raw[raw.algo == "DNAK-Lazy"].groupby("config").agg(
    rebuilds=("rebuilds", "mean"), iters=("iters", "mean"),
    examined=("examined", "sum"), failed=("failed", "sum"))
lz["fail_rate"] = lz.failed / lz.examined
L = L.join(Lsd, rsuffix="_std").join(lz).reindex(configs)
L.to_csv(os.path.join(RESULTS, "summary_lazy.csv"), float_format="%.4f")
lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Configuration & DNAK & DNAK-Lazy & \shortstack[r]{Rebuilds /\\ iterations} & \shortstack[r]{Certificate\\ failures} \\",
         r"\midrule"]
for c, r in L.iterrows():
    lines.append(f"{CONFIG_TEX[c]} & {fmt(r.DNAK, r.DNAK_std)} & "
                 f"{fmt(r['DNAK-Lazy'], r['DNAK-Lazy_std'])} & "
                 f"{r.rebuilds:.0f} / {r.iters:.0f} & {100 * r.fail_rate:.1f}\\% \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "table_lazy.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------- DNAK diagnostics
D = raw[raw.algo == "DNAK"].groupby("config").agg(
    walks=("walks", "sum"), hops=("hops", "sum"),
    graph=("graph_time_s", "sum"), total=("time_s", "sum"),
    comps=("comps", "sum"))
D["hops_per_walk"] = D.hops / D.walks
D["graph_share"] = D.graph / D.total
D = D.reindex(configs)
D.to_csv(os.path.join(RESULTS, "summary_dnak_diagnostics.csv"),
         float_format="%.4f")
lines = [r"\begin{tabular}{lrr}", r"\toprule",
         r"Configuration & \shortstack[r]{Hops per\\ walk} & \shortstack[r]{Share of time in\\ graph construction} \\",
         r"\midrule"]
for c, r in D.iterrows():
    lines.append(f"{CONFIG_TEX[c]} & {r.hops_per_walk:.2f} & {100 * r.graph_share:.1f}\\% \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "table_diagnostics.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------- bar figure
fig, axes = plt.subplots(2, 1, figsize=(10, 6.4))
x = np.arange(len(configs)); w = 0.105
for ax, col, title in ((axes[0], "speedup_mean", "Wall-clock speed-up over Lloyd's algorithm"),
                       (axes[1], "dcred_mean", "Reduction in distance computations over Lloyd's algorithm")):
    meths_f = [m for m in METHODS[1:] if not (col == "speedup_mean" and m == "DNAK-6")]
    for i, m in enumerate(meths_f):
        vals = [S[(S.config == c) & (S.algo == m)][col].iloc[0] for c in configs]
        ax.bar(x + (i - 3) * w, vals, w, label=LABEL.get(m, m).replace("$", ""),
               color=COLORS[m])
        if m == ("DNAK-6" if col != "speedup_mean" else "DNAK"):
            for xi, v in zip(x, vals):
                ax.text(xi + (i - 3) * w, v * 1.06, f"{v:.0f}", ha="center",
                        fontsize=7, color=COLORS[m])
    ax.set_yscale("log"); ax.set_title(title, fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in configs], fontsize=8)
    ax.axhline(1, color="grey", lw=0.6, ls=":")
axes[0].legend(ncol=7, fontsize=8, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.28))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_main_bars.pdf"))
plt.close(fig)

# ------------------------------------------------------------- k-sweep
kp = os.path.join(RESULTS, "raw_ksweep.csv")
if os.path.exists(kp):
    K = pd.read_csv(kp)
    assert K.exact.all()
    K = per_seed(K)
    SK = summarise(K)
    SK.to_csv(os.path.join(RESULTS, "summary_ksweep.csv"), index=False,
              float_format="%.4f")
    algos = [m for m in ["Hamerly", "Yinyang", "Exponion", "Ball", "DNAK"]
             if m in set(SK.algo)]
    dims = sorted(SK.d.unique())
    fig, axes = plt.subplots(1, len(dims), figsize=(5.2 * len(dims), 3.6))
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, dims):
        for m in algos:
            sub = SK[(SK.d == d) & (SK.algo == m)].sort_values("k")
            ax.errorbar(sub.k, sub.speedup_mean, yerr=sub.speedup_std,
                        marker="s" if m == "DNAK" else "o",
                        ls="-" if m == "DNAK" else "--", capsize=2,
                        lw=1.8 if m == "DNAK" else 1.0, ms=4,
                        color=COLORS[m], label=LABEL.get(m, m).replace("$", ""))
        ks = sorted(SK[SK.d == d].k.unique())
        ax.set_xscale("log", base=2); ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("$k$"); ax.set_title(f"$d={d}$, $n=10^5$", fontsize=10)
    axes[0].set_ylabel("Speed-up over Lloyd's algorithm")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_ksweep.pdf"))
    plt.close(fig)

    lines = [r"\begin{tabular}{rr" + "r" * len(algos) + "}", r"\toprule",
             "$d$ & $k$ & " + " & ".join(LABEL.get(m, m) for m in algos) + r" \\",
             r"\midrule"]
    for d in dims:
        for k in sorted(SK[SK.d == d].k.unique()):
            sub = SK[(SK.d == d) & (SK.k == k)].set_index("algo")
            best = sub.loc[algos, "speedup_mean"].max()
            cells = []
            for m in algos:
                c = fmt(sub.loc[m, "speedup_mean"], sub.loc[m, "speedup_std"])
                cells.append(r"\textbf{" + c + "}" if np.isclose(sub.loc[m, "speedup_mean"], best) else c)
            lines.append(f"{d} & {k} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if d != dims[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    open(os.path.join(TAB, "table_ksweep.tex"), "w").write("\n".join(lines) + "\n")

print("tables:", sorted(os.listdir(TAB)))
print("figures:", sorted(os.listdir(FIG)))

# ------------------------------------------------------------- DNAK vs Exponion, paired by seed
rows = []
for f in ("raw_main.csv", "raw_ksweep.csv"):
    p = os.path.join(RESULTS, f)
    if not os.path.exists(p):
        continue
    d = pd.read_csv(p)
    pv = d.pivot_table(index=["config", "seed"], columns="algo", values="time_s")
    ratio = (pv["Exponion"] / pv["DNAK"]).groupby(level=0)
    meta = d.groupby("config").agg(d=("d", "first"), k=("k", "first"))
    for c, x in ratio:
        rows.append(dict(source=f, config=c, d=meta.loc[c, "d"], k=meta.loc[c, "k"],
                         time_ratio_gmean=float(np.exp(np.log(x).mean())),
                         dnak_faster=int((x > 1).sum()), seeds=int(x.size)))
P = pd.DataFrame(rows)
P.to_csv(os.path.join(RESULTS, "summary_dnak_vs_exponion.csv"), index=False,
         float_format="%.3f")
lines = [r"\begin{tabular}{lrr}", r"\toprule",
         r"Configuration & \shortstack[r]{Time ratio\\ Exponion/DNAK} & \shortstack[r]{Seeds where\\ DNAK is faster} \\",
         r"\midrule"]
for c in configs:
    r = P[(P.source == "raw_main.csv") & (P.config == c)].iloc[0]
    lines.append(f"{CONFIG_TEX[c]} & {r.time_ratio_gmean:.2f} & {r.dnak_faster} of {r.seeds} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "table_pairwise.tex"), "w").write("\n".join(lines) + "\n")
print("pairwise table written")

# ------------------------------------------------------------- ablation and Delaunay degree
ABL_TEX = {"synth-2D-k128": r"Syn. 2-D, 128", "synth-3D-k64": r"Syn. 3-D, 64",
           "uber-k256": r"Uber 2-D, 256", "road-k64": r"Road 3-D, 64",
           "sweep-2D-k512": r"Syn. 2-D, 512", "sweep-3D-k512": r"Syn. 3-D, 512"}
ap = os.path.join(RESULTS, "raw_ablation.csv")
if os.path.exists(ap):
    A = pd.read_csv(ap); assert A.exact.all()
    A.loc[A.algo == "Exponion", "algo"] = "Exponion (full sort)"
    G = A.groupby(["config", "algo"]).agg(t=("time_s", "mean"), c=("comps", "mean"),
                                          deg=("deg_mean", "mean"), dmax=("deg_max", "max")).reset_index()
    G.to_csv(os.path.join(RESULTS, "summary_ablation.csv"), index=False, float_format="%.4f")
    order = [c for c in ABL_TEX if c in set(G.config)]
    meth = ["Hamerly", "Hamerly-noS", "DNAK", "DNAK+s"]
    lines = [r"\begin{tabular}{l" + "r" * len(meth) * 2 + "}", r"\toprule",
             r" & \multicolumn{4}{c}{Time (s)} & \multicolumn{4}{c}{Distance computations ($10^6$)} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"Data, $k$ & " + " & ".join(meth) + " & " + " & ".join(meth) + r" \\", r"\midrule"]
    for c in order:
        sub = G[G.config == c].set_index("algo")
        lines.append(ABL_TEX[c] + " & " + " & ".join(f"{sub.loc[m, 't']:.2f}" for m in meth) + " & "
                     + " & ".join(f"{sub.loc[m, 'c'] / 1e6:.1f}" for m in meth) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "table_ablation.tex"), "w").write("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"Data, $k$ & Mean degree & Maximum degree \\", r"\midrule"]
    for c in order:
        r = G[(G.config == c) & (G.algo == "DNAK")].iloc[0]
        lines.append(f"{ABL_TEX[c]} & {r.deg:.1f} & {int(r.dmax)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "table_degree.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------- Exponion implementation check
ep = os.path.join(RESULTS, "raw_exponion_check.csv")
if os.path.exists(ep):
    E = pd.read_csv(ep)
    E["assign_s"] = E.time_s - E.centroid_side_s
    G = E.groupby(["config", "algo"]).agg(t=("time_s", "mean"), cs=("centroid_side_s", "mean"),
                                          asg=("assign_s", "mean"), c=("comps", "mean")).reset_index()
    pv = E.pivot_table(index=["config", "seed"], columns="algo", values="time_s")
    wins = (pv["Exponion-annuli"] > pv["DNAK"]).groupby(level=0).sum()
    G.to_csv(os.path.join(RESULTS, "summary_exponion_check.csv"), index=False, float_format="%.4f")
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Data, $k$ & Method & Total (s) & \shortstack[r]{Centroid-side\\ work (s)} & \shortstack[r]{Assignment\\ pass (s)} & DC ($10^6$) \\",
             r"\midrule"]
    names = {"Exponion": "Exponion, full sort", "Exponion-annuli": "Exponion, annuli", "DNAK": "DNAK"}
    for c in [x for x in ABL_TEX if x in set(G.config)]:
        first = True
        for m in ["Exponion", "Exponion-annuli", "DNAK"]:
            r = G[(G.config == c) & (G.algo == m)].iloc[0]
            lines.append(f"{ABL_TEX[c] if first else ''} & {names[m]} & {r.t:.2f} & {r.cs:.2f} & {r.asg:.2f} & {r.c / 1e6:.1f} \\\\")
            first = False
        lines.append(r"\midrule" if c != "sweep-3D-k512" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    open(os.path.join(TAB, "table_exponion_check.tex"), "w").write("\n".join(lines) + "\n")
    print("DNAK faster than annulus Exponion on", int(wins.sum()), "of", int(pv.shape[0]), "seeds")

# ------------------------------------------------------------- degenerate inputs
dp = os.path.join(RESULTS, "degenerate.csv")
if os.path.exists(dp):
    Dg = pd.read_csv(dp)
    T = Dg.groupby("case").agg(runs=("exact", "size"), exact=("exact", "sum"))
    T.to_csv(os.path.join(RESULTS, "summary_degenerate.csv"))
    lines = [r"\begin{tabular}{lr}", r"\toprule", r"Input & Identical to Lloyd \\", r"\midrule"]
    for c, r in T.iterrows():
        c = c.replace("R^2", "the plane").replace("1e6", "$10^{6}$").replace("1e-6", "$10^{-6}$")
        c = c.replace("60x60", r"$60\times60$").replace("(x3)", "(three times)")
        lines.append(f"{c} & {int(r.exact)} of {int(r.runs)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "table_degenerate.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------- n-scaling
np_path = os.path.join(RESULTS, "raw_nscale.csv")
if os.path.exists(np_path):
    N = pd.read_csv(np_path)
    assert N.exact_vs_dnak.all()
    N["us_per_point_iter"] = 1e6 * N.time_s / (N.n * N.iters)
    SN = N.groupby(["n", "algo"]).agg(time_mean=("time_s", "mean"),
                                      time_std=("time_s", "std"),
                                      iters=("iters", "mean"),
                                      us_pt_it=("us_per_point_iter", "mean")).reset_index()
    SN.to_csv(os.path.join(RESULTS, "summary_nscale.csv"), index=False,
              float_format="%.4f")
    pv = N.pivot_table(index=["n", "seed"], columns="algo", values="time_s")
    ratio = (pv["Exponion"] / pv["DNAK"]).groupby(level=0)
    meths = ["Hamerly", "Yinyang", "Exponion", "Ball", "DNAK"]
    lines = [r"\begin{tabular}{r" + "r" * len(meths) + "rr}", r"\toprule",
             "$n$ & " + " & ".join(LABEL.get(m, m) for m in meths) +
             r" & \shortstack[r]{Exponion/DNAK\\ time ratio} & \shortstack[r]{DNAK\\ faster} \\",
             r"\midrule"]
    for n in sorted(N.n.unique()):
        sub = SN[SN.n == n].set_index("algo")
        best = sub.loc[meths, "time_mean"].min()
        cells = []
        for m in meths:
            c = f"{sub.loc[m, 'time_mean']:.2f}"
            cells.append(r"\textbf{" + c + "}" if np.isclose(sub.loc[m, "time_mean"], best) else c)
        x = ratio.get_group(n)
        lines.append(f"{n:,}".replace(",", "{,}") + " & " + " & ".join(cells) +
                     f" & {np.exp(np.log(x).mean()):.2f} & {(x > 1).sum()} of {x.size} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "table_nscale.tex"), "w").write("\n".join(lines) + "\n")
    print("n-scaling table written")

# ------------------------------------------------------------- original implementations
rows_o = []
for f in ("raw_original_main.csv", "raw_original_large.csv"):
    p = os.path.join(RESULTS, f)
    if os.path.exists(p):
        rows_o.append(pd.read_csv(p))
if rows_o:
    O = pd.concat(rows_o)
    ok = O[O.iters >= 0]
    assert ok.same_labels_as_dnak.all() and ok.same_iters_as_dnak.all()
    O["col"] = O.method + "|" + O.implementation
    SO = O.groupby(["config", "col"]).time_s.mean().unstack()
    SO.to_csv(os.path.join(RESULTS, "summary_original.csv"), float_format="%.4f")
    cols = [("DNAK|ours (Numba)", r"\shortstack[r]{DNAK\\ (ours)}"),
            ("Exponion|ours (Numba)", r"\shortstack[r]{Exponion\\ (ours)}"),
            ("Exponion|eakmeans (C++)", r"\shortstack[r]{Exponion\\ (orig.)}"),
            ("Yinyang|eakmeans (C++)", r"\shortstack[r]{Yinyang\\ (orig.)}"),
            ("Hamerly|eakmeans (C++)", r"\shortstack[r]{Hamerly\\ (orig.)}"),
            ("Ball k-means|original (C++)", r"\shortstack[r]{Ball\\ (orig.)}"),
            ("Ball k-means (ring)|original (C++)", r"\shortstack[r]{Ball ring\\ (orig.)}")]
    names = {"synth-2D-k128": r"Synth., $d{=}2$, $k{=}128$",
             "sweep-2D-k256": r"Synth., $d{=}2$, $k{=}256$",
             "sweep-2D-k512": r"Synth., $d{=}2$, $k{=}512$",
             "sweep-3D-k512": r"Synth., $d{=}3$, $k{=}512$",
             "uber-k256": r"Uber, $n{=}2{\times}10^5$, $k{=}256$",
             "road-k64": r"Road, $d{=}3$, $k{=}64$",
             "uber-full-k1024": r"Uber, $n{=}1.5{\times}10^6$, $k{=}1024^\ast$"}
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             "Configuration & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    for cfg in names:
        if cfg not in SO.index:
            continue
        r = SO.loc[cfg]
        vals = [r.get(c[0], np.nan) for c in cols]
        best = np.nanmin(vals)
        cells = []
        for v in vals:
            if not np.isfinite(v):
                cells.append("--")
            elif np.isclose(v, best):
                cells.append(r"\textbf{" + f"{v:.2f}" + "}")
            else:
                cells.append(f"{v:.2f}")
        lines.append(names[cfg] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "table_original.tex"), "w").write("\n".join(lines) + "\n")
    print("original-implementation table written")


# ------------------------------------------------------------- DNAK-6 against original code (Colab)
cp = os.path.join(RESULTS, "raw_dnak6_original_colab.csv")
if os.path.exists(cp):
    G = pd.read_csv(cp)
    assert G.same_as_dnak6.all()
    order6 = ["synth-2D-k128", "sweep-2D-k512", "sweep-2D-k1024", "sweep-3D-k512",
              "sweep-3D-k1024", "uber-k256", "road-k64"]
    name6 = {"synth-2D-k128": r"Synth., $d{=}2$, $k{=}128$", "sweep-2D-k512": r"Synth., $d{=}2$, $k{=}512$",
             "sweep-2D-k1024": r"Synth., $d{=}2$, $k{=}1024$", "sweep-3D-k512": r"Synth., $d{=}3$, $k{=}512$",
             "sweep-3D-k1024": r"Synth., $d{=}3$, $k{=}1024$", "uber-k256": r"Uber, $d{=}2$, $k{=}256$",
             "road-k64": r"Road, $d{=}3$, $k{=}64$"}
    SG = G.groupby(["config", "method"]).time_s.mean().unstack()
    pv = G.pivot_table(index=["config", "seed", "rep"], columns="method", values="time_s")
    cols6 = ["DNAK-6", "Exponion (eakmeans)", "Ball k-means (original)"]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Configuration & DNAK & \shortstack[r]{Exponion\\ (orig.)} & \shortstack[r]{Ball\\ (orig.)} & "
             r"\shortstack[r]{Ratio\\ Exp./DNAK} & \shortstack[r]{Ratio\\ Ball/DNAK} & \shortstack[r]{DNAK faster\\ (Exp.\,/\,Ball)} \\",
             r"\midrule"]
    rows6 = []
    for c in order6:
        if c not in SG.index:
            continue
        r = SG.loc[c]; x = pv.loc[c]
        re_ = x["Exponion (eakmeans)"] / x["DNAK-6"]; rb = x["Ball k-means (original)"] / x["DNAK-6"]
        best = min(r[cols6]); cells = []
        for m in cols6:
            v = f"{r[m]:.2f}"
            cells.append(r"\textbf{" + v + "}" if np.isclose(r[m], best) else v)
        lines.append(name6[c] + " & " + " & ".join(cells) +
                     f" & {np.exp(np.log(re_).mean()):.2f} & {np.exp(np.log(rb).mean()):.2f}"
                     f" & {(re_ > 1).sum()}, {(rb > 1).sum()} \\\\")
        rows6.append(dict(config=c, dnak6=r["DNAK-6"], exponion=r["Exponion (eakmeans)"], ball=r["Ball k-means (original)"],
                          ratio_exp=float(np.exp(np.log(re_).mean())), ratio_ball=float(np.exp(np.log(rb).mean())),
                          wins_exp=int((re_ > 1).sum()), wins_ball=int((rb > 1).sum()), runs=int(re_.size)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "table_dnak6_original.tex"), "w").write("\n".join(lines) + "\n")
    pd.DataFrame(rows6).to_csv(os.path.join(RESULTS, "summary_dnak6_original.csv"), index=False, float_format="%.3f")
    print("DNAK-6 vs original (Colab) table written")

# ------------------------------------------------------------- paired DNAK-B vs DNAK (same session)
pp = os.path.join(RESULTS, "raw_paired_b6.csv")
if os.path.exists(pp):
    B = pd.read_csv(pp); assert B.exact.all()
    B["tratio"] = B.time_dnak_b / B.time_dnak6
    B["cratio"] = B.comps_dnak_b / B.comps_dnak6
    SB = B.groupby("config").agg(seeds=("seed", "nunique"), t_b=("time_dnak_b", "mean"), t6=("time_dnak6", "mean"),
                                 tratio=("tratio", lambda x: float(np.exp(np.log(x).mean()))),
                                 wins=("tratio", lambda x: int((x > 1).sum())),
                                 cratio=("cratio", lambda x: float(np.exp(np.log(x).mean())))).reindex(configs)
    SB.to_csv(os.path.join(RESULTS, "summary_paired_b6.csv"), float_format="%.4f")
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Configuration & \shortstack[r]{DNAK-B\\ (s)} & \shortstack[r]{DNAK\\ (s)} & \shortstack[r]{Time\\ ratio} & \shortstack[r]{DNAK\\ faster} & \shortstack[r]{Distance\\ ratio} \\",
             r"\midrule"]
    for c, r in SB.iterrows():
        lines.append(f"{CONFIG_TEX[c]} & {r.t_b:.2f} & {r.t6:.2f} & {r.tratio:.2f} & {r.wins:.0f} of {r.seeds:.0f} & {r.cratio:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "table_paired_b6.tex"), "w").write("\n".join(lines) + "\n")
    print("paired DNAK-B/DNAK table written")

# ------------------------------------------------------------- ablation of DNAK mechanisms
ap = os.path.join(RESULTS, "raw_dnak_ablation.csv")
if os.path.exists(ap):
    A = pd.read_csv(ap); assert A.exact.all()
    VARL = {"DNAK": "DNAK-B", "DNAK-2": "+ Delaunay $s$-test, annulus stop",
            "DNAK-5": "+ local bounds, walk initialisation", "DNAK-6": "+ certificate reuse (DNAK)"}
    cfgs_a = ["sweep-2D-k512", "sweep-3D-k512", "uber-k256"]
    names_a = {"sweep-2D-k512": r"Synth., $d{=}2$, $k{=}512$", "sweep-3D-k512": r"Synth., $d{=}3$, $k{=}512$", "uber-k256": r"Uber, $d{=}2$, $k{=}256$"}
    SA = A.groupby(["config", "variant"]).agg(t=("time_s", "mean"), c=("comps", "mean"), g=("graph_time_s", "mean"))
    lines = [r"\begin{tabular}{l" + "rr" * len(cfgs_a) + "}", r"\toprule",
             "Variant & " + " & ".join(r"\multicolumn{2}{c}{" + names_a[c] + "}" for c in cfgs_a) + r" \\",
             " & " + " & ".join("time (s) & dist. (M)" for _ in cfgs_a) + r" \\", r"\midrule"]
    for v in ["DNAK", "DNAK-2", "DNAK-5", "DNAK-6"]:
        cells = []
        for c in cfgs_a:
            r = SA.loc[(c, v)]
            cells.append(f"{r.t:.2f} & {r.c / 1e6:.1f}")
        lines.append(VARL[v] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "table_dnak_ablation.tex"), "w").write("\n".join(lines) + "\n")
    print("DNAK ablation table written")

# ------------------------------------------------------------- graph share and rebuilds of DNAK in the paired runs
if os.path.exists(pp):
    R = raw[raw.algo == "DNAK-6"].groupby("config").agg(gs=("graph_time_s", "sum"), ts=("time_s", "sum"), rb=("rebuilds", "sum"), it=("iters", "sum")).reindex(configs)
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Configuration & \shortstack[r]{DNAK-B\\ (s)} & \shortstack[r]{DNAK\\ (s)} & \shortstack[r]{Time\\ ratio} & \shortstack[r]{DNAK\\ faster} & \shortstack[r]{Distance\\ ratio} & \shortstack[r]{Graph\\ share} & \shortstack[r]{Rebuilt\\ iterations} \\",
             r"\midrule"]
    for c, r in SB.iterrows():
        q = R.loc[c]
        lines.append(f"{CONFIG_TEX[c]} & {r.t_b:.2f} & {r.t6:.2f} & {r.tratio:.2f} & {r.wins:.0f} of {r.seeds:.0f} & {r.cratio:.1f} & {100 * q.gs / q.ts:.0f}\\% & {100 * q.rb / q.it:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "table_paired_b6.tex"), "w").write("\n".join(lines) + "\n")
    print("paired table with graph share written")
