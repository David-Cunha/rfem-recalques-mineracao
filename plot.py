"""Figura 2: reproducao de Ahmed e Hussain (2020). Linhas cheias: simulacao; tracejadas: Eq. (12)
com gamma calculado (verificacoes.py); losangos: valores publicados. Uso: python plot.py"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator

VIRG = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
D = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(D, "out", "resultados.json")))
vp = os.path.join(D, "out", "verificacoes.json")
pred = json.load(open(vp))["teste_analitico"]["casos"] if os.path.exists(vp) else []
mc = r["monte_carlo"]
cols = {0.1: "#86b6ef", 0.3: "#3987e5", 0.5: "#184f95"}
TH = (0.05, 0.5, 1.0, 5.0, 20.0)
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.9), dpi=200)
for c in (0.1, 0.3, 0.5):
    rows = sorted([m for m in mc if m["COV_E"] == c], key=lambda m: m["theta_m"])
    th = [m["theta_m"] for m in rows]
    lbl = f"COV$_E$ = {str(c).replace('.', ',')}"
    axs[0].plot(th, [m["inc_pct"] for m in rows], "-o", color=cols[c], lw=2, ms=4, label=lbl)
    axs[1].plot(th, [m["cov_s"] for m in rows], "-o", color=cols[c], lw=2, ms=4, label=lbl)
    pr = sorted([p for p in pred if p["COV_E"] == c], key=lambda p: p["theta"])
    if pr:
        axs[0].plot([p["theta"] for p in pr], [p["inc_pred"] for p in pr], "--", color=cols[c], lw=1)
axs[0].plot([0.05, 20], [12.6, 25.9], "D", mfc="white", mec="#eb6834", mew=1.5, ms=6,
            label="Ahmed e Hussain (2020)")
axs[0].plot([], [], "--", color="#52514e", lw=1, label="Eq. (12)")
axs[0].set_ylabel("Acréscimo da média sobre δ$_{det}$ (%)")
axs[1].set_ylabel("COV do recalque")
for ax, l in zip(axs, "ab"):
    ax.set_xscale("log"); ax.set_xlabel("θ (m)")
    ax.xaxis.set_major_locator(FixedLocator(TH)); ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(VIRG); ax.yaxis.set_major_formatter(VIRG)
    ax.grid(True, alpha=0.25, lw=0.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title(f"({l})", fontsize=9, loc="left")
axs[0].legend(frameon=False, fontsize=6.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "out", "fig_resultados.png"))
print("ok")
