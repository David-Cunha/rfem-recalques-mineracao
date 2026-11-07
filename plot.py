import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(D, "out", "resultados.json")))
mc = r["monte_carlo"]
cols = {0.1: "#86b6ef", 0.3: "#3987e5", 0.5: "#184f95"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
for c in (0.1, 0.3, 0.5):
    rows = sorted([m for m in mc if m["COV_E"] == c], key=lambda m: m["theta_m"])
    th = [m["theta_m"] for m in rows]
    axs[0].plot(th, [m["inc_pct"] for m in rows], "-o", color=cols[c], lw=2, ms=6, label=f"COV$_E$ = {str(c).replace('.', ',')}")
    axs[1].plot(th, [m["cov_s"] for m in rows], "-o", color=cols[c], lw=2, ms=6, label=f"COV$_E$ = {str(c).replace('.', ',')}")
axs[0].plot([0.05, 20], [12.6, 25.9], "D", mfc="white", mec="#eb6834", mew=2, ms=8,
            label="Ahmed e Hussain (2020), COV$_E$ = 0,5")
axs[0].set_ylabel("Acréscimo da média sobre δ$_{det}$ (%)")
axs[1].set_ylabel("Coeficiente de variação do recalque")
for ax in axs:
    ax.set_xscale("log"); ax.set_xlabel("Escala de flutuação θ (m)")
    ax.grid(True, color="#e6e5df", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].legend(frameon=False, fontsize=8)
axs[0].set_title("(a) Viés da média", loc="left"); axs[1].set_title("(b) Dispersão", loc="left")
fig.tight_layout()
fig.savefig(os.path.join(D, "out", "fig_resultados.png"))
print("ok")
