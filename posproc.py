"""
Pos-processamento dos cenarios (le out/cenarios/*.npy e *.json, gera tabelas e figuras).

Recalque dimensional a partir do recalque normalizado g (ver cenarios.py):
        delta = q * B * g / muE        (q e muE em kPa, B em m, delta em m)
Pressoes (seção 4.3, Eq. 12): q = (1 + MR) m g0 / (B L) (total) e q_eq = m g0 / (B L) (pos-instalacao).
Limites (Quadro 6): 25 mm (total, q total); 13 mm (equipamento, q_eq); beta = 1/750 e 1/300 (C4, q_eq).
Sobol: C2 -> Y = P(delta > 25 mm); C4 -> Y = percentil 95% de beta (P(beta > 1/750) e nula em todo o fatorial).

Sensibilidade: indices de Sobol de primeira ordem (S1) e totais (ST) obtidos por decomposicao
de variancia (ANOVA) sobre o planejamento fatorial completo, com cada fator uniforme discreto
nos niveis simulados.

Uso: python posproc.py   (gera out/cenarios/resumo.json e out/cenarios/fig_*.png)
"""
import os, json, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator
VIRG = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))  # separador decimal brasileiro

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "cenarios")
G0 = 9.81
EQUIP = {  # massa (t) e planta da fundacao B x L (m): fichas tecnicas + folga de 0,5 m por lado
    "C1": (37.0, 3.50, 3.955), "C2": (77.0, 4.20, 4.866), "C4": (37.0, 3.50, 3.955)}
C3_Q = dict(q_eq=(3000 + 310) * G0 / (21 * 28), q_tot=(3000 + 310 + 6000) * G0 / (21 * 28))  # Han; Guevara (2013)
MR_BASE, MUE_BASE = 3.0, 40e3
MR_LEV = (1.5, 2.0, 2.5, 3.0)
MUE_LEV = (15e3, 25e3, 40e3, 55e3, 70e3)
COV_LEV = (0.1, 0.2, 0.3, 0.4, 0.5)
TH_LEV = (0.5, 1.0, 2.0, 5.0, 20.0)


def pressures(scen, MR=MR_BASE):
    if scen.startswith("C3"):
        return C3_Q["q_eq"], C3_Q["q_tot"]
    m, B, L = EQUIP[scen[:2]]
    qeq = m * G0 / (B * L)
    return qeq, (1 + MR) * qeq


def load(name, cov, thx, thy):
    return np.load(os.path.join(D, f"{name}_cov{cov}_thx{thx}_thy{thy}.npy"))


def B_of(name):
    return {"C1": 3.5, "C2": 4.2, "C3": 21.0}.get(name[:2], 3.5)


def stats_single(name, G, muE=MUE_BASE, MR=MR_BASE, g_det=None):
    B = B_of(name); qeq, qt = pressures(name, MR)
    g = G[:, 0]
    d_tot = qt * B * g / muE * 1e3; d_eq = qeq * B * g / muE * 1e3
    out = dict(q_eq=qeq, q_tot=qt, delta_det_tot_mm=qt * B * g_det / muE * 1e3 if g_det else None,
               mean_tot_mm=d_tot.mean(), sd_tot_mm=d_tot.std(ddof=1), p95_tot_mm=np.percentile(d_tot, 95),
               P_tot_gt25=(d_tot > 25).mean(), mean_eq_mm=d_eq.mean(), p95_eq_mm=np.percentile(d_eq, 95),
               P_eq_gt13=(d_eq > 13).mean(), cov_delta=g.std(ddof=1) / g.mean())
    if G.shape[1] == 2:
        Lc = {"2B": 2, "4B": 4, "6B": 6}[name.split("_")[1]] * B
        dd = np.abs(G[:, 0] - G[:, 1]) * qeq * B / muE          # m, pos-instalacao
        beta = dd / Lc
        out.update(L_m=Lc, dd_eq_mean_mm=dd.mean() * 1e3, dd_eq_p95_mm=np.percentile(dd, 95) * 1e3,
                   P_beta_750=(beta > 1 / 750).mean(), P_beta_300=(beta > 1 / 300).mean(),
                   beta_p95=np.percentile(beta, 95))
    return out


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, c - h), min(1.0, c + h)]


def sobol_factorial(Y):
    """Y: array fatorial completo (um eixo por fator). Retorna S1 e ST por eixo."""
    V = Y.var()
    if V == 0:
        return [0.0] * Y.ndim, [0.0] * Y.ndim
    S1, ST = [], []
    for i in range(Y.ndim):
        others = tuple(j for j in range(Y.ndim) if j != i)
        S1.append(float(Y.mean(axis=others).var() / V))
        ST.append(float(1 - Y.mean(axis=i).var() / V))
    return S1, ST


if __name__ == "__main__":
    res = {}
    base = {r["cenario"]: r for r in json.load(open(os.path.join(D, "base.json")))}
    res["base"] = {}
    for name, r in base.items():
        G = load(name, 0.4, 2.0, 2.0)
        s = stats_single(name, G, g_det=r["g_det"])
        s.update(N=len(G), g_det=r["g_det"], g_mean=r["g_mean"], inc_pct=100 * (r["g_mean"] / r["g_det"] - 1))
        for k in ("P_tot_gt25", "P_eq_gt13", "P_beta_750", "P_beta_300"):
            if k in s: s[k + "_IC95"] = wilson(s[k], len(G))
        res["base"][name] = s

    # probabilidade de excedencia de 25 mm x muE (caso base, MR = 3)
    mues = np.linspace(10e3, 150e3, 141)
    fig, ax = plt.subplots(figsize=(6.3, 3.6), dpi=200)
    cols = {"C1": "#2a78d6", "C2": "#eb6834", "C3": "#1baf7a"}
    mk = {"C1": "o", "C2": "s", "C3": "^"}
    lab = {"C1": "C1 (G150)", "C2": "C2 (M210)", "C3": "C3 (moinho)"}
    curves = {}
    for name in ("C1", "C2", "C3"):
        G = load(name, 0.4, 2.0, 2.0); qeq, qt = pressures(name); B = B_of(name)
        P = np.array([(qt * B * G[:, 0] / mu * 1e3 > 25).mean() for mu in mues])
        curves[name] = P.tolist()
        ax.plot(mues / 1e3, P, color=cols[name], lw=2, label=lab[name], marker=mk[name], markevery=14, ms=5)
    ax.axvline(40, color="#52514e", lw=0.8, ls="--")
    ax.text(40.8, 0.93, "μE base = 40 MPa", fontsize=7.5, color="#52514e")
    ax.set_xlabel("Módulo de elasticidade médio μE (MPa)"); ax.set_ylabel("P(δ > 25 mm)")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=False, fontsize=8)
    ax.yaxis.set_major_formatter(VIRG)
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(D, "fig_pexc_muE.png")); plt.close(fig)
    res["curva_pexc_muE"] = dict(muE_kPa=mues.tolist(), **curves)

    # grades
    seq = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]  # sequencial (COV crescente)
    for gname, scen in (("grade_C2.json", "C2"), ("grade_C4_4B.json", "C4_4B")):
        path = os.path.join(D, gname)
        if not os.path.exists(path):
            continue
        rows = json.load(open(path)); gd = rows[0]["g_det"]
        tab = []
        for c, th in itertools.product(COV_LEV, TH_LEV):
            G = load(scen, c, th, th); s = stats_single(scen, G, g_det=gd)
            s.update(COV_E=c, theta=th, inc_pct=100 * (G[:, 0].mean() / gd - 1))
            tab.append(s)
        res["grade_" + scen] = tab
        # sensibilidade (fatorial completo)
        if scen == "C2":
            Y = np.zeros((5, 5, 5, 4))
            for (i, c), (j, th) in itertools.product(enumerate(COV_LEV), enumerate(TH_LEV)):
                g = load(scen, c, th, th)[:, 0]
                for (k, mu), (l, MR) in itertools.product(enumerate(MUE_LEV), enumerate(MR_LEV)):
                    qt = pressures(scen, MR)[1]
                    Y[i, j, k, l] = (qt * 4.2 * g / mu * 1e3 > 25).mean()
            S1, ST = sobol_factorial(Y)
            res["sobol_C2_P25"] = dict(fatores=["COV_E", "theta", "muE", "MR"], S1=S1, ST=ST)
        else:
            Y = np.zeros((5, 5, 5))
            for (i, c), (j, th) in itertools.product(enumerate(COV_LEV), enumerate(TH_LEV)):
                G = load(scen, c, th, th); qeq = pressures(scen)[0]
                for k, mu in enumerate(MUE_LEV):
                    beta = np.abs(G[:, 0] - G[:, 1]) * qeq * 3.5 / mu / 14.0
                    Y[i, j, k] = np.percentile(beta, 95)
            S1, ST = sobol_factorial(Y)
            res["sobol_C4_beta95"] = dict(fatores=["COV_E", "theta", "muE"], S1=S1, ST=ST,
                                          beta95_max=float(Y.max()), nota="P(beta > 1/750) = 0 em todo o fatorial")
        # figura
        fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.9), dpi=200)
        for i, c in enumerate(COV_LEV):
            sub = [t for t in tab if t["COV_E"] == c]
            th = [t["theta"] for t in sub]
            if scen == "C2":
                y1 = [t["cov_delta"] for t in sub]; y2 = [t["P_tot_gt25"] for t in sub]
            else:
                y1 = [t["dd_eq_mean_mm"] for t in sub]; y2 = [t["beta_p95"] * 1e4 for t in sub]
            axs[0].plot(th, y1, color=seq[i], lw=2, marker="o", ms=4, label=f"COV$_E$ = {c}".replace(".", ","))
            axs[1].plot(th, y2, color=seq[i], lw=2, marker="o", ms=4)
        if scen == "C2":
            axs[0].set_ylabel("COV do recalque"); axs[1].set_ylabel("P(δ > 25 mm)")
        else:
            axs[0].set_ylabel("Média de |Δδ| (mm)"); axs[1].set_ylabel("Percentil 95% de β (×10$^{-4}$)")
        for ax in axs:
            ax.set_xscale("log"); ax.set_xlabel("θ (m)"); ax.grid(alpha=0.25, lw=0.5)
            ax.xaxis.set_major_locator(FixedLocator(TH_LEV)); ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(VIRG); ax.yaxis.set_major_formatter(VIRG)
            for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
        axs[0].legend(frameon=False, fontsize=6.5)
        for ax, l in zip(axs, "ab"): ax.set_title(f"({l})", fontsize=9, loc="left")
        fig.tight_layout(); fig.savefig(os.path.join(D, f"fig_grade_{scen}.png")); plt.close(fig)

    # anisotropia
    ap = os.path.join(D, "aniso.json")
    if os.path.exists(ap):
        res["aniso"] = {}
        for r in json.load(open(ap)):
            G = load(r["cenario"], 0.4, 40.0, 2.0)
            s = stats_single(r["cenario"], G, g_det=r["g_det"])
            s.update(inc_pct=100 * (r["g_mean"] / r["g_det"] - 1))
            res["aniso"][r["cenario"]] = s

    conv = json.load(open(os.path.join(D, "convergencia.json")))
    res["convergencia"] = conv
    cast = lambda o: float(o) if isinstance(o, (np.floating,)) else o
    json.dump(res, open(os.path.join(D, "resumo.json"), "w"), indent=1, default=cast)
    print(json.dumps(res["base"], indent=1, default=cast))
    for k in ("sobol_C2_P25", "sobol_C4_beta95"):
        if k in res: print(k, res[k])
