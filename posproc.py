"""
Pos-processamento dos cenarios C1, C2 e C3 (le out/cenarios/*.npy e *.json; gera tabelas e figuras).

Recalque dimensional a partir do recalque normalizado g (ver cenarios.py):
        delta = q * B * g / (muE * eps)      (q e muE em kPa, B em m, delta em m)
eps: incerteza de transformacao/media do modulo (lognormal, media 1, COV_t); eps = 1 no caso base.
Pressoes (Eq. 12): q = (1 + MR) m g0 / (B L) (total) e q_eq = m g0 / (B L) (pos-instalacao).
Limiares hipoteticos (Quadro 6): 25 mm (total, q); 13 mm (equipamento, q_eq);
beta = 1/750 e 1/300 (C3, q_eq), com L = vao do elemento que liga as duas fundacoes (calha ou transportador).

Criterios de modulo minimo para delta <= 25 mm (C1 e C2, caso base):
  (a) deterministico com o modulo medio;
  (b) deterministico com o modulo caracteristico E_k = (0,4/0,6) muE (limite inferior de Sandroni);
  (c) probabilistico, P(delta > 25 mm) <= 5 %, so variabilidade espacial;
  (d) como (c), com incerteza de transformacao COV_t = 0,2 (hipotese).

Sensibilidade (C2): (i) indices de Sobol por fatorial completo na faixa larga muE = 15-70 MPa
(saida P(delta > 25 mm), quase um degrau); (ii) condicional, muE = 35-45 MPa; (iii) condicional a
muE = 40 MPa com saida continua (percentil 95 % do recalque). C3: percentil 95 % de beta.

Testes de lognormalidade (Kolmogorov-Smirnov e Anderson-Darling, 5 %) sobre ln g de todos os conjuntos.

Uso: python posproc.py
"""
import os, json, glob, itertools
import numpy as np
import scipy.stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator

VIRG = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "cenarios")
G0 = 9.81
EQUIP = {"C1": (37.0, 3.50, 3.955), "C2": (77.0, 4.20, 4.866), "C3": (37.0, 3.50, 3.955)}
MR_BASE, MUE_BASE, COVT = 3.0, 40e3, 0.2
MR_LEV = (1.5, 2.0, 2.5, 3.0)
MUE_LEV = (15e3, 25e3, 40e3, 55e3, 70e3)
MUE_COND = (35e3, 37.5e3, 40e3, 42.5e3, 45e3)
COV_LEV = (0.1, 0.2, 0.3, 0.4, 0.5)
TH_LEV = (0.5, 1.0, 2.0, 5.0, 20.0)
SANDRONI_KF = 0.4 / 0.6


def pressures(scen, MR=MR_BASE):
    m, B, L = EQUIP[scen[:2]]
    qeq = m * G0 / (B * L)
    return qeq, (1 + MR) * qeq


def B_of(name):
    return EQUIP[name[:2]][1]


def load(name, cov, thx, thy, extra=""):
    return np.load(os.path.join(D, f"{name}_cov{cov}_thx{thx}_thy{thy}{extra}.npy"))


def eps_samples(n, covt, seed=20261108):
    if covt == 0:
        return np.ones(n)
    s2 = np.log(1 + covt ** 2)
    return np.exp(-s2 / 2 + np.sqrt(s2) * np.random.default_rng(seed).standard_normal(n))


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, c - h), min(1.0, c + h)]


def stats_single(name, G, muE=MUE_BASE, MR=MR_BASE, g_det=None, covt=0.0):
    B = B_of(name); qeq, qt = pressures(name, MR)
    eps = eps_samples(len(G), covt)
    g = G[:, 0] / eps
    d_tot = qt * B * g / muE * 1e3; d_eq = qeq * B * g / muE * 1e3
    out = dict(q_eq=qeq, q_tot=qt, delta_det_tot_mm=qt * B * g_det / muE * 1e3 if g_det else None,
               mean_tot_mm=d_tot.mean(), sd_tot_mm=d_tot.std(ddof=1), p95_tot_mm=np.percentile(d_tot, 95),
               P_tot_gt25=(d_tot > 25).mean(), mean_eq_mm=d_eq.mean(), p95_eq_mm=np.percentile(d_eq, 95),
               P_eq_gt13=(d_eq > 13).mean(), cov_delta=g.std(ddof=1) / g.mean())
    if G.shape[1] == 2:
        Lc = {"2B": 2, "4B": 4, "6B": 6}[name.split("_")[1]] * B
        dd = np.abs(G[:, 0] - G[:, 1]) / eps * qeq * B / muE
        beta = dd / Lc
        out.update(L_m=Lc, dd_eq_mean_mm=dd.mean() * 1e3, dd_eq_p95_mm=np.percentile(dd, 95) * 1e3,
                   P_beta_750=(beta > 1 / 750).mean(), P_beta_300=(beta > 1 / 300).mean(),
                   beta_p95=np.percentile(beta, 95))
    return out


def sobol_factorial(Y):
    V = Y.var()
    if V == 0:
        return [0.0] * Y.ndim, [0.0] * Y.ndim
    S1, ST = [], []
    for i in range(Y.ndim):
        others = tuple(j for j in range(Y.ndim) if j != i)
        S1.append(float(Y.mean(axis=others).var() / V))
        ST.append(float(1 - Y.mean(axis=i).var() / V))
    return S1, ST


def mu_min(g, qB, lim_mm=25.0, p=0.05):
    """Menor muE (kPa) com P(q B g / muE > lim) <= p, para amostra g (ja dividida por eps)."""
    return qB * np.quantile(g, 1 - p) / (lim_mm / 1e3)


def style(ax):
    ax.grid(alpha=0.25, lw=0.5)
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)


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

    # criterios de modulo minimo (C1, C2)
    res["modulo_minimo_MPa"] = {}
    for name in ("C1", "C2"):
        G = load(name, 0.4, 2.0, 2.0); r = base[name]; qB = pressures(name)[1] * B_of(name)
        det = qB * r["g_det"] / 0.025
        res["modulo_minimo_MPa"][name] = dict(
            a_det_media=det / 1e3, b_det_caracteristico=det / SANDRONI_KF / 1e3,
            c_prob_variabilidade=mu_min(G[:, 0], qB) / 1e3,
            d_prob_com_transformacao=mu_min(G[:, 0] / eps_samples(len(G), COVT), qB) / 1e3,
            N_a=(det / 1e3 / 0.6) ** (1 / 1.4), N_b=(det / 1e3 / 0.4) ** (1 / 1.4),
            N_c=(mu_min(G[:, 0], qB) / 1e3 / 0.6) ** (1 / 1.4),
            N_d=(mu_min(G[:, 0] / eps_samples(len(G), COVT), qB) / 1e3 / 0.6) ** (1 / 1.4))

    # Figura: P(delta > 25 mm) x muE
    mues = np.linspace(10e3, 70e3, 121)
    fig, ax = plt.subplots(figsize=(6.3, 3.4), dpi=200)
    cols = {"C1": "#2a78d6", "C2": "#eb6834"}; mk = {"C1": "o", "C2": "s"}
    lab = {"C1": "C1 (G150)", "C2": "C2 (M210)"}
    curves = {}
    for name in ("C1", "C2"):
        G = load(name, 0.4, 2.0, 2.0); qeq, qt = pressures(name); B = B_of(name)
        for covt, ls, suf in ((0.0, "-", ""), (COVT, "--", " + transf.")):
            g = G[:, 0] / eps_samples(len(G), covt)
            P = np.array([(qt * B * g / mu * 1e3 > 25).mean() for mu in mues])
            curves[name + ("_transf" if covt else "")] = P.tolist()
            ax.plot(mues / 1e3, P, color=cols[name], lw=2, ls=ls, label=lab[name] + suf,
                    marker=mk[name] if not covt else None, markevery=15, ms=4.5)
    ax.axhline(0.05, color="#52514e", lw=0.8, ls=":")
    ax.text(68, 0.07, "5%", fontsize=7.5, color="#52514e", ha="right")
    ax.set_xlabel("Módulo de elasticidade médio μE (MPa)"); ax.set_ylabel("P(δ > 25 mm)")
    ax.set_ylim(-0.02, 1.02); ax.yaxis.set_major_formatter(VIRG); style(ax)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(D, "fig_pexc_muE.png")); plt.close(fig)
    res["curva_pexc_muE"] = dict(muE_kPa=mues.tolist(), **curves)

    # grades e sensibilidade
    seq = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
    for gname, scen in (("grade_C2.json", "C2"), ("grade_C3_4B.json", "C3_4B")):
        rows = json.load(open(os.path.join(D, gname))); gd = rows[0]["g_det"]
        tab = []
        for c, th in itertools.product(COV_LEV, TH_LEV):
            G = load(scen, c, th, th); s = stats_single(scen, G, g_det=gd)
            s.update(COV_E=c, theta=th, inc_pct=100 * (G[:, 0].mean() / gd - 1))
            tab.append(s)
        res["grade_" + scen] = tab
        if scen == "C2":
            qB = lambda MR: pressures(scen, MR)[1] * 4.2
            Y = np.zeros((5, 5, 5, 4)); Yc = np.zeros((5, 5, 5, 4)); Y95 = np.zeros((5, 5, 4))
            for (i, c), (j, th) in itertools.product(enumerate(COV_LEV), enumerate(TH_LEV)):
                g = load(scen, c, th, th)[:, 0]
                for l, MR in enumerate(MR_LEV):
                    for k, mu in enumerate(MUE_LEV):
                        Y[i, j, k, l] = (qB(MR) * g / mu * 1e3 > 25).mean()
                    for k, mu in enumerate(MUE_COND):
                        Yc[i, j, k, l] = (qB(MR) * g / mu * 1e3 > 25).mean()
                    Y95[i, j, l] = qB(MR) * np.percentile(g, 95) / MUE_BASE * 1e3
            S1, ST = sobol_factorial(Y)
            res["sobol_C2_P25_faixa_larga"] = dict(fatores=["COV_E", "theta", "muE", "MR"], S1=S1, ST=ST)
            S1, ST = sobol_factorial(Yc)
            res["sobol_C2_P25_muE_35_45"] = dict(fatores=["COV_E", "theta", "muE", "MR"], S1=S1, ST=ST)
            S1, ST = sobol_factorial(Y95)
            res["sobol_C2_P95_muE40"] = dict(fatores=["COV_E", "theta", "MR"], S1=S1, ST=ST)
        else:
            Y = np.zeros((5, 5, 5))
            for (i, c), (j, th) in itertools.product(enumerate(COV_LEV), enumerate(TH_LEV)):
                G = load(scen, c, th, th); qeq = pressures(scen)[0]
                for k, mu in enumerate(MUE_LEV):
                    Y[i, j, k] = np.percentile(np.abs(G[:, 0] - G[:, 1]) * qeq * 3.5 / mu / 14.0, 95)
            S1, ST = sobol_factorial(Y)
            res["sobol_C3_beta95"] = dict(fatores=["COV_E", "theta", "muE"], S1=S1, ST=ST,
                                          beta95_max=float(Y.max()), nota="P(beta > 1/750) = 0 em todo o fatorial")
        fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.9), dpi=200)
        for i, c in enumerate(COV_LEV):
            sub = [t for t in tab if t["COV_E"] == c]; th = [t["theta"] for t in sub]
            if scen == "C2":
                y1 = [t["cov_delta"] for t in sub]; y2 = [t["P_tot_gt25"] for t in sub]
            else:
                y1 = [t["dd_eq_mean_mm"] for t in sub]; y2 = [t["beta_p95"] * 1e4 for t in sub]
            lbl = f"COV$_E$ = {c}".replace(".", ",")
            axs[0].plot(th, y1, color=seq[i], lw=2, marker="o", ms=4, label=lbl)
            axs[1].plot(th, y2, color=seq[i], lw=2, marker="o", ms=4)
        if scen == "C2":
            axs[0].set_ylabel("COV do recalque"); axs[1].set_ylabel("P(δ > 25 mm)")
        else:
            axs[0].set_ylabel("Média de |Δδ| (mm)"); axs[1].set_ylabel("Percentil 95% de β (×10$^{-4}$)")
        for ax, l in zip(axs, "ab"):
            ax.set_xscale("log"); ax.set_xlabel("θ (m)"); style(ax)
            ax.xaxis.set_major_locator(FixedLocator(TH_LEV)); ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(VIRG); ax.yaxis.set_major_formatter(VIRG)
            ax.set_title(f"({l})", fontsize=9, loc="left")
        axs[0].legend(frameon=False, fontsize=6.5)
        fig.tight_layout(); fig.savefig(os.path.join(D, f"fig_grade_{scen}.png")); plt.close(fig)

    # anisotropia, tendencia, H4
    res["aniso"] = {}
    for r in json.load(open(os.path.join(D, "aniso.json"))):
        G = load(r["cenario"], 0.4, 40.0, 2.0)
        s = stats_single(r["cenario"], G, g_det=r["g_det"]); s.update(inc_pct=100 * (r["g_mean"] / r["g_det"] - 1))
        res["aniso"][r["cenario"]] = s
    tp = os.path.join(D, "tendencia.json")
    if os.path.exists(tp):
        r = json.load(open(tp))[0]; G = load("C2", 0.4, 2.0, 2.0, "_tend")
        s = stats_single("C2", G, g_det=r["g_det"]); s.update(inc_pct=100 * (r["g_mean"] / r["g_det"] - 1))
        res["tendencia_C2"] = s
    h4 = []
    qeq = pressures("C3")[0]
    pts = [(t["theta"], t["dd_eq_mean_mm"]) for t in res["grade_C3_4B"] if t["COV_E"] == 0.4]
    for r in json.load(open(os.path.join(D, "h4.json"))):
        G = load("C3_4B", 0.4, r["theta_x"], r["theta_y"])
        pts.append((r["theta_x"], float(np.abs(G[:, 0] - G[:, 1]).mean() * qeq * 3.5 / MUE_BASE * 1e3)))
    res["h4_dd_mean_mm"] = sorted(pts)

    # testes de lognormalidade
    tests = []
    for f in sorted(glob.glob(os.path.join(D, "*.npy"))):
        l = np.log(np.load(f)[:, 0]); z = (l - l.mean()) / l.std(ddof=1)
        ad = st.anderson(l, "norm")
        tests.append(dict(arquivo=os.path.basename(f), KS_p=float(st.kstest(z, "norm").pvalue),
                          AD=float(ad.statistic), AD_crit5=float(ad.critical_values[2]),
                          AD_rejeita=bool(ad.statistic > ad.critical_values[2])))
    res["lognormalidade"] = dict(n=len(tests), rejeicoes_KS=sum(t["KS_p"] < 0.05 for t in tests),
                                 rejeicoes_AD=sum(t["AD_rejeita"] for t in tests), conjuntos=tests)

    res["convergencia"] = json.load(open(os.path.join(D, "convergencia.json")))
    cast = lambda o: float(o) if isinstance(o, np.floating) else (bool(o) if isinstance(o, np.bool_) else o)
    json.dump(res, open(os.path.join(D, "resumo.json"), "w"), indent=1, default=cast)
    for k in ("modulo_minimo_MPa", "sobol_C2_P25_faixa_larga", "sobol_C2_P25_muE_35_45", "sobol_C2_P95_muE40",
              "sobol_C3_beta95", "h4_dd_mean_mm"):
        print(k, json.dumps(res[k], default=cast))
    print("lognormalidade: n =", res["lognormalidade"]["n"], "KS rej =", res["lognormalidade"]["rejeicoes_KS"],
          "AD rej =", res["lognormalidade"]["rejeicoes_AD"])
