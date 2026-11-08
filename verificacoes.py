"""
Verificacoes complementares citadas no texto (seções 5.2, 5.5 e 5.9 da monografia).

1) Teste analitico do gerador (Fenton e Griffiths, 2002, Eq. 11 da monografia):
   delta = delta_det * muE / E_g, ln E_g ~ N(mu_lnE, gamma * s2)  =>
   E[delta]/delta_det = exp(s2 (1 + gamma) / 2),  sd(ln delta) = sqrt(gamma s2),
   com s2 = ln(1 + COV^2) e gamma = funcao de variancia da media sobre B x H (aqui B = 1 m, H = 2 m).
   Limites: gamma -> 0 e gamma -> 1.
2) Testes de lognormalidade (Kolmogorov-Smirnov e Anderson-Darling, 5 %) nos 15 casos da reproducao.
3) Fatores de Harr (1966 apud VELLOSO; LOPES, 2011, p. 94, Tab. 5.2): razao entre o fator h*Ih da
   faixa infinita e o do retangulo das fundacoes de C1 e C2 (carga flexivel, centro, camada finita).

Uso: python verificacoes.py   (le out/resultados.json e out/s_*.npy; grava out/verificacoes.json)
"""
import os, json
import numpy as np
import scipy.stats as st

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def rho(dx, dy, th, kind):
    if kind == "markov1":
        return np.exp(-2 * np.hypot(dx, dy) / th)
    tx, ty = 4 * np.abs(dx) / th, 4 * np.abs(dy) / th
    return (1 + tx) * (1 + ty) * np.exp(-(tx + ty))


def gamma(B, H, th, kind, n=48):
    """Funcao de variancia: media de rho sobre pares de pontos de B x H (pontos medios, n x 2n)."""
    x = (np.arange(n) + 0.5) * B / n; y = (np.arange(2 * n) + 0.5) * H / (2 * n)
    X, Y = np.meshgrid(x, y); X, Y = X.ravel(), Y.ravel()
    tot = 0.0
    for i in range(0, len(X), 512):
        tot += rho(X[i:i + 512, None] - X[None, :], Y[i:i + 512, None] - Y[None, :], th, kind).sum()
    return tot / len(X) ** 2


if __name__ == "__main__":
    out = {}
    r = json.load(open(os.path.join(D, "resultados.json")))
    kind = r.get("correlacao", "markov1")
    rows = []
    for c in r["monte_carlo"]:
        s2 = np.log(1 + c["COV_E"] ** 2); g = gamma(1.0, 2.0, c["theta_m"], kind)
        rows.append(dict(COV_E=c["COV_E"], theta=c["theta_m"], gamma=g,
                         inc_sim=c["inc_pct"], inc_min=100 * (np.exp(s2 / 2) - 1),
                         inc_max=100 * (np.exp(s2) - 1), inc_pred=100 * (np.exp(s2 * (1 + g) / 2) - 1),
                         sdln_sim=c["sd_ln"], sdln_pred=float(np.sqrt(g * s2))))
    out["teste_analitico"] = dict(correlacao=kind, casos=rows)
    tests = []
    for c in r["monte_carlo"]:
        s = np.load(os.path.join(D, f"s_cov{c['COV_E']}_th{c['theta_m']}.npy")); l = np.log(s)
        ad = st.anderson(l, "norm")
        tests.append(dict(COV_E=c["COV_E"], theta=c["theta_m"],
                          KS_p=float(st.kstest((l - l.mean()) / l.std(ddof=1), "norm").pvalue),
                          AD=float(ad.statistic), AD_rejeita=bool(ad.statistic > ad.critical_values[2])))
    out["lognormalidade_reproducao"] = tests
    # Harr (1966) - Tab. 5.2 de Velloso e Lopes (2011): h*Ih para h/a = 5 e 7 (a = B/2)
    m_cols = [1, 2, 3, 5, 7, 10]
    t5 = [0.800, 1.010, 1.119, 1.201, 1.238, 1.256]; s5 = 1.323
    t7 = [0.842, 1.094, 1.223, 1.346, 1.402, 1.442]; s7 = 1.532
    harr = {}
    for name, B, L in (("C1", 3.50, 3.955), ("C2", 4.20, 4.866)):
        m = L / B
        rect = 0.5 * (np.interp(m, m_cols, t5) + np.interp(m, m_cols, t7))  # h/a = 6
        harr[name] = dict(L_B=m, h_a=6.0, razao_faixa_retangulo=0.5 * (s5 + s7) / rect)
    out["harr"] = harr
    json.dump(out, open(os.path.join(D, "verificacoes.json"), "w"), indent=1)
    for x in rows:
        print("COV %.1f th %5.2f gamma %.3f inc sim %5.1f pred %5.1f [%4.1f, %4.1f] | sdln sim %.3f pred %.3f" % (
            x["COV_E"], x["theta"], x["gamma"], x["inc_sim"], x["inc_pred"], x["inc_min"], x["inc_max"],
            x["sdln_sim"], x["sdln_pred"]))
    print("AD rejeicoes:", sum(t["AD_rejeita"] for t in tests), "de", len(tests),
          "| KS rejeicoes:", sum(t["KS_p"] < 0.05 for t in tests))
    print(harr)
