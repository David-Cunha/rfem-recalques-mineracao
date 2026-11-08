"""Verificacao independente com elementos triangulares quadraticos (P2, scikit-fem).

Sapata rigida e rugosa (ux = 0, uy prescrito) de largura B sobre camada elastica de
espessura h com base rigida e rugosa, estado plano de deformacao, contornos laterais
com ux = 0. Recalque para P = 100 kN/m e E = 4.000 kPa, comparado com Milovic et al.
(1970 apud POULOS; DAVIS, 1974, Tab. 7.10) e com o modelo Q4 (rfem.py).

Uso: python verif_p2.py   (cerca de 2 min; o refino de 0,0125 m exige mais de 8 GB)
"""
import json
import numpy as np
from skfem import MeshTri, Basis, ElementVector, ElementTriP2, asm, solve, condense
from skfem.models.elasticity import linear_elasticity, lame_parameters

B, P, E, W = 1.0, 100.0, 4000.0, 24.0


def coords(a, b, h):
    return np.linspace(a, b, int(round((b - a) / h)) + 1)


def settlement(h_layer, nu, hf):
    """hf = tamanho do elemento junto a sapata (|x| <= 1,5 m); fora dela, 4 hf."""
    xs = np.unique(np.concatenate([coords(-W / 2, -1.5, 4 * hf), coords(-1.5, 1.5, hf), coords(1.5, W / 2, 4 * hf)]))
    ys = coords(-h_layer, 0.0, hf)
    m = MeshTri.init_tensor(xs, ys)
    ib = Basis(m, ElementVector(ElementTriP2()))
    lam, mu = lame_parameters(E, nu)                     # plano de deformacao
    K = asm(linear_elasticity(lam, mu), ib)
    tol = 1e-9
    base = ib.get_dofs(lambda x: x[1] < -h_layer + tol)
    lado = ib.get_dofs(lambda x: (x[0] < -W / 2 + tol) | (x[0] > W / 2 - tol))
    sap = ib.get_dofs(lambda x: (x[1] > -tol) & (np.abs(x[0]) <= B / 2 + tol))
    D = np.unique(np.concatenate([base.all(), lado.nodal["u^1"], lado.facet["u^1"],
                                  sap.all()]))
    u = np.zeros(ib.N)
    uy_sap = np.concatenate([sap.nodal["u^2"], sap.facet["u^2"]])
    u[uy_sap] = -1.0                                      # deslocamento unitario prescrito
    u = solve(*condense(K, np.zeros(ib.N), x=u, D=D))
    F = -(K @ u)[uy_sap].sum()                            # reacao vertical total (kN/m por m de recalque)
    return P / F * 1e3, ib.N


if __name__ == "__main__":
    out = {"refino_h2_nu03": [], "tabela": []}
    for hf in (0.1, 0.05, 0.025):
        d, n = settlement(2.0, 0.3, hf)
        out["refino_h2_nu03"].append(dict(hf=hf, gdl=int(n), delta_mm=round(d, 3)))
        print(f"h/B = 2, nu = 0,3, elemento {hf:.4f} m: {d:.3f} mm ({n} gdl)")
    d = [r["delta_mm"] for r in out["refino_h2_nu03"]]
    r = (d[1] - d[0]) / (d[2] - d[1]); out["richardson_mm"] = round(d[2] + (d[2] - d[1]) / (r - 1), 2)
    out["ordem_observada"] = round(np.log2(r), 2)
    print("Richardson:", out["richardson_mm"], "mm; ordem observada", out["ordem_observada"])
    milovic = {(0.005, 1): 19.75, (0.005, 2): 29.90, (0.005, 3): 37.54, (0.3, 1): 16.71, (0.3, 2): 26.71,
               (0.3, 3): 33.81, (0.45, 1): 10.43, (0.45, 2): 19.05, (0.45, 3): 24.85}
    for (nu, hb), ref in milovic.items():
        d, _ = settlement(float(hb), nu, 0.05)
        out["tabela"].append(dict(nu=nu, h_B=hb, P2_mm=round(d, 2), milovic_mm=ref, dif_pct=round(100 * (d / ref - 1), 1)))
        print(f"nu = {nu}, h/B = {hb}: P2 {d:.2f} mm x Milovic {ref} mm ({100 * (d / ref - 1):+.1f}%)")
    json.dump(out, open("out/verif_p2.json", "w"), indent=1)
