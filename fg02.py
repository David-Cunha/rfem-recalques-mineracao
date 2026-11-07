"""
Reproducao dos exemplos numericos de Fenton e Griffiths (2002), J. Geotech. Geoenviron. Eng. 128(5):381-390.
Exemplo 1 (secao 3.1): sapata unica, B = 2 m, camada H = 10 m, P = 1000 kN/m, muE = 40 MPa, sdE = 40 MPa,
                       theta = 3,0 m (o ponto decimal se perde na extracao do PDF), nu = 0,25. Publicado: delta_det = 0,03531 m; MC (5000): media 0,0562 m,
                       dp 0,0201 m, P[delta > 0,10 m] = 0,032.
Exemplo 2 (secao 4.1): duas sapatas B = 2 m, afastamento 10 m entre eixos, theta = 1,0 m (o texto diz depois "se theta aumentar para 10,0 m"). Publicado:
                       delta_det = 0,03578 m; MC (5000): media 0,0530 m, dp 0,0081 m, rho ~ 0, E|Delta| = 0,009 m,
                       P[|Delta| > 0,028 m] = 0,0204.
Dominio 30 m x 10 m (proporcao 3:1 do artigo), malha 60 x 20 Q4, laterais u_x = 0, base fixa,
sapatas rigidas e rugosas. Campo lognormal, correlacao markoviana, medias locais por elemento (Cholesky).
Uso: python3 fg02.py [N]   (padrao N = 5000)
"""
import sys, json, os, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from rfem import mesh, ke_unit, local_avg_cov
from scipy.linalg import cholesky

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

class MultiFooting:
    def __init__(self, nx, ny, W, H, centers, B, nu):
        self.nodes, self.conn, self.cent, self.a, self.b = mesh(nx, ny, W, H)
        self.ne = len(self.conn); self.ndof = 2 * len(self.nodes)
        K0 = ke_unit(self.a, self.b, nu)
        dofs = np.zeros((self.ne, 8), int); dofs[:, 0::2] = 2 * self.conn; dofs[:, 1::2] = 2 * self.conn + 1
        self.rows = np.repeat(dofs, 8, axis=1).ravel(); self.cols = np.tile(dofs, (1, 8)).ravel(); self.K0 = K0.ravel()
        x, y = self.nodes[:, 0], self.nodes[:, 1]; tol = 1e-9
        fixed = set()
        for n in np.where(np.abs(y + H) < tol)[0]: fixed |= {2 * n, 2 * n + 1}
        for n in np.where(np.abs(np.abs(x) - W / 2) < tol)[0]: fixed.add(2 * n)
        self.feet = [np.where((np.abs(y) < tol) & (np.abs(x - c) <= B / 2 + tol))[0] for c in centers]
        for f in self.feet:
            for n in f: fixed |= {2 * n, 2 * n + 1}      # rugosa: u_x = 0; u_y prescrito
        self.fd = np.array(sorted(fixed)); m = np.ones(self.ndof, bool); m[self.fd] = False; self.free = np.where(m)[0]

    def settlements(self, E, P):
        """Recalques de sapatas rigidas sob cargas iguais P (kN/m), por superposicao."""
        K = sp.csr_matrix(((E[:, None] * self.K0[None, :]).ravel(), (self.rows, self.cols)), shape=(self.ndof,) * 2)
        Kff = K[self.free][:, self.free].tocsc(); Kfc = K[self.free][:, self.fd]
        lu = spla.splu(Kff); nf = len(self.feet); R = np.zeros((nf, nf))
        for j, fj in enumerate(self.feet):
            u = np.zeros(self.ndof); u[2 * fj + 1] = -1.0
            u[self.free] = lu.solve(-(Kfc @ u[self.fd]))
            r = K @ u
            for i, fi in enumerate(self.feet): R[i, j] = -r[2 * fi + 1].sum()
        return np.linalg.solve(R, np.full(nf, P))

def fields(m, theta, muE, sdE, N, rng):
    C = local_avg_cov(m.cent, m.a, m.b, theta); C[np.diag_indices_from(C)] += 1e-10
    L = cholesky(C, lower=True); s2 = np.log(1 + (sdE / muE) ** 2); mu = np.log(muE) - s2 / 2
    return np.exp(mu + np.sqrt(s2) * (L @ rng.standard_normal((m.ne, N))))

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    rng = np.random.default_rng(20261106); res = {}
    muE, sdE, P = 40e3, 40e3, 1000.0     # kPa, kPa, kN/m
    # Exemplo 1
    m1 = MultiFooting(60, 20, 30.0, 10.0, [0.0], 2.0, 0.25)
    d1 = m1.settlements(np.full(m1.ne, muE), P)[0]
    t = time.time(); Ef = fields(m1, 3.0, muE, sdE, N, rng)
    s = np.array([m1.settlements(Ef[:, k], P)[0] for k in range(N)])
    res["ex1"] = {"delta_det_m": d1, "pub_delta_det_m": 0.03531, "N": N, "mean_m": s.mean(), "pub_mean_m": 0.0562,
                  "std_m": s.std(ddof=1), "pub_std_m": 0.0201, "P_gt_010": float((s > 0.10).mean()), "pub_P_gt_010": 0.032,
                  "t_s": time.time() - t}
    print("ex1", json.dumps(res["ex1"]), flush=True)
    # Exemplo 2
    m2 = MultiFooting(60, 20, 30.0, 10.0, [-5.0, 5.0], 2.0, 0.25)
    d2 = m2.settlements(np.full(m2.ne, muE), P)
    t = time.time(); Ef = fields(m2, 1.0, muE, sdE, N, rng)
    S = np.array([m2.settlements(Ef[:, k], P) for k in range(N)]); D = np.abs(S[:, 0] - S[:, 1])
    res["ex2"] = {"delta_det_m": d2.tolist(), "pub_delta_det_m": 0.03578, "N": N,
                  "mean_m": S.mean(), "pub_mean_m": 0.0530, "std_m": S.std(ddof=1), "pub_std_m": 0.0081,
                  "rho": float(np.corrcoef(S[:, 0], S[:, 1])[0, 1]), "pub_rho": -0.014,
                  "mean_absdiff_m": D.mean(), "pub_mean_absdiff_m": 0.009,
                  "P_absdiff_gt_0028": float((D > 0.028).mean()), "pub_P_absdiff_gt_0028": 0.0204, "t_s": time.time() - t}
    print("ex2", json.dumps(res["ex2"]), flush=True)
    json.dump(res, open(os.path.join(OUT, "fg02_resultados.json"), "w"), indent=2)
