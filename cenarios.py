"""
Cenarios de fundacao de equipamentos de mineracao (C1-C4) - RFEM 2D com Monte Carlo.

Saida NORMALIZADA: com o modelo elastico linear e E(x) = muE * Etil(x) (Etil com media 1),
o recalque de cada sapata e
        delta = (P / muE) * g,       P = q * B (kN/m), muE em kPa, delta em m,
em que g (adimensional) depende so da geometria, de nu e do campo normalizado Etil.
Por isso uma unica simulacao por (geometria, COV_E, theta) serve para qualquer q e muE
(ver posproc.py).

Geometria (seção 4.3 da monografia):
  C1  B = 3,50 m  (Kubria G150)       H = 3B   dominio 6B
  C2  B = 4,20 m  (Kubria M210)       H = 3B   dominio 6B
  C3  B = 21,0 m  (radier de moinho)  H = B    dominio 6B
  C4  duas sapatas B = 3,50 m, afastamento entre eixos s = 2B, 4B, 6B; H = 3B; dominio s + 6B
Malha: nB elementos na largura B (nB par; 12 em C1, C2 e C4; 28 em C3, elemento de 0,75 m).
Correlacao markoviana rho = exp(-2 sqrt((dx/thx)^2 + (dy/thy)^2)); isotropica se thx = thy.
Campo por medias locais (Gauss 2x2) e Cholesky, como em rfem.py.

Uso:
  python cenarios.py convergencia          # estudo de malha deterministico
  python cenarios.py base [N]              # caso base (COV 0,4; theta 2 m) de todos os cenarios
  python cenarios.py grade CENARIO [N]     # grade COV x theta (CENARIO = C2 ou C4)
  python cenarios.py aniso [N]             # thh = 40 m, thv = 2 m em C2 e C4
  python cenarios.py h4 [N]                # C4_4B, COV 0,4, theta = 50 e 200 m (teste da hipotese H4)
"""
import sys, os, json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import cholesky
from rfem import mesh, ke_unit

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "cenarios")
os.makedirs(OUT, exist_ok=True)
NU = 0.3
SEED = 20261107

GEOM = {
    "C1": dict(B=3.50, Hr=3.0, s=None, nB=12),
    "C2": dict(B=4.20, Hr=3.0, s=None, nB=12),
    "C3": dict(B=21.0, Hr=1.0, s=None, nB=28),
    "C4_2B": dict(B=3.50, Hr=3.0, s=2.0, nB=12),
    "C4_4B": dict(B=3.50, Hr=3.0, s=4.0, nB=12),
    "C4_6B": dict(B=3.50, Hr=3.0, s=6.0, nB=12),
}


class Footings:
    """Sapatas corridas rigidas e rugosas, cargas iguais; base fixa, laterais u_x = 0."""

    def __init__(self, B, Hr, s=None, nB=12, nu=NU):
        H = Hr * B
        W = (s * B if s else 0.0) + 6 * B
        nx, ny = int(round(W / B * nB)), int(round(Hr * nB))
        self.B, self.H, self.W, self.nx, self.ny = B, H, W, nx, ny
        self.L = s * B if s else None
        centers = [-s * B / 2, s * B / 2] if s else [0.0]
        self.nodes, self.conn, self.cent, self.a, self.b = mesh(nx, ny, W, H)
        self.ne = len(self.conn); self.ndof = 2 * len(self.nodes)
        K0 = ke_unit(self.a, self.b, nu)
        dofs = np.zeros((self.ne, 8), int); dofs[:, 0::2] = 2 * self.conn; dofs[:, 1::2] = 2 * self.conn + 1
        self.rows = np.repeat(dofs, 8, axis=1).ravel(); self.cols = np.tile(dofs, (1, 8)).ravel()
        self.K0 = K0.ravel()
        x, y = self.nodes[:, 0], self.nodes[:, 1]; tol = 1e-6 * B
        fixed = set()
        for n in np.where(np.abs(y + H) < tol)[0]: fixed |= {2 * n, 2 * n + 1}
        for n in np.where(np.abs(np.abs(x) - W / 2) < tol)[0]: fixed.add(2 * n)
        self.feet = [np.where((np.abs(y) < tol) & (np.abs(x - c) <= B / 2 + tol))[0] for c in centers]
        for f in self.feet:
            assert len(f) == nB + 1, "bordas da sapata devem coincidir com nos"
            for n in f: fixed |= {2 * n, 2 * n + 1}
        self.fd = np.array(sorted(fixed)); m = np.ones(self.ndof, bool); m[self.fd] = False
        self.free = np.where(m)[0]

    def g(self, Etil):
        """Recalques normalizados g = delta * muE / P de cada sapata (P = 1, muE = 1)."""
        K = sp.csr_matrix(((Etil[:, None] * self.K0[None, :]).ravel(), (self.rows, self.cols)),
                          shape=(self.ndof,) * 2)
        Kff = K[self.free][:, self.free].tocsc(); Kfc = K[self.free][:, self.fd]
        lu = spla.splu(Kff); nf = len(self.feet); R = np.zeros((nf, nf))
        for j, fj in enumerate(self.feet):
            u = np.zeros(self.ndof); u[2 * fj + 1] = -1.0
            u[self.free] = lu.solve(-(Kfc @ u[self.fd]))
            r = K @ u
            for i, fi in enumerate(self.feet): R[i, j] = -r[2 * fi + 1].sum()
        return np.linalg.solve(R, np.ones(nf))


def local_avg_corr(cent, a, b, thx, thy):
    g = 1 / np.sqrt(3)
    offs = [(sx * g * a / 2, sy * g * b / 2) for sx in (-1, 1) for sy in (-1, 1)]
    n = len(cent); C = np.zeros((n, n))
    for ox1, oy1 in offs:
        for ox2, oy2 in offs:
            dx = (cent[:, None, 0] + ox1) - (cent[None, :, 0] + ox2)
            dy = (cent[:, None, 1] + oy1) - (cent[None, :, 1] + oy2)
            C += np.exp(-2 * np.sqrt((dx / thx) ** 2 + (dy / thy) ** 2))
    return C / 16.0


def simulate(name, cov, thx, thy, N, rng):
    m = Footings(**GEOM[name])
    t0 = time.time()
    g_det = m.g(np.ones(m.ne))
    L = cholesky(local_avg_corr(m.cent, m.a, m.b, thx, thy) + 1e-10 * np.eye(m.ne), lower=True)
    s2 = np.log(1 + cov ** 2); mu = -s2 / 2
    G = np.empty((N, len(m.feet)))
    for k0 in range(0, N, 200):
        k1 = min(N, k0 + 200)
        Et = np.exp(mu + np.sqrt(s2) * (L @ rng.standard_normal((m.ne, k1 - k0))))
        for k in range(k1 - k0):
            G[k0 + k] = m.g(Et[:, k])
    tag = f"{name}_cov{cov}_thx{thx}_thy{thy}"
    np.save(os.path.join(OUT, tag + ".npy"), G)
    g1 = G[:, 0]
    row = dict(cenario=name, B=m.B, H=m.H, W=m.W, malha=[m.nx, m.ny], L=m.L, COV_E=cov,
               theta_x=thx, theta_y=thy, N=N, g_det=float(g_det[0]),
               g_mean=float(g1.mean()), g_std=float(g1.std(ddof=1)),
               g_p05=float(np.percentile(g1, 5)), g_p50=float(np.percentile(g1, 50)),
               g_p95=float(np.percentile(g1, 95)), t_s=time.time() - t0)
    if len(m.feet) == 2:
        dg = np.abs(G[:, 0] - G[:, 1])
        row.update(dg_mean=float(dg.mean()), dg_std=float(dg.std(ddof=1)),
                   dg_p95=float(np.percentile(dg, 95)),
                   rho_12=float(np.corrcoef(G[:, 0], G[:, 1])[0, 1]))
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    return row


def save(rows, fname):
    path = os.path.join(OUT, fname)
    old = json.load(open(path)) if os.path.exists(path) else []
    key = lambda r: (r["cenario"], r["COV_E"], r["theta_x"], r["theta_y"])
    merged = {key(r): r for r in old}; merged.update({key(r): r for r in rows})
    json.dump(sorted(merged.values(), key=key), open(path, "w"), indent=1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    if mode == "convergencia":
        res = []
        for name, nBs in [("C1", (6, 12, 24, 48)), ("C3", (14, 28, 56))]:
            for nB in nBs:
                gm = dict(GEOM[name]); gm["nB"] = nB
                m = Footings(**gm); t = time.time()
                gd = float(m.g(np.ones(m.ne))[0])
                res.append(dict(cenario=name, nB=nB, h_m=m.B / nB, malha=[m.nx, m.ny], g_det=gd, t_s=time.time() - t))
                print(res[-1], flush=True)
        json.dump(res, open(os.path.join(OUT, "convergencia.json"), "w"), indent=1)
    elif mode == "base":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
        rng = np.random.default_rng(SEED)
        rows = [simulate(n, 0.4, 2.0, 2.0, N, rng) for n in GEOM]
        save(rows, "base.json")
    elif mode == "grade":
        name = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
        name = {"C4": "C4_4B"}.get(name, name)
        rng = np.random.default_rng(SEED + (1 if name.startswith("C4") else 0))
        rows = [simulate(name, c, th, th, N, rng) for c in (0.1, 0.2, 0.3, 0.4, 0.5)
                for th in (0.5, 1.0, 2.0, 5.0, 20.0)]
        save(rows, f"grade_{name}.json")
    elif mode == "aniso":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
        rng = np.random.default_rng(SEED + 2)
        rows = [simulate(n, 0.4, 40.0, 2.0, N, rng) for n in ("C2", "C4_4B")]
        save(rows, "aniso.json")
    elif mode == "h4":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
        rng = np.random.default_rng(SEED + 3)
        rows = [simulate("C4_4B", 0.4, th, th, N, rng) for th in (50.0, 200.0)]
        save(rows, "h4.json")
