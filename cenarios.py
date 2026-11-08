"""
Cenarios de fundacao de equipamentos de mineracao (C1-C3) - RFEM 2D com Monte Carlo.

Saida NORMALIZADA: com o modelo elastico linear e E(x) = muE * Etil(x) (Etil com media 1),
o recalque de cada sapata e
        delta = (P / muE) * g,       P = q * B (kN/m), muE em kPa, delta em m,
em que g (adimensional) depende so da geometria, de nu e do campo normalizado Etil.
Por isso uma unica simulacao por (geometria, COV_E, theta) serve para qualquer q e muE
(ver posproc.py).

Geometria (seção 4.3 da monografia):
  C1  B = 3,50 m  (Kubria G150)       H = 3B   dominio 6B
  C2  B = 4,20 m  (Kubria M210)       H = 3B   dominio 6B
  C3  duas sapatas B = 3,50 m, afastamento entre eixos s = 2B, 4B, 6B; H = 3B; dominio s + 6B
Malha: nB = 12 elementos na largura B (nB par, bordas da sapata nos nos).
Correlacao markoviana rho = exp(-2 sqrt((dx/thx)^2 + (dy/thy)^2)); isotropica se thx = thy.
Campo por medias locais (Gauss 2x2) e Cholesky, como em rfem.py.

Uso:
  python cenarios.py convergencia          # estudo de malha deterministico
  python cenarios.py base [N]              # caso base (COV 0,4; theta 2 m) de todos os cenarios
  python cenarios.py grade CENARIO [N]     # grade COV x theta (CENARIO = C2 ou C3)
  python cenarios.py aniso [N]             # thh = 40 m, thv = 2 m em C2 e C3
  python cenarios.py h4 [N]                # C3_4B, COV 0,4, theta = 10, 14, 30, 50 e 200 m (pico do diferencial)
  python cenarios.py tendencia [N]         # C2 com modulo crescente com a profundidade (caso base)
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
    "C3_2B": dict(B=3.50, Hr=3.0, s=2.0, nB=12),
    "C3_4B": dict(B=3.50, Hr=3.0, s=4.0, nB=12),
    "C3_6B": dict(B=3.50, Hr=3.0, s=6.0, nB=12),
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


# Perfil medio de N_SPT do Sitio 1 de Guerra Colque et al. (2025, Tabela 2), 1 a 17 m
Z_N = np.arange(1, 18)
N_SPT = np.array([6.65, 9.35, 10.2, 14.15, 12.25, 15.25, 17.3, 19.15, 22.05, 23.1, 25.25,
                  26.45, 30.2, 32.95, 35.45, 38.75, 42.95])


def trend_factor(m):
    """f(z) proporcional a N(z)^1,4 (forma da correlacao de Sandroni); a escala e ajustada em simulate()."""
    z = -m.cent[:, 1]
    f = np.interp(z, Z_N, N_SPT) ** 1.4
    return f / f.mean()


def simulate(name, cov, thx, thy, N, rng, trend=False, tag_extra=""):
    m = Footings(**GEOM[name])
    t0 = time.time()
    f = np.ones(m.ne)
    if trend:  # escala da tendencia ajustada para o mesmo recalque deterministico do caso homogeneo
        f = trend_factor(m); f = f * m.g(f)[0] / m.g(np.ones(m.ne))[0]
    g_det = m.g(f)
    L = cholesky(local_avg_corr(m.cent, m.a, m.b, thx, thy) + 1e-10 * np.eye(m.ne), lower=True)
    s2 = np.log(1 + cov ** 2); mu = -s2 / 2
    G = np.empty((N, len(m.feet)))
    for k0 in range(0, N, 200):
        k1 = min(N, k0 + 200)
        Et = f[:, None] * np.exp(mu + np.sqrt(s2) * (L @ rng.standard_normal((m.ne, k1 - k0))))
        for k in range(k1 - k0):
            G[k0 + k] = m.g(Et[:, k])
    tag = f"{name}_cov{cov}_thx{thx}_thy{thy}{tag_extra}"
    np.save(os.path.join(OUT, tag + ".npy"), G)
    g1 = G[:, 0]
    row = dict(cenario=name + tag_extra, B=m.B, H=m.H, W=m.W, malha=[m.nx, m.ny], L=m.L, COV_E=cov,
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
        for name, nBs in [("C1", (6, 12, 24, 48))]:
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
        name = {"C3": "C3_4B"}.get(name, name)
        rng = np.random.default_rng(SEED + (1 if name.startswith("C3") else 0))
        rows = [simulate(name, c, th, th, N, rng) for c in (0.1, 0.2, 0.3, 0.4, 0.5)
                for th in (0.5, 1.0, 2.0, 5.0, 20.0)]
        save(rows, f"grade_{name}.json")
    elif mode == "aniso":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
        rng = np.random.default_rng(SEED + 2)
        rows = [simulate(n, 0.4, 40.0, 2.0, N, rng) for n in ("C2", "C3_4B")]
        save(rows, "aniso.json")
    elif mode == "h4":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
        rng = np.random.default_rng(SEED + 3)
        done = {r["theta_x"] for r in json.load(open(os.path.join(OUT, "h4.json")))} if os.path.exists(os.path.join(OUT, "h4.json")) else set()
        rows = [simulate("C3_4B", 0.4, th, th, N, rng) for th in (10.0, 14.0, 30.0, 50.0, 200.0) if th not in done]
        save(rows, "h4.json")
    elif mode == "tendencia":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
        rng = np.random.default_rng(SEED + 4)
        rows = [simulate("C2", 0.4, 2.0, 2.0, N, rng, trend=True, tag_extra="_tend")]
        save(rows, "tendencia.json")
