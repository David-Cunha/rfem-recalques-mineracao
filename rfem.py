"""
Reproducao preliminar (RFEM 2D) - Ahmed & Hussain (2020), J. Univ. Duhok 23(2):617-629.
Sapata corrida rigida, B = 1 m, camada H = 2 m sobre base rigida, estado plano de deformacao.
Malha Q4 120 x 40 (elemento 0,05 m -> largura 6 m, 2,5B de cada lado da sapata).
E lognormal, nu = 0,3 fixo, carga 100 kN/m. Correlacao markoviana de 2a ordem separavel,
como em Ahmed & Hussain (2020): rho = prod_i (1 + 4|tau_i|/theta) exp(-4|tau_i|/theta).
Opcao 'markov1' (rho = exp(-2 tau/theta), isotropica) mantida para comparacao.
Campo gerado por decomposicao de Cholesky da matriz de covariancia das MEDIAS LOCAIS
(aproximacao da LAS por quadratura de Gauss 2x2 por elemento).
Uso: python3 rfem.py [N] [markov2|markov1]  (padrao: N = 2000, markov2; resultados em ./out)
"""
import json, time, os, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import cholesky

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- geometria
def mesh(nx, ny, W=6.0, H=2.0):
    xs = np.linspace(-W / 2, W / 2, nx + 1)
    ys = np.linspace(-H, 0.0, ny + 1)
    X, Y = np.meshgrid(xs, ys)               # (ny+1, nx+1)
    nodes = np.c_[X.ravel(), Y.ravel()]
    nid = lambda i, j: j * (nx + 1) + i
    conn = np.array([[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
                     for j in range(ny) for i in range(nx)])
    cent = nodes[conn].mean(axis=1)
    return nodes, conn, cent, W / nx, H / ny

def ke_unit(a, b, nu):
    """Rigidez Q4 retangular (a x b), E = 1, EPD, Gauss 2x2."""
    D = 1.0 / ((1 + nu) * (1 - 2 * nu)) * np.array(
        [[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]])
    g = 1 / np.sqrt(3)
    K = np.zeros((8, 8))
    for xi in (-g, g):
        for eta in (-g, g):
            dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
            dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
            dNx, dNy = dN_dxi * 2 / a, dN_deta * 2 / b
            B = np.zeros((3, 8))
            B[0, 0::2] = dNx; B[1, 1::2] = dNy
            B[2, 0::2] = dNy; B[2, 1::2] = dNx
            K += B.T @ D @ B * (a * b / 4)
    return K

class Model:
    def __init__(self, nx=120, ny=40, W=6.0, H=2.0, B=1.0, nu=0.3, rough=True):
        self.nodes, self.conn, self.cent, self.a, self.b = mesh(nx, ny, W, H)
        self.ne = len(self.conn); self.ndof = 2 * len(self.nodes)
        K0 = ke_unit(self.a, self.b, nu)
        dofs = np.zeros((self.ne, 8), int)
        dofs[:, 0::2] = 2 * self.conn; dofs[:, 1::2] = 2 * self.conn + 1
        self.rows = np.repeat(dofs, 8, axis=1).ravel()
        self.cols = np.tile(dofs, (1, 8)).ravel()
        self.K0 = K0.ravel()
        x, y = self.nodes[:, 0], self.nodes[:, 1]
        tol = 1e-9
        fixed = {}
        for n in np.where(np.abs(y + H) < tol)[0]:            # base rigida
            fixed[2 * n] = 0.0; fixed[2 * n + 1] = 0.0
        for n in np.where(np.abs(np.abs(x) - W / 2) < tol)[0]:  # laterais: u_x = 0
            fixed[2 * n] = 0.0
        self.foot = np.where((np.abs(y) < tol) & (np.abs(x) <= B / 2 + tol))[0]
        for n in self.foot:                                    # sapata rigida: u_y = -1
            fixed[2 * n + 1] = -1.0
            if rough: fixed[2 * n] = 0.0
        self.fdofs = np.array(sorted(fixed)); self.fvals = np.array([fixed[d] for d in self.fdofs])
        mask = np.ones(self.ndof, bool); mask[self.fdofs] = False
        self.free = np.where(mask)[0]

    def settlement(self, E, P=100.0):
        """E: vetor (kPa) por elemento; P: carga (kN/m). Retorna recalque (m)."""
        data = (E[:, None] * self.K0[None, :]).ravel()
        K = sp.csr_matrix((data, (self.rows, self.cols)), shape=(self.ndof, self.ndof))
        Kff = K[self.free][:, self.free]; Kfc = K[self.free][:, self.fdofs]
        u = np.zeros(self.ndof); u[self.fdofs] = self.fvals
        u[self.free] = spla.spsolve(Kff.tocsc(), -Kfc @ self.fvals)
        R = K @ u
        F = -R[2 * self.foot + 1].sum()   # forca vertical para recalque unitario
        return P / F

# ------------------------------------------------------------ campo aleatorio
def rho_fn(dx, dy, theta, kind="markov1"):
    if kind == "markov1":
        return np.exp(-2 * np.sqrt(dx ** 2 + dy ** 2) / theta)
    tx, ty = 4 * np.abs(dx) / theta, 4 * np.abs(dy) / theta
    return (1 + tx) * (1 + ty) * np.exp(-(tx + ty))

def local_avg_cov(cent, a, b, theta, kind="markov1"):
    """Correlacao entre medias locais (Gauss 2x2 em cada elemento)."""
    g = 1 / np.sqrt(3)
    offs = [(sx * g * a / 2, sy * g * b / 2) for sx in (-1, 1) for sy in (-1, 1)]
    n = len(cent); C = np.zeros((n, n))
    for (ox1, oy1) in offs:
        p1 = cent + [ox1, oy1]
        for (ox2, oy2) in offs:
            p2 = cent + [ox2, oy2]
            C += rho_fn(p1[:, None, 0] - p2[None, :, 0], p1[:, None, 1] - p2[None, :, 1], theta, kind)
    return C / 16.0

def gen_fields(m, theta, cov_E, muE, N, rng, kind="markov1"):
    C = local_avg_cov(m.cent, m.a, m.b, theta, kind)
    C[np.diag_indices_from(C)] += 1e-10
    L = cholesky(C, lower=True)
    s2 = np.log(1 + cov_E ** 2); s = np.sqrt(s2); mu = np.log(muE) - s2 / 2
    G = L @ rng.standard_normal((m.ne, N))
    return np.exp(mu + s * G), float(np.mean(np.diag(C)))   # variance-reduction factor

# --------------------------------------------------------------------- main
if __name__ == "__main__":
    res = {}
    muE, P = 4000.0, 100.0  # kPa, kN/m
    # 1) Verificacao deterministica + convergencia de malha
    conv = []
    # malhas com as bordas da sapata (x = +-0,5 m) coincidentes com nos
    for nx, ny in [(60, 20), (120, 40), (240, 80), (480, 160)]:
        t = time.time(); m = Model(nx, ny)
        s = m.settlement(np.full(m.ne, muE), P)
        conv.append({"nx": nx, "ny": ny, "h_m": 6.0 / nx, "s_mm": s * 1e3, "t_s": time.time() - t})
        print("malha", nx, ny, "recalque = %.3f mm" % (s * 1e3))
    smooth = Model(120, 40, rough=False).settlement(np.full(4800, muE), P)
    res["convergencia"] = conv; res["s_det_liso_mm"] = smooth * 1e3
    m = Model(120, 40)
    s_det = m.settlement(np.full(m.ne, muE), P); res["s_det_mm"] = s_det * 1e3

    # 2) Monte Carlo
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    KIND = sys.argv[2] if len(sys.argv) > 2 else "markov2"
    res["correlacao"] = KIND
    rng = np.random.default_rng(20261106)
    cases = [(c, th) for c in (0.1, 0.3, 0.5) for th in (0.05, 0.5, 1.0, 5.0, 20.0)]
    mc = []
    for cov_E, th in cases:
        t = time.time()
        Ef, vr = gen_fields(m, th, cov_E, muE, N, rng, KIND)
        s = np.array([m.settlement(Ef[:, k], P) for k in range(N)]) * 1e3
        ln = np.log(s)
        row = {"COV_E": cov_E, "theta_m": th, "N": N, "gamma_elem": vr,
               "mean_mm": s.mean(), "std_mm": s.std(ddof=1),
               "cov_s": s.std(ddof=1) / s.mean(),
               "inc_pct": 100 * (s.mean() / (s_det * 1e3) - 1),
               "p05_mm": np.percentile(s, 5), "p50_mm": np.percentile(s, 50), "p95_mm": np.percentile(s, 95),
               "mu_ln": ln.mean(), "sd_ln": ln.std(ddof=1),
               "P_gt_30mm": float((s > 30).mean()), "P_gt_25mm": float((s > 25).mean()),
               "ic95_mean_mm": [s.mean() - 1.96 * s.std(ddof=1) / np.sqrt(N), s.mean() + 1.96 * s.std(ddof=1) / np.sqrt(N)],
               "t_s": time.time() - t}
        mc.append(row)
        np.save(os.path.join(OUT, f"s_cov{cov_E}_th{th}.npy"), s)
        print(f"COV={cov_E} theta={th}: media={row['mean_mm']:.2f} dp={row['std_mm']:.2f} "
              f"inc={row['inc_pct']:.1f}% P(>30)={row['P_gt_30mm']:.3f} t={row['t_s']:.0f}s", flush=True)
    res["monte_carlo"] = mc
    json.dump(res, open(os.path.join(OUT, "resultados.json"), "w"), indent=2)
