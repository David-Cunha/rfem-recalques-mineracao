"""Verificacoes deterministicas do modelo MEF (rfem.Model).
1) Teste edometrico (carga em toda a largura): solucao exata delta = q H / M.
2) Efeito da largura do dominio.
3) Comparacao com Milovic et al. (1970) via Poulos e Davis (1974, Tabela 7.10, p. 177):
   rho = (P/E) * v0N, sapata rigida rugosa sobre camada finita com base rigida rugosa.
"""
import numpy as np
from rfem import Model
E, P = 4000.0, 100.0
# 1) edometrico
nu, H, W, q = 0.3, 2.0, 6.0, 100.0
m = Model(30, 10, W, H, W, nu, True)
d = m.settlement(np.full(300, E), q * W)
M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
print(f"edometrico: MEF {d*1e3:.6f} mm | exato {q*H/M*1e3:.6f} mm")
# 2) largura do dominio
for W, nx in [(6, 120), (12, 240), (24, 480), (48, 960)]:
    d = Model(nx, 40, W, 2.0, 1.0, 0.3, True).settlement(np.full(nx * 40, E), P)
    print(f"W = {W:>3} m: {d*1e3:.2f} mm")
# 3) Milovic et al. (1970)
v0N = {0.005: {1: 0.7900, 2: 1.1959, 3: 1.5015},
       0.3:   {1: 0.6684, 2: 1.0685, 3: 1.3523},
       0.45:  {1: 0.4170, 2: 0.7618, 3: 0.9940}}
for nu in v0N:
    for h in (1, 2, 3):
        nx, ny = 480, 20 * h
        d = Model(nx, ny, 24.0, float(h), 1.0, nu, True).settlement(np.full(nx * ny, E), P) * 1e3
        r = P / E * v0N[nu][h] * 1e3
        print(f"nu={nu:<5} h/B={h}: MEF {d:6.2f} | Milovic {r:6.2f} | dif {100*(d/r-1):+.1f}%")
