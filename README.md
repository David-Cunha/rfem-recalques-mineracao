# RFEM — recalques de fundações rasas em instalações de mineração

Código do Trabalho de Conclusão de Curso de Engenharia Civil (UNI7, Fortaleza):
**análise probabilística de recalques de fundações rasas em instalações industriais de mineração**
pelo Método dos Elementos Finitos Aleatórios (RFEM) com simulação de Monte Carlo.

Autor: David Cunha do Nascimento

## Modelo

- Solo elástico linear em estado plano de deformação; elementos Q4 com integração de Gauss 2 × 2.
- Sapata corrida rígida e rugosa (deslocamento vertical prescrito); camada finita sobre base rígida.
- Módulo de elasticidade como campo aleatório lognormal, correlação markoviana
  ρ(τ) = exp(−2τ/θ), médias locais por elemento geradas por decomposição de Cholesky
  (aproximação da LAS de Fenton e Vanmarcke, 1990).
- Coeficiente de Poisson determinístico.

## Instalação

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate    Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Testado com Python 3.13, NumPy 2.5, SciPy 1.18 e Matplotlib 3.11.

## Como reproduzir os resultados da monografia

| Comando | Gera | Na monografia | Tempo aprox. |
|---|---|---|---|
| `python verif_pd.py` | teste edométrico, largura do domínio, comparação com Milovic et al. (1970) | seção 5.1, Tabela 2 | 1 min |
| `python rfem.py 1000` | convergência de malha e Monte Carlo da reprodução de Ahmed e Hussain (2020) → `out/resultados.json`, `out/s_*.npy` | Tabelas 1 e 4 | 25 min |
| `python fg02.py 5000` | exemplos de Fenton e Griffiths (2002) → `out/fg02_resultados.json` | Tabela 3 | 2 min |
| `python plot.py` | gráfico a partir de `out/resultados.json` → `out/fig_resultados.png` | Figura 2 | segundos |

As sementes aleatórias são fixas (`20261106`), de modo que os números se repetem a cada execução.
A pasta `out/` já contém os resultados usados no texto.

## Referências do modelo

- AHMED, R. H.; HUSSAIN, M. S. Probabilistic settlement analysis of strip footing on spatially random soil.
  *Journal of University of Duhok*, v. 23, n. 2, p. 617-629, 2020. https://doi.org/10.26682/csjuod.2020.23.2.48
- FENTON, G. A.; GRIFFITHS, D. V. Probabilistic foundation settlement on spatially random soil.
  *Journal of Geotechnical and Geoenvironmental Engineering*, v. 128, n. 5, p. 381-390, 2002.
- FENTON, G. A.; VANMARCKE, E. H. Simulation of random fields via local average subdivision.
  *Journal of Engineering Mechanics*, v. 116, n. 8, p. 1733-1749, 1990.
- POULOS, H. G.; DAVIS, E. H. *Elastic solutions for soil and rock mechanics*. New York: Wiley, 1974.

## Em andamento

- `cenarios.py`: cenários de fundação de equipamentos C1–C4 (britadores cônicos, moinho de bolas, par de sapatas).
- Anisotropia (θh ≠ θv), análise de sensibilidade (índices de Sobol) e verificação cruzada em FEniCSx.

## Licença

MIT — ver `LICENSE`.
