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
| `python cenarios.py convergencia` | estudo de malha dos cenários → `out/cenarios/convergencia.json` | seção 5.6 | segundos |
| `python cenarios.py base 2000` | caso base de C1–C4 → `out/cenarios/base.json` e `*.npy` | Tabela 5 | 15 min |
| `python cenarios.py grade C2 2000` | grade COV × θ do cenário C2 | Figura 4, Tabela 6 | 35 min |
| `python cenarios.py grade C4 2000` | grade COV × θ do cenário C4 (s = 4B) | Figura 5, Tabela 6 | 65 min |
| `python cenarios.py aniso 2000` | campo anisotrópico (θh = 40 m, θv = 2 m) | seção 5.9 | 5 min |
| `python cenarios.py h4 2000` | C4 com θ = 50 e 200 m (hipótese H4) | seção 5.8 | 5 min |
| `python posproc.py` | recalques dimensionais, probabilidades, índices de Sobol → `out/cenarios/resumo.json`, `fig_*.png` | Tabelas 5–6, Figuras 3–5 | < 1 min |

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

## Cenários (C1–C4)

`cenarios.py` grava o recalque **normalizado** g = δ·μE/(q·B). Como o modelo é linear, uma simulação por
(geometria, COV_E, θ) serve para qualquer módulo médio μE e pressão q, aplicados em `posproc.py`:

    δ = q · B · g / μE

| Cenário | Equipamento | B (m) | H |
|---|---|---|---|
| C1 | britador cônico Kubria G150 (37 t) | 3,50 | 3B |
| C2 | britador cônico Kubria M210 (77 t) | 4,20 | 3B |
| C3 | radier de moinho de bolas (Han e Guevara, 2013) | 21,0 | B |
| C4 | dois G150, afastamento 2B, 4B e 6B | 3,50 | 3B |

Caso base: μE = 40 MPa, COV_E = 0,4, θ = 2 m, ν = 0,3, razão de massas bloco/máquina MR = 3.

## Em andamento

- Verificação cruzada em FEniCSx.

## Licença

MIT — ver `LICENSE`.
