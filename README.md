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
| `python rfem.py 2000 markov2` | convergência de malha e reprodução de Ahmed e Hussain (2020) → `out/resultados.json`, `out/s_*.npy` | Tabelas 1 e 4 | 60 min |
| `python verificacoes.py` | teste analítico do gerador, lognormalidade da reprodução, fatores de Harr → `out/verificacoes.json` | Tabela 3, seção 5.9 | 1 min |
| `python plot.py` | Figura 2 → `out/fig_resultados.png` | Figura 2 | segundos |
| `python fg02.py 5000` | exemplos de Fenton e Griffiths (2002) → `out/fg02_resultados.json` | Tabela 5 | 2 min |
| `python cenarios.py convergencia` | estudo de malha dos cenários | seção 4.2 | segundos |
| `python cenarios.py base 2000` | caso base de C1, C2 e C3 | Tabela 6 | 15 min |
| `python cenarios.py grade C2 2000` | grade COV × θ do cenário C2 | Figura 4, Tabela 8 | 35 min |
| `python cenarios.py grade C3 2000` | grade COV × θ do cenário C3 (s = 4B) | Figura 5, Tabela 8 | 65 min |
| `python cenarios.py aniso 2000` | campo anisotrópico (θh = 40 m, θv = 2 m) | seção 5.8 | 5 min |
| `python cenarios.py aniso_local 2000` | θh = 20 e 40 m, θv = 8 m (escalas locais de Natal: Cunha, 2016; Lucena, 2018) | seção 5.8 | 7 min |
| `python verif_p2.py` | verificação independente com elementos P2 (scikit-fem) contra Milovic et al. | seção 5.1 | 1 min |
| `python cenarios.py tendencia 2000` | C2 com módulo crescente com a profundidade | seção 5.8 | 3 min |
| `python cenarios.py h4 2000` | C3 com θ = 10, 14, 30, 50 e 200 m | seção 5.7 | 12 min |
| `python posproc.py` | probabilidades, módulo mínimo, Sobol, lognormalidade → `out/cenarios/resumo.json`, `fig_*.png` | Tabelas 6–8, Figuras 3–5 | 1 min |

As sementes aleatórias são fixas, de modo que os números se repetem a cada execução.
A pasta `out/` já contém os resultados usados no texto; `out/markov1_N1000/` guarda a primeira
reprodução, com correlação de primeira ordem e 1.000 realizações, citada na seção 5.3.

## Referências do modelo

- AHMED, R. H.; HUSSAIN, M. S. Probabilistic settlement analysis of strip footing on spatially random soil.
  *Journal of University of Duhok*, v. 23, n. 2, p. 617-629, 2020. https://doi.org/10.26682/csjuod.2020.23.2.48
- FENTON, G. A.; GRIFFITHS, D. V. Probabilistic foundation settlement on spatially random soil.
  *Journal of Geotechnical and Geoenvironmental Engineering*, v. 128, n. 5, p. 381-390, 2002.
- FENTON, G. A.; VANMARCKE, E. H. Simulation of random fields via local average subdivision.
  *Journal of Engineering Mechanics*, v. 116, n. 8, p. 1733-1749, 1990.
- POULOS, H. G.; DAVIS, E. H. *Elastic solutions for soil and rock mechanics*. New York: Wiley, 1974.

## Cenários (C1–C3)

`cenarios.py` grava o recalque **normalizado** g = δ·μE/(q·B). Como o modelo é linear, uma simulação por
(geometria, COV_E, θ) serve para qualquer módulo médio μE, pressão q e fator de transformação ε,
aplicados em `posproc.py`:

    δ = q · B · g / (μE · ε)

| Cenário | Equipamento | B (m) | H |
|---|---|---|---|
| C1 | britador cônico Kubria G150 (37 t) | 3,50 | 3B |
| C2 | britador cônico Kubria M210 (77 t) | 4,20 | 3B |
| C3 | dois G150, afastamento 2B, 4B e 6B | 3,50 | 3B |

Caso base: μE = 40 MPa, COV_E = 0,4, θ = 2 m (correlação markoviana de 1ª ordem), ν = 0,3,
razão de massas bloco/máquina MR = 3. Limites de 25 mm, 13 mm, 1/750 e 1/300 são hipóteses de referência.

## Limitações

Modelo bidimensional (estado plano de deformação), elástico linear e estático; os recalques são limites
superiores dos de fundações retangulares (fator de Harr de cerca de 1,7 em C1 e C2).

## Licença

MIT — ver `LICENSE`.
