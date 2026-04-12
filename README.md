# Uso de IA para classificação de exoplanetas e similaridade com a Terra

**Iniciação Científica – IFSP Campus Campos do Jordão (2025)**

---

## Sobre o projeto

Este repositório documenta minha Iniciação Científica, que aplica técnicas de Inteligência Artificial para classificar exoplanetas e identificar aqueles com maior similaridade com a Terra.

> **Status atual:** O projeto está passando por revisões de código e documentação. A análise de similaridade com a Terra e a documentação completa serão disponibilizadas em breve.

---

## O que você encontra aqui

- Scripts em Python (`.py`) e notebooks (`.ipynb`)
- Imagens geradas com Matplotlib
- Dados originais da NASA Exoplanet Archive
- Dados tratados para cada etapa da classificação
- Demais arquivos pertinentes ao projeto

---

## Informações técnicas

| Item | Descrição |
|------|------------|
| **Linguagem** | Python |
| **Ambiente** | Google Colab |
| **Base de dados** | NASA Exoplanet Archive – Planetary System Composite Data (PScompPars) – Julho/2025 |
| **Bibliotecas principais** | NumPy, Pandas, Scikit-Learn, Matplotlib |

---

## Metodologia

### Parâmetros utilizados para classificação

| Parâmetro | Descrição |
|-----------|------------|
| `pl_masse` | Massa do planeta |
| `pl_rade` | Raio do planeta |
| `pl_orbper` | Período orbital (dias) |
| `pl_orbeccen` | Excentricidade da órbita |
| `pl_orbsmax` | Semi-eixo maior da órbita |

### Etapas

1. **Pré-processamento**
   - Padronização com `StandardScaler`
   - Remoção de exoplanetas com parâmetros faltantes
   - Exclusão de valores com incerteza > 50%

2. **Classificação**
   - Hierarchical clustering (não supervisionado)
   - Label Propagation (semi-supervisionado)
   - Support Vector Machine – SVM (supervisionado)

3. **Análise de similaridade com a Terra** *(em revisão)*
   - Comparação percentual entre os parâmetros da Terra e os exoplanetas classificados como terrestres

---

## Observações importantes

- Não foram utilizadas técnicas de inferência para preenchimento de parâmetros faltantes.
- Foram considerados apenas os exoplanetas com todos os cinco parâmetros disponíveis e incerteza inferior a 50%.

---

## Agradecimento (Acknowledgment)

Este projeto utilizou dados do **NASA Exoplanet Archive**, operado pelo California Institute of Technology sob contrato com a National Aeronautics and Space Administration dentro do Exoplanet Exploration Program.

> *This research has made use of the NASA Exoplanet Archive, which is operated by the California Institute of Technology, under contract with the National Aeronautics and Space Administration under the Exoplanet Exploration Program.*

---

## Contribuições e sugestões

Os códigos são **open source** e podem ser usados como base para novos projetos.

Você está convidada(o) a contribuir! Sugestões, críticas ou dúvidas são muito bem-vindas: **devbeatrizhelena@gmail.com**

---

## Nota da autora

> *Sou aluna do curso de Tecnologia em Análise e Desenvolvimento de Sistemas. Se houver considerações sobre essa ou outras áreas envolvidas no projeto, fique à vontade para me contatar. Estou sempre aberta a aprender e melhorar!*
