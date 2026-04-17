## AI for Exoplanet Classification and Similarity to Earth

**Undergraduate Research – IFSP Campus Campos do Jordao (2025)**

[Switch to Portuguese version](#uso-de-ia-para-classificacao-de-exoplanetas-e-similaridade-com-a-terra)

---

## About the project

This repository documents my undergraduate research project, which applies Artificial Intelligence techniques to classify exoplanets and identify those most similar to Earth.

> **Current status:** The project is undergoing code and documentation revisions. The similarity-to-Earth analysis and complete documentation will be available soon.

### What you'll find here

- Python scripts (.py) and notebooks (.ipynb)
- Images generated with Matplotlib
- Raw data from NASA Exoplanet Archive
- Processed data for each classification stage
- Other relevant project files

### Technical information

| Item | Description |
|------|-------------|
| **Language** | Python |
| **Environment** | Google Colab |
| **Dataset** | NASA Exoplanet Archive – Planetary System Composite Data (PScompPars) – July 2025 |
| **Main libraries** | NumPy, Pandas, Scikit-Learn, Matplotlib |

### Methodology

### Parameters used for classification

| Parameter | Description |
|-----------|-------------|
| `pl_masse` | Planet mass |
| `pl_rade` | Planet radius |
| `pl_orbper` | Orbital period (days) |
| `pl_orbeccen` | Orbital eccentricity |
| `pl_orbsmax` | Semi-major axis of orbit |

### Stages

1. **Pre-processing**
   - Standardization with StandardScaler
   - Removal of exoplanets with missing parameters
   - Exclusion of values with uncertainty > 50%

2. **Classification**
   - Hierarchical clustering (unsupervised)
   - Label Propagation (semi-supervised)
   - Support Vector Machine – SVM (supervised)

3. **Similarity-to-Earth analysis** *(under revision)*
   - Percentage comparison between Earth's parameters and exoplanets classified as terrestrial

### Important notes

- No inference techniques were used to fill missing parameters.
- Only exoplanets with all five parameters available and uncertainty below 50% were considered.

### Acknowledgment

This research has made use of the NASA Exoplanet Archive, which is operated by the California Institute of Technology, under contract with the National Aeronautics and Space Administration under the Exoplanet Exploration Program.

### Contributions and suggestions

The code is **open source** and can be used as a foundation for new projects.

You are invited to contribute! Suggestions, feedback, and questions are very welcome:

**devbeatrizhelena@gmail.com**

- **Prof. Dr. Josivan Pereira da Silva (https://github.com/josivanSilvaCodes)** – Project Tutor

#### Author's note

*I am a student in the Technology in Systems Analysis and Development program. If you have any considerations regarding this or other related areas, please feel free to contact me. I am always open to learning and improving!*



---
# Uso de IA para classificacao de exoplanetas e similaridade com a Terra

**Iniciacao Cientifica – IFSP Campus Campos do Jordao (2025)**

[Switch to English version](#english-version)

---

## Sobre o projeto

Este repositorio documenta minha Iniciacao Cientifica, que aplica tecnicas de Inteligencia Artificial para classificar exoplanetas e identificar aqueles com maior similaridade com a Terra.

> **Status atual:** O projeto esta passando por revisoes de codigo e documentacao. A analise de similaridade com a Terra e a documentacao completa serao disponibilizadas em breve.


### O que voce encontra aqui

- Scripts em Python (.py) e notebooks (.ipynb)
- Imagens geradas com Matplotlib
- Dados originais da NASA Exoplanet Archive
- Dados tratados para cada etapa da classificacao
- Demais arquivos pertinentes ao projeto

### Informacoes tecnicas

| Item | Descricao |
|------|------------|
| **Linguagem** | Python |
| **Ambiente** | Google Colab |
| **Base de dados** | NASA Exoplanet Archive – Planetary System Composite Data (PScompPars) – Julho/2025 |
| **Bibliotecas principais** | NumPy, Pandas, Scikit-Learn, Matplotlib |


### Metodologia

### Parametros utilizados para classificacao

| Parametro | Descricao |
|-----------|------------|
| `pl_masse` | Massa do planeta |
| `pl_rade` | Raio do planeta |
| `pl_orbper` | Periodo orbital (dias) |
| `pl_orbeccen` | Excentricidade da orbita |
| `pl_orbsmax` | Semi-eixo maior da orbita |

### Etapas

1. **Pre-processamento**
   - Padronizacao com StandardScaler
   - Remocao de exoplanetas com parametros faltantes
   - Exclusao de valores com incerteza > 50%

2. **Classificacao**
   - Hierarchical clustering (nao supervisionado)
   - Label Propagation (semi-supervisionado)
   - Support Vector Machine – SVM (supervisionado)

3. **Analise de similaridade com a Terra** *(em revisao)*
   - Comparacao percentual entre os parametros da Terra e os exoplanetas classificados como terrestres


### Observacoes importantes

- Nao foram utilizadas tecnicas de inferencia para preenchimento de parametros faltantes.
- Foram considerados apenas os exoplanetas com todos os cinco parametros disponiveis e incerteza inferior a 50%.


### Agradecimento (Acknowledgment)

Este projeto utilizou dados do **NASA Exoplanet Archive**, operado pelo California Institute of Technology sob contrato com a National Aeronautics and Space Administration dentro do Exoplanet Exploration Program.

*This research has made use of the NASA Exoplanet Archive, which is operated by the California Institute of Technology, under contract with the National Aeronautics and Space Administration under the Exoplanet Exploration Program.*

### Contribuicoes e sugestoes

Os codigos sao **open source** e podem ser usados como base para novos projetos.

Voce esta convidada(o) a contribuir! Sugestoes, criticas ou duvidas sao muito bem-vindas:

**devbeatrizhelena@gmail.com**

#### Nota da autora

*Sou aluna do curso de Tecnologia em Analise e Desenvolvimento de Sistemas. Se houver consideracoes sobre essa ou outras areas envolvidas no projeto, fique a vontade para me contatar. Estou sempre aberta a aprender e melhorar!*

---


