# AI for Exoplanet Classification and Similarity to Earth

**Undergraduate Research Project — IFSP Campus Campos do Jordão, Brazil (2025)**

[Switch to Portuguese version](https://github.com/MajorArctures/Exoplanets_Classification/tree/main#uso-de-ia-para-classificacao-de-exoplanetas-e-similaridade-com-a-terra)

A modular machine learning pipeline for the classification of exoplanets using data from the **NASA Exoplanet Archive – Planetary Systems Composite Data**.

The project combines statistical preprocessing, dimensionality reduction, hierarchical clustering, semi-supervised learning, supervised classification, and a similarity-to-Earth analysis.

---

## About the Project

This project was developed as part of an **Undergraduate Research (Iniciação Científica)** project in computational astronomy and machine learning.

The main objective is to develop a reproducible computational pipeline capable of classifying exoplanets into two project-defined categories:

* **TT — Terrestrial Type**
* **NT — Non-Terrestrial Type**

The classification is based on physical and orbital parameters obtained from the NASA Exoplanet Archive.

In addition to classification, the pipeline computes a **similarity index to Earth** for the analyzed exoplanets.

> **Important:** TT and NT are categories defined for this computational study. They should not be interpreted as official classifications provided by NASA or the NASA Exoplanet Archive.

---

## Dataset

The project uses data from the **Planetary Systems Composite Data** available through the NASA Exoplanet Archive.

The main parameters used in the analysis are:

| Parameter     | Description                  |
| ------------- | ---------------------------- |
| `pl_masse`    | Planet mass                  |
| `pl_rade`     | Planet radius                |
| `pl_orbper`   | Orbital period (days)        |
| `pl_orbeccen` | Orbital eccentricity         |
| `pl_orbsmax`  | Semi-major axis of the orbit |

The preprocessing stage removes records that do not satisfy the project's data-quality requirements. No imputation or inference is performed to replace missing physical parameters.

---

## Methodology

The complete pipeline is divided into five main stages.

### 1. Preprocessing

The preprocessing module prepares the dataset for the subsequent machine learning stages.

Operations include:

* Selection of the parameters used in the analysis
* Data-quality filtering
* Treatment of outliers
* Standardization
* Principal Component Analysis (PCA)
* Definition of terrestrial parameter ranges
* Generation of the transformed dataset used by the clustering stage

Several outlier-detection strategies are supported:

* IQR
* Z-score
* MAD
* No outlier filtering

The preprocessing stage also generates the scaler used later by the similarity-to-Earth analysis, ensuring that both analyses operate within the same transformed feature space.

---

### 2. Hierarchical Clustering

Hierarchical clustering is used as the unsupervised stage of the pipeline.

The current implementation uses:

* **Ward linkage**
* **4 clusters**

The resulting clusters are subsequently analyzed to determine their relationship with the terrestrial/non-terrestrial categories used by the project.

---

### 3. Label Propagation

The clustering results are followed by a semi-supervised learning stage using **Label Propagation**.

The current implementation uses:

* k-NN kernel
* `k = 7`

This stage propagates labels through the feature space based on the proximity between observations.

---

### 4. Support Vector Machine

The final classification stage uses a supervised **Support Vector Machine (SVM)**.

The current implementation uses:

* RBF kernel
* Stratified data splitting
* Training, validation, and test subsets

The default split is:

* **75% training**
* **15% validation**
* **10% test**

The resulting model is used to classify the observations into the project-defined TT and NT categories.

---

### 5. Similarity to Earth

The pipeline also calculates a similarity index between Earth and the analyzed exoplanets.

This stage operates in parallel with the final classification and uses the scaler generated during preprocessing to maintain consistency with the transformed feature space.

The resulting file contains the calculated similarity measures for the analyzed exoplanets.

---

## Pipeline Overview

```text
PSCompData.xlsx
       │
       │ 1. preprocessing
       │    filters + scaling + PCA
       │    + terrestrial ranges
       ▼
DFHierarchicalClustering.xlsx
ranges_terrestres.json
       │
       │ 2. hierarchical_clustering
       │    Ward linkage
       │    4 clusters
       ▼
DFLabelPropagation.xlsx
info_proximidade.json
       │
       │ 3. label_propagation
       │    k-NN kernel
       │    k = 7
       ▼
DFsvm.xlsx
       │
       │ 4. SVM
       │    RBF kernel
       │    75/15/10 split
       ▼
resultados_svm.csv

       └───────────────┐
                       │
                       ▼
          similaridade_exoplanetas_terra.csv
              similarity-to-Earth analysis
```

---

## Repository Structure

```text
exoplanetas/
│
├── data/
│   └── PSCompData.xlsx
│
├── pipeline/
│   ├── preprocessing.py
│   ├── hierarchical_clustering.py
│   ├── label_propagation.py
│   ├── svm.py
│   ├── similaridade_terra.py
│   └── diagnostics.py
│
├── notebooks/
│   ├── pipeline_completo.ipynb
│   ├── comparacao_outliers.ipynb
│   └── comparacao_metodos_outliers.ipynb
│
├── docs/
│   ├── source/
│   └── build/html/
│
├── outputs/
│
├── run_pipeline.py
├── requirements.txt
├── NOTAS.md
├── LICENSE
└── README.md
```

### Main Components

| Component          | Purpose                                                    |
| ------------------ | ---------------------------------------------------------- |
| `pipeline/`        | Importable Python modules implementing each pipeline stage |
| `notebooks/`       | Exploratory analyses and visualizations                    |
| `docs/`            | Sphinx documentation                                       |
| `outputs/`         | Generated datasets and analysis results                    |
| `run_pipeline.py`  | Command-line pipeline orchestrator                         |
| `NOTAS.md`         | Methodological notes and observations                      |
| `requirements.txt` | Python dependencies                                        |

---

## Installation

### Requirements

* Python **3.10+**
* pip

Clone the repository and install the dependencies:

```bash
git clone <repository-url>
cd exoplanetas
pip install -r requirements.txt
```

### Main Dependencies

```text
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.10
matplotlib>=3.7
seaborn>=0.12
openpyxl>=3.1
jupyter>=1.0
```

Documentation dependencies are optional:

```text
sphinx>=7.0
sphinx-rtd-theme>=2.0
myst-parser>=2.0
```

---

## Running the Pipeline

### Command Line

Run the default configuration:

```bash
python run_pipeline.py
```

Different outlier-detection methods can be selected from the command line:

```bash
python run_pipeline.py --outlier-method zscore --out outputs_zscore

python run_pipeline.py --outlier-method mad --out outputs_mad

python run_pipeline.py --outlier-method none --out outputs_none
```

Use an alternative proximity metric:

```bash
python run_pipeline.py --metrica euclidiana
```

Customize the IQR threshold:

```bash
python run_pipeline.py --outlier-method iqr --outlier-threshold 2.0
```

Run the pipeline with a custom dataset:

```bash
python run_pipeline.py \
    --data caminho/para/seu.csv \
    --out meus_resultados
```

For all available arguments:

```bash
python run_pipeline.py --help
```

---

## Jupyter Notebooks

The repository contains three notebooks:

| Notebook                            | Description                                               |
| ----------------------------------- | --------------------------------------------------------- |
| `pipeline_completo.ipynb`           | Executes the complete pipeline with inline visualizations |
| `comparacao_outliers.ipynb`         | Compares IQR filtering with no outlier filtering          |
| `comparacao_metodos_outliers.ipynb` | Compares four outlier-detection approaches                |

The notebooks include a setup cell that automatically detects whether the project is running locally or in **Google Colab**.

---

## Google Colab

To run the notebooks in Google Colab:

1. Upload the repository to your Google Drive.
2. Open the desired notebook in Colab.
3. Run the setup cell.
4. The setup automatically mounts the Drive and configures the project imports.

Additional instructions are available in the project documentation:

`docs/build/html/colab.html`

---

## Programmatic Usage

The pipeline modules can also be imported directly into Python:

```python
from pipeline import (
    preprocessing,
    hierarchical_clustering,
    label_propagation,
    svm,
    similaridade_terra,
    diagnostics,
)

resultado = preprocessing.main(
    caminho_entrada="data/PSCompData.xlsx",
    caminho_saida="outputs/DFHierarchicalClustering.xlsx",
    caminho_rgjson="outputs/ranges_terrestres.json",
    outlier_method="iqr",
)

scaler = resultado["scaler"]
pca = resultado["pca_std"]
```

This structure allows individual stages to be executed and analyzed independently.

---

## Documentation

The project documentation is generated with **Sphinx**, using autodoc and Napoleon.

The documentation includes:

* Project introduction
* Installation and usage guides
* Google Colab execution guide
* Methodological analyses
* Outlier comparison
* Pipeline considerations
* API reference for the Python modules

To rebuild the documentation:

```bash
cd docs
make html
```

Then open:

```text
docs/build/html/index.html
```

A pre-built HTML version is included in the repository.

---

## Reproducibility

The project is organized as a modular pipeline so that each stage can be executed, inspected, and reproduced independently.

The repository contains:

* Source code
* Jupyter notebooks
* Configuration and dependency information
* Intermediate datasets
* Generated results
* Methodological documentation

The use of explicit preprocessing and transformation steps also allows subsequent analyses to reuse the same fitted transformations when required.

---

## Scientific Context

This project explores the application of **machine learning to computational astronomy**, combining unsupervised, semi-supervised, and supervised learning techniques in a single analytical workflow.

The methodology was developed during an undergraduate research project at:

**Instituto Federal de São Paulo (IFSP) — Campus Campos do Jordão**

---

## Data Source

This project uses data from the **NASA Exoplanet Archive**, operated by the California Institute of Technology under contract with NASA's Exoplanet Exploration Program.

**NASA Exoplanet Archive — Planetary Systems Composite Data**

https://exoplanetarchive.ipac.caltech.edu/

When using this repository or its results in academic work, please also cite the original dataset and relevant scientific references.

---

## License

See [`LICENSE`](LICENSE) for the terms under which this project is distributed.

---

## Author

**Beatriz Helena Silva**

Undergraduate researcher and student of Technology in Systems Analysis and Development.

This project combines my interests in **data, machine learning, and astronomy**.

Questions, suggestions, and technical feedback are welcome.

---

# Uso de IA para Classificação de Exoplanetas e Similaridade com a Terra

**Projeto de Iniciação Científica — IFSP Campus Campos do Jordão, Brasil (2025)**

[Switch to English version](https://github.com/MajorArctures/Exoplanets_Classification/tree/main)

Um pipeline modular de aprendizado de máquina para classificação de exoplanetas utilizando dados do **NASA Exoplanet Archive – Planetary Systems Composite Data**.

O projeto combina pré-processamento estatístico, redução de dimensionalidade, agrupamento hierárquico, aprendizado semissupervisionado, classificação supervisionada e análise de similaridade com a Terra.

---

## Sobre o Projeto

Este projeto foi desenvolvido como parte de uma **Iniciação Científica** na área de astronomia computacional e aprendizado de máquina.

O principal objetivo é desenvolver um pipeline computacional reprodutível capaz de classificar exoplanetas em duas categorias definidas pelo projeto:

* **TT — Tipo Terrestre**
* **NT — Não Terrestre**

A classificação utiliza parâmetros físicos e orbitais obtidos do NASA Exoplanet Archive.

Além da classificação, o pipeline calcula um **índice de similaridade com a Terra** para os exoplanetas analisados.

> **Importante:** TT e NT são categorias definidas para este estudo computacional. Elas não devem ser interpretadas como classificações oficiais fornecidas pela NASA ou pelo NASA Exoplanet Archive.

---

## Base de Dados

O projeto utiliza dados do **Planetary Systems Composite Data**, disponibilizados pelo NASA Exoplanet Archive.

Os principais parâmetros utilizados na análise são:

| Parâmetro     | Descrição                |
| ------------- | ------------------------ |
| `pl_masse`    | Massa do planeta         |
| `pl_rade`     | Raio do planeta          |
| `pl_orbper`   | Período orbital (dias)   |
| `pl_orbeccen` | Excentricidade orbital   |
| `pl_orbsmax`  | Semieixo maior da órbita |

A etapa de pré-processamento remove registros que não atendem aos critérios de qualidade definidos no projeto. Não são utilizadas técnicas de imputação ou inferência para substituir parâmetros físicos ausentes.

---

## Metodologia

O pipeline completo é dividido em cinco etapas principais.

### 1. Pré-processamento

O módulo de pré-processamento prepara os dados para as etapas seguintes de aprendizado de máquina.

As operações incluem:

* Seleção dos parâmetros utilizados na análise
* Filtragem da qualidade dos dados
* Tratamento de outliers
* Padronização
* Análise de Componentes Principais (PCA)
* Definição dos intervalos de parâmetros terrestres
* Geração do conjunto de dados transformado utilizado no agrupamento

São suportados diferentes métodos de detecção de outliers:

* IQR
* Z-score
* MAD
* Sem filtragem de outliers

A etapa de pré-processamento também gera o scaler utilizado posteriormente na análise de similaridade com a Terra, mantendo a consistência entre os espaços de características utilizados nas análises.

---

### 2. Agrupamento Hierárquico

O agrupamento hierárquico é utilizado como etapa não supervisionada do pipeline.

A implementação atual utiliza:

* **Método de Ward**
* **4 clusters**

Os clusters obtidos são posteriormente analisados em relação às categorias terrestre/não terrestre utilizadas pelo projeto.

---

### 3. Propagação de Rótulos

Após o agrupamento, é utilizada uma etapa de aprendizado semissupervisionado com **Label Propagation**.

A implementação atual utiliza:

* Kernel k-NN
* `k = 7`

Essa etapa propaga os rótulos pelo espaço de características considerando a proximidade entre as observações.

---

### 4. Support Vector Machine

A etapa final de classificação utiliza uma **Support Vector Machine (SVM)** supervisionada.

A implementação atual utiliza:

* Kernel RBF
* Divisão estratificada dos dados
* Conjuntos de treinamento, validação e teste

A divisão padrão é:

* **75% treinamento**
* **15% validação**
* **10% teste**

O modelo resultante é utilizado para classificar as observações nas categorias TT e NT definidas pelo projeto.

---

### 5. Similaridade com a Terra

O pipeline também calcula um índice de similaridade entre a Terra e os exoplanetas analisados.

Essa etapa utiliza o scaler gerado durante o pré-processamento, mantendo a consistência com o espaço de características transformado.

O resultado contém as medidas de similaridade calculadas para os exoplanetas analisados.

---

## Estrutura do Pipeline

```text
PSCompData.xlsx
       │
       │ 1. preprocessing
       │    filtros + scaling + PCA
       │    + intervalos terrestres
       ▼
DFHierarchicalClustering.xlsx
ranges_terrestres.json
       │
       │ 2. hierarchical_clustering
       │    método de Ward
       │    4 clusters
       ▼
DFLabelPropagation.xlsx
info_proximidade.json
       │
       │ 3. label_propagation
       │    kernel k-NN
       │    k = 7
       ▼
DFsvm.xlsx
       │
       │ 4. SVM
       │    kernel RBF
       │    divisão 75/15/10
       ▼
resultados_svm.csv

       └───────────────┐
                       │
                       ▼
          similaridade_exoplanetas_terra.csv
              análise de similaridade com a Terra
```

---

## Estrutura do Repositório

```text
exoplanetas/
│
├── data/
│   └── PSCompData.xlsx
│
├── pipeline/
│   ├── preprocessing.py
│   ├── hierarchical_clustering.py
│   ├── label_propagation.py
│   ├── svm.py
│   ├── similaridade_terra.py
│   └── diagnostics.py
│
├── notebooks/
│   ├── pipeline_completo.ipynb
│   ├── comparacao_outliers.ipynb
│   └── comparacao_metodos_outliers.ipynb
│
├── docs/
│   ├── source/
│   └── build/html/
│
├── outputs/
│
├── run_pipeline.py
├── requirements.txt
├── NOTAS.md
├── LICENSE
└── README.md
```

### Principais Componentes

| Componente         | Função                                               |
| ------------------ | ---------------------------------------------------- |
| `pipeline/`        | Módulos Python responsáveis pelas etapas do pipeline |
| `notebooks/`       | Análises exploratórias e visualizações               |
| `docs/`            | Documentação gerada com Sphinx                       |
| `outputs/`         | Datasets e resultados gerados                        |
| `run_pipeline.py`  | Orquestrador do pipeline via linha de comando        |
| `NOTAS.md`         | Observações e análises metodológicas                 |
| `requirements.txt` | Dependências do projeto                              |

---

## Instalação

### Requisitos

* Python **3.10+**
* pip

Clone o repositório e instale as dependências:

```bash
git clone <repository-url>
cd exoplanetas
pip install -r requirements.txt
```

### Principais Dependências

```text
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.10
matplotlib>=3.7
seaborn>=0.12
openpyxl>=3.1
jupyter>=1.0
```

As dependências para documentação são opcionais:

```text
sphinx>=7.0
sphinx-rtd-theme>=2.0
myst-parser>=2.0
```

---

## Execução do Pipeline

### Linha de Comando

Execute a configuração padrão:

```bash
python run_pipeline.py
```

Diferentes métodos de detecção de outliers podem ser selecionados:

```bash
python run_pipeline.py --outlier-method zscore --out outputs_zscore

python run_pipeline.py --outlier-method mad --out outputs_mad

python run_pipeline.py --outlier-method none --out outputs_none
```

Utilize uma métrica alternativa de proximidade:

```bash
python run_pipeline.py --metrica euclidiana
```

Personalize o threshold do IQR:

```bash
python run_pipeline.py --outlier-method iqr --outlier-threshold 2.0
```

Execute o pipeline com um dataset personalizado:

```bash
python run_pipeline.py \
    --data caminho/para/seu.csv \
    --out meus_resultados
```

Para visualizar todos os argumentos disponíveis:

```bash
python run_pipeline.py --help
```

---

## Notebooks Jupyter

O repositório contém três notebooks:

| Notebook                            | Descrição                                        |
| ----------------------------------- | ------------------------------------------------ |
| `pipeline_completo.ipynb`           | Executa o pipeline completo com visualizações    |
| `comparacao_outliers.ipynb`         | Compara o filtro IQR com a ausência de filtragem |
| `comparacao_metodos_outliers.ipynb` | Compara quatro métodos de detecção de outliers   |

Os notebooks possuem uma célula de configuração que identifica automaticamente se o projeto está sendo executado localmente ou no **Google Colab**.

---

## Google Colab

Para executar os notebooks no Google Colab:

1. Faça o upload do repositório para o Google Drive.
2. Abra o notebook desejado no Colab.
3. Execute a célula de configuração.
4. A configuração monta automaticamente o Drive e ajusta os imports do projeto.

Instruções adicionais estão disponíveis na documentação:

`docs/build/html/colab.html`

---

## Uso Programático

Os módulos do pipeline também podem ser importados diretamente em Python:

```python
from pipeline import (
    preprocessing,
    hierarchical_clustering,
    label_propagation,
    svm,
    similaridade_terra,
    diagnostics,
)

resultado = preprocessing.main(
    caminho_entrada="data/PSCompData.xlsx",
    caminho_saida="outputs/DFHierarchicalClustering.xlsx",
    caminho_rgjson="outputs/ranges_terrestres.json",
    outlier_method="iqr",
)

scaler = resultado["scaler"]
pca = resultado["pca_std"]
```

Essa estrutura permite executar e analisar cada etapa individualmente.

---

## Documentação

A documentação do projeto é gerada com **Sphinx**, utilizando autodoc e Napoleon.

A documentação inclui:

* Introdução ao projeto
* Guias de instalação e uso
* Guia de execução no Google Colab
* Análises metodológicas
* Comparações de métodos de detecção de outliers
* Considerações sobre o pipeline
* Referência da API dos módulos Python

Para gerar novamente a documentação:

```bash
cd docs
make html
```

Depois, abra:

```text
docs/build/html/index.html
```

Uma versão HTML pré-gerada está incluída no repositório.

---

## Reprodutibilidade

O projeto foi organizado como um pipeline modular, permitindo que cada etapa seja executada, analisada e reproduzida de forma independente.

O repositório contém:

* Código-fonte
* Notebooks Jupyter
* Informações de configuração e dependências
* Datasets intermediários
* Resultados gerados
* Documentação metodológica

O uso de etapas explícitas de pré-processamento e transformação também permite reutilizar as mesmas transformações ajustadas em análises posteriores quando necessário.

---

## Contexto Científico

Este projeto explora a aplicação de **aprendizado de máquina à astronomia computacional**, combinando técnicas de aprendizado não supervisionado, semissupervisionado e supervisionado em um único fluxo de análise.

A metodologia foi desenvolvida durante um projeto de Iniciação Científica no:

**Instituto Federal de São Paulo (IFSP) — Campus Campos do Jordão**

---

## Fonte dos Dados

Este projeto utiliza dados do **NASA Exoplanet Archive**, operado pelo California Institute of Technology sob contrato com o programa de exploração de exoplanetas da NASA.

**NASA Exoplanet Archive — Planetary Systems Composite Data**

https://exoplanetarchive.ipac.caltech.edu/

Ao utilizar este repositório ou seus resultados em trabalhos acadêmicos, recomenda-se também citar a base de dados original e as referências científicas relevantes.

---

## Licença

Consulte [`LICENSE`](LICENSE) para os termos de uso e distribuição deste projeto.

---

## Autora

**Beatriz Helena Silva**

Pesquisadora de Iniciação Científica e estudante de Análise e Desenvolvimento de Sistemas.

Este projeto reúne meus interesses em **dados, aprendizado de máquina e astronomia**.

Dúvidas, sugestões e feedback técnico são bem-vindos.
