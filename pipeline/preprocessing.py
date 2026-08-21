# -*- coding: utf-8 -*-
"""
Módulo de Pré-processamento
============================

Pipeline de tratamento dos dados de exoplanetas da base
*Planetary Systems Composite Data* (NASA Exoplanet Archive):

1. Filtragem de colunas relevantes
2. Filtragem por limites (pl_*lim ∈ {-1, 0, 1})
3. Remoção de linhas com incerteza relativa > 50%
4. Identificação e remoção de outliers
5. Padronização (StandardScaler) e PCA (2 componentes)
6. Cálculo dos ranges terrestres no espaço padronizado
7. Análise das cargas fatoriais e visualização
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


COLUNAS_NUMERICAS = ["pl_orbper", "pl_orbsmax", "pl_rade", "pl_bmasse", "pl_orbeccen"]

COL_NAMES = {
    "pl_orbper": "Período Orbital do Planeta",
    "pl_orbsmax": "Semi-eixo Maior da Órbita",
    "pl_rade": "Raio do Planeta",
    "pl_bmasse": "Massa do Planeta",
    "pl_orbeccen": "Excentricidade Orbital",
}


def _save_and_show(fig_name, save_dir, show):
    """Helper interno: salva figura (se save_dir) e exibe (se show)."""
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_dir) / fig_name, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def carregar_arquivo(caminho):
    """
    0 - Carregamento da base de dados.

    Carrega a base *Planetary Systems Composite Data* em formato XLSX ou CSV.
    O CSV do NASA Exoplanet Archive vem com header comentado por ``#``,
    tratado automaticamente.

    Parâmetros
    ----------
    caminho : str
        Caminho do arquivo Excel (.xlsx) ou CSV (.csv).

    Retorna
    -------
    pandas.DataFrame
        DataFrame carregado.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não for encontrado.
    """
    caminho = str(caminho)
    try:
        if caminho.endswith(".csv"):
            return pd.read_csv(caminho, comment="#", low_memory=False)
        else:
            return pd.read_excel(caminho)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}") from e


def limpeza_dados(df):
    """
    1.1 - Remoção das Colunas que não serão utilizadas.

    Nesta etapa, todas as linhas da planilha são mantidas.

    Dicionários de variáveis
    ------------------------
    - pl_name = Nome do Planeta
    - pl_orbper = Período Orbital (Dias)
    - pl_orbsmax = Semi-eixo Maior da Órbita
    - pl_rade = Raio do Planeta
    - pl_bmasse = Massa do Planeta
    - pl_orbeccen = Excentricidade Orbital

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame a ser processado.

    Retorna
    -------
    pandas.DataFrame
        DataFrame processado.
    """
    col_checked = [
        "pl_name",
        "pl_orbper", "pl_orbpererr1", "pl_orbpererr2", "pl_orbperlim",
        "pl_orbsmax", "pl_orbsmaxerr1", "pl_orbsmaxerr2", "pl_orbsmaxlim",
        "pl_rade", "pl_radeerr1", "pl_radeerr2", "pl_radelim",
        "pl_bmasse", "pl_bmasseerr1", "pl_bmasseerr2", "pl_bmasselim",
        "pl_orbeccen", "pl_orbeccenerr1", "pl_orbeccenerr2", "pl_orbeccenlim",
    ]
    return df[col_checked]


def def_lines_limits(df_columns_filter):
    """
    1.2 - Remoção das Linhas de limites diferentes de 0, -1 e 1.

    Nesta etapa, deixam-se apenas as linhas que contêm valores confirmados
    ou estimados com precisão e/ou aqueles estimados com grau de incerteza
    inferior a 50%.

    Descrição de variáveis
    ----------------------
    - pl_orbperlim = Limite da distância máxima da estrela host
    - pl_orbsmaxlim = Limite do semi-eixo maior da órbita
    - pl_radelim = Limite do raio do planeta
    - pl_bmasselim = Limite da massa do planeta
    - pl_orbeccenlim = Limite da excentricidade orbital

    Parâmetros
    ----------
    df_columns_filter : pandas.DataFrame
        DataFrame a ser processado.

    Retorna
    -------
    df_lim : pandas.DataFrame
        DataFrame processado.
    """
    return df_columns_filter[
        (df_columns_filter["pl_orbperlim"].isin([0, 1, -1])) &
        (df_columns_filter["pl_orbsmaxlim"].isin([0, 1, -1])) &
        (df_columns_filter["pl_radelim"].isin([0, 1, -1])) &
        (df_columns_filter["pl_bmasselim"].isin([0, 1, -1])) &
        (df_columns_filter["pl_orbeccenlim"].isin([0, 1, -1]))
    ]


def remocao_incertezas(df_lines_limits):
    """
    1.3 - Remoção das Linhas de alta incerteza.

    Eliminam-se as linhas que contêm grau de incerteza superior a 50%.

    Critérios
    ---------
    - Valores == 0 são considerados precisos, não sendo necessário verificar
      incerteza.
    - Valores != 0 devem apresentar incerteza relativa ≤ 50%.

    Parâmetros
    ----------
    df_lim : pandas.DataFrame
        DataFrame a ser processado.

    Retorna
    -------
    pandas.DataFrame
        DataFrame processado.
    """
    df = df_lines_limits
    return df[
        (
            (df["pl_orbperlim"] == 0) |
            (
                (df["pl_orbper"] != 0) &
                (df["pl_orbpererr1"].abs() / df["pl_orbper"].abs() <= 0.5) &
                (df["pl_orbpererr2"].abs() / df["pl_orbper"].abs() <= 0.5)
            )
        )
        &
        (
            (df["pl_orbsmaxlim"] == 0) |
            (
                (df["pl_orbsmax"] != 0) &
                (df["pl_orbsmaxerr1"].abs() / df["pl_orbsmax"].abs() <= 0.5) &
                (df["pl_orbsmaxerr2"].abs() / df["pl_orbsmax"].abs() <= 0.5)
            )
        )
        &
        (
            (df["pl_bmasselim"] == 0) |
            (
                (df["pl_bmasse"] != 0) &
                (df["pl_bmasseerr1"].abs() / df["pl_bmasse"].abs() <= 0.5) &
                (df["pl_bmasseerr2"].abs() / df["pl_bmasse"].abs() <= 0.5)
            )
        )
        &
        (
            (df["pl_radelim"] == 0) |
            (
                (df["pl_rade"] != 0) &
                (df["pl_radeerr1"].abs() / df["pl_rade"].abs() <= 0.5) &
                (df["pl_radeerr2"].abs() / df["pl_rade"].abs() <= 0.5)
            )
        )
        &
        (
            (df["pl_orbeccenlim"] == 0) |
            (
                (df["pl_orbeccen"] != 0) &
                (df["pl_orbeccenerr1"].abs() / df["pl_orbeccen"].abs() <= 0.5) &
                (df["pl_orbeccenerr2"].abs() / df["pl_orbeccen"].abs() <= 0.5)
            )
        )
    ]


def id_outliers(df_inc_rm, save_dir=None, show=True):
    """
    1.4 - Identificação dos Outliers.

    Gera um boxplot para cada uma das 5 variáveis físicas, permitindo
    inspeção visual da distribuição e localização dos outliers.

    Método
    ------
    - Define o tamanho da figura (polegadas)
    - Remove valores NaN para evitar erros
    - Cria um boxplot para cada coluna numérica
    - Define um grid para visualização (0.6 = 60% opaco)

    Critérios
    ---------
    Cada boxplot sumariza a distribuição dos dados através de:
    [Q1 - 1.5×IQR, Q3 + 1.5×IQR], onde IQR = Q3 - Q1.

    - Linha central: mediana (Q2)
    - Caixa: intervalo interquartil (IQR = Q3 - Q1)
    - Whiskers: estendem-se até 1.5×IQR após os quartis Q1 e Q3
    - Pontos além dos whiskers: outliers estatísticos (valores atípicos)

    Parâmetros
    ----------
    df_inc_rm : pandas.DataFrame
        DataFrame a ser processado.
    save_dir : str, opcional
        Se fornecido, salva cada boxplot como PNG neste diretório.
    show : bool, opcional
        Se ``True`` (padrão), exibe os boxplots interativamente.

    Retorna
    -------
    None
        A função apenas exibe os boxplots.
    """
    for col in COLUNAS_NUMERICAS:
        if col not in df_inc_rm.columns:
            raise KeyError(f"Coluna '{col}' não encontrada no DataFrame.")

    for col in COLUNAS_NUMERICAS:
        plt.figure(figsize=(6, 4))
        plt.boxplot(df_inc_rm[col].dropna())
        plt.title(f"Boxplot - {COL_NAMES[col]}", fontsize=12, fontweight="bold")
        plt.ylabel("Valores")
        plt.xlabel(COL_NAMES[col])
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        _save_and_show(f"boxplot_{col}.png", save_dir, show)


def rm_outliers(df_inc_rm, method="iqr", threshold=None, colunas=None):
    """
    1.5 - Identificação das linhas que contém os outliers.

    Como parte implícita dessa função todas as linhas que contenham valores
    NaN (Not a Number) em qualquer uma das variáveis analisadas são
    automaticamente removidas do dataset.

    Métodos disponíveis
    -------------------
    - **iqr** (padrão, Tukey 1977): remove se ``x < Q1 - k·IQR`` ou
      ``x > Q3 + k·IQR``. Threshold padrão ``k=1.5``. Método utilizado no
      pipeline original.
    - **zscore**: remove se ``|x - μ| / σ > k``. Threshold padrão ``k=3``.
    - **mad** (Iglewicz-Hoaglin 1993): modified z-score
      ``|0.6745·(x - mediana) / MAD| > k``, onde ``MAD = mediana(|x - mediana|)``.
      Threshold padrão ``k=3.5``.
    - **none**: apenas dropna, sem remoção estatística.

    Método (aplicado por variável)
    ------------------------------
    - Inicializa a condição como True (todos os pontos mantidos por padrão)
    - Para cada coluna, calcula os limites conforme o método escolhido
    - Atualiza a condição AND para manter apenas pontos DENTRO dos limites
    - Filtra o DataFrame original (mantém apenas linhas sem outliers em
      NENHUMA das variáveis)

    Parâmetros
    ----------
    df_inc_rm : pandas.DataFrame
        DataFrame a ser processado.
    method : {'iqr', 'zscore', 'mad', 'none'}, opcional
        Método de detecção. Padrão ``'iqr'``.
    threshold : float, opcional
        Multiplicador/limite. Se ``None``, usa o padrão de cada método.
    colunas : list of str, opcional
        Colunas onde aplicar. Se ``None``, usa :data:`COLUNAS_NUMERICAS`.

    Retorna
    -------
    pandas.DataFrame
        DataFrame filtrado, contendo apenas linhas sem outliers em nenhuma
        das variáveis analisadas.
    """
    colunas = colunas or COLUNAS_NUMERICAS
    df_clean = df_inc_rm.dropna(subset=colunas)

    if method == "none":
        return df_clean

    cond = pd.Series(True, index=df_clean.index)

    if method == "iqr":
        k = threshold if threshold is not None else 1.5
        for coluna in colunas:
            Q1 = df_clean[coluna].quantile(0.25)
            Q3 = df_clean[coluna].quantile(0.75)
            IQR = Q3 - Q1
            cond &= ((df_clean[coluna] >= Q1 - k * IQR) &
                     (df_clean[coluna] <= Q3 + k * IQR))

    elif method == "zscore":
        k = threshold if threshold is not None else 3.0
        for coluna in colunas:
            mean = df_clean[coluna].mean()
            std = df_clean[coluna].std()
            if std == 0:
                continue
            z = (df_clean[coluna] - mean) / std
            cond &= z.abs() <= k

    elif method == "mad":
        k = threshold if threshold is not None else 3.5
        for coluna in colunas:
            median = df_clean[coluna].median()
            mad = (df_clean[coluna] - median).abs().median()
            if mad == 0:
                mad = df_clean[coluna].std() / 1.4826
                if mad == 0:
                    continue
            modified_z = 0.6745 * (df_clean[coluna] - median) / mad
            cond &= modified_z.abs() <= k

    else:
        raise ValueError(
            f"method='{method}' inválido. Use 'iqr', 'zscore', 'mad' ou 'none'."
        )

    return df_clean[cond]


def pca_standard_scaler(df_inc_rm, n_components=2):
    """
    2 - NORMALIZAÇÃO/PADRONIZAÇÃO DE DADOS.

    Esta etapa realiza a seleção das variáveis numéricas, a padronização
    dos dados utilizando o método StandardScaler e a aplicação da técnica
    de redução de dimensionalidade PCA, com foco nos dois primeiros
    componentes principais.

    Método
    ------
    - Seleção das colunas numéricas
    - Aplicação do StandardScaler
    - Aplicação do PCA
    - Análise da Variância Explicada

    Parâmetros
    ----------
    df_inc_rm : pandas.DataFrame
        DataFrame a ser processado.
    n_components : int, opcional
        Número de componentes principais. Padrão 2.

    Retorna
    -------
    df_to_hc : pandas.DataFrame
        DataFrame com os dados padronizados e reduzidos (PC1, PC2 + 5 vars std).
    pca_std : sklearn.decomposition.PCA
        Objeto PCA ajustado sobre os dados padronizados.
    scaler : sklearn.preprocessing.StandardScaler
        Scaler ajustado (necessário para transformar novos dados).
    colunas_numericas : list of str
        Lista com os nomes das variáveis numéricas utilizadas no PCA.
    variancia_pc1 : float
        Variância explicada pelo PC1.
    variancia_pc2 : float
        Variância explicada pelo PC2.
    """
    df_numerico = df_inc_rm[COLUNAS_NUMERICAS].copy()

    scaler = StandardScaler()
    df_normalizado = scaler.fit_transform(df_numerico)

    pca_std = PCA(n_components=n_components)
    df_pca = pca_std.fit_transform(df_normalizado)

    df_to_hc = pd.DataFrame(df_pca, columns=["PC1", "PC2"], index=df_inc_rm.index)
    df_std = pd.DataFrame(df_normalizado, columns=COLUNAS_NUMERICAS, index=df_inc_rm.index)
    df_to_hc = pd.concat([df_to_hc, df_std], axis=1)

    variancia_pc1, variancia_pc2 = pca_std.explained_variance_ratio_
    variancia_total = variancia_pc1 + variancia_pc2

    print("Análise PCA:")
    print(f"  Variância explicada total: {variancia_total:.1%}")
    print(f"  PC1: {variancia_pc1:.1%} | PC2: {variancia_pc2:.1%}")

    return df_to_hc, pca_std, scaler, COLUNAS_NUMERICAS, variancia_pc1, variancia_pc2


def ranges_terrestres(scaler, colunas_numericas, caminho_rgjson):
    """
    3.4 - Referencial Terrestre no Espaço Padronizado.

    Define os ranges físicos característicos de planetas terrestres e os
    converte para o espaço padronizado (z-score). Persiste em JSON para uso
    pelas etapas seguintes.

    Ranges físicos utilizados
    -------------------------
    - pl_orbper: 88 – 687 dias (Mercúrio a Marte)
    - pl_orbsmax: 0,39 – 1,52 UA
    - pl_rade: 0,5 – 1,6 raios terrestres
    - pl_bmasse: 1 – 15 massas terrestres
    - pl_orbeccen: 0 – 0,25

    Parâmetros
    ----------
    scaler : sklearn.preprocessing.StandardScaler
        Scaler ajustado aos dados físicos originais.
    colunas_numericas : list of str
        Lista com os nomes das variáveis numéricas (mesma ordem do scaler).
    caminho_rgjson : str
        Caminho do arquivo JSON de saída.

    Retorna
    -------
    ranges_padronizados : dict
        Ranges terrestres no espaço padronizado.

    Raises
    ------
    ValueError
        Se o scaler não estiver ajustado.
    """
    if not hasattr(scaler, "mean_"):
        raise ValueError("Scaler precisa estar ajustado antes de usar esta função.")

    ranges_fisicos = {
        "pl_orbper": (88, 687),
        "pl_orbsmax": (0.39, 1.52),
        "pl_rade": (0.5, 1.6),
        "pl_bmasse": (1, 15),
        "pl_orbeccen": (0, 0.25),
    }

    ranges_padronizados = {}
    for var, (low, high) in ranges_fisicos.items():
        idx = colunas_numericas.index(var)
        mean = scaler.mean_[idx]
        std = scaler.scale_[idx]
        ranges_padronizados[var] = ((low - mean) / std, (high - mean) / std)

    Path(caminho_rgjson).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_rgjson, "w") as f:
        json.dump(ranges_padronizados, f, indent=4)

    print("\nRanges terrestres no espaço padronizado:")
    for var, (lo, hi) in ranges_padronizados.items():
        print(f"  {var}: {lo:.5f} – {hi:.5f}")

    return ranges_padronizados


def analise_cargas(pca_std, colunas_numericas):
    """
    2.7 - Análise das cargas (loadings) e contribuição das variáveis no PCA.

    Esta etapa calcula as cargas fatoriais (loadings) das variáveis originais
    em cada componente principal e estima a contribuição percentual de cada
    variável nos dois primeiros componentes (PC1 e PC2). Além disso, é
    calculada a contribuição acumulada das três variáveis mais relevantes em
    cada componente.

    Parâmetros
    ----------
    pca_std : sklearn.decomposition.PCA
        Objeto PCA já ajustado sobre os dados padronizados.
    colunas_numericas : list of str
        Lista com os nomes das variáveis numéricas utilizadas no PCA.

    Retorna
    -------
    df_loadings : pandas.DataFrame
        Cargas fatoriais das variáveis para PC1 e PC2.
    top3_pc1 : pandas.Series
        Contribuição das 3 variáveis mais relevantes em PC1.
    top3_pc2 : pandas.Series
        Contribuição das 3 variáveis mais relevantes em PC2.
    soma_top3_pc1 : float
        Soma da contribuição das top-3 em PC1.
    soma_top3_pc2 : float
        Soma da contribuição das top-3 em PC2.
    """
    loadings = pca_std.components_.T * np.sqrt(pca_std.explained_variance_)
    df_loadings = pd.DataFrame(loadings, index=colunas_numericas, columns=["PC1", "PC2"])

    contrib_pc1 = (df_loadings["PC1"].abs() / df_loadings["PC1"].abs().sum() * 100).round(1)
    contrib_pc2 = (df_loadings["PC2"].abs() / df_loadings["PC2"].abs().sum() * 100).round(1)

    top3_pc1 = contrib_pc1.sort_values(ascending=False).head(3)
    top3_pc2 = contrib_pc2.sort_values(ascending=False).head(3)

    return df_loadings, top3_pc1, top3_pc2, top3_pc1.sum(), top3_pc2.sum()


def pca_plot_2D(df_to_hc, variancia_pc1, variancia_pc2,
                top3_pc1, top3_pc2, soma_top3_pc1, soma_top3_pc2,
                save_dir=None, show=True):
    """
    2.9 - Visualização do PCA 2D com contribuição acumulada.

    Esta função cria um gráfico de dispersão 2D com os dados tratados,
    incluindo a variância explicada por PC1/PC2 e as top-3 variáveis
    contribuidoras nos labels dos eixos.

    Parâmetros
    ----------
    df_to_hc : pandas.DataFrame
        DataFrame contendo os dados (colunas PC1, PC2).
    variancia_pc1 : float
        Variância explicada pelo PC1.
    variancia_pc2 : float
        Variância explicada pelo PC2.
    top3_pc1 : pandas.Series
        Contribuição percentual das três variáveis mais relevantes no PC1.
    top3_pc2 : pandas.Series
        Contribuição percentual das três variáveis mais relevantes no PC2.
    soma_top3_pc1 : float
        Soma da contribuição percentual das três variáveis mais relevantes no PC1.
    soma_top3_pc2 : float
        Soma da contribuição percentual das três variáveis mais relevantes no PC2.
    save_dir : str, opcional
        Se fornecido, salva o gráfico como ``pca_2d.png``.
    show : bool, opcional
        Se ``True``, exibe interativamente.

    Retorna
    -------
    None
        A função apenas exibe/salva o gráfico.
    """
    plt.figure(figsize=(9, 7))
    plt.scatter(df_to_hc["PC1"], df_to_hc["PC2"], alpha=0.6, s=25, color="hotpink")
    plt.title("Análise de Componentes Principais (PCA) - Exoplanetas",
              fontsize=12, fontweight="bold", pad=15)
    plt.xlabel(f"PC1 ({variancia_pc1:.1%} var.)\n"
               f"{', '.join(top3_pc1.index)} = {soma_top3_pc1}%", fontsize=10)
    plt.ylabel(f"PC2 ({variancia_pc2:.1%} var.)\n"
               f"{', '.join(top3_pc2.index)} = {soma_top3_pc2}%", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    _save_and_show("pca_2d.png", save_dir, show)


def main(caminho_entrada, caminho_saida, caminho_rgjson,
         outlier_method="iqr", outlier_threshold=None,
         remover_outliers=None,
         save_dir=None, show=False):
    """
    Executa toda a etapa 1 (pré-processamento) do pipeline.

    Encadeia as sub-etapas:

    1. :func:`carregar_arquivo`
    2. :func:`limpeza_dados` → :func:`def_lines_limits` → :func:`remocao_incertezas`
    3. :func:`id_outliers` → :func:`rm_outliers`
    4. :func:`pca_standard_scaler`
    5. :func:`ranges_terrestres` e :func:`analise_cargas`
    6. :func:`pca_plot_2D`

    E persiste o DataFrame resultante para a etapa seguinte.

    Parâmetros
    ----------
    caminho_entrada : str
        Caminho do arquivo de entrada (.xlsx ou .csv).
    caminho_saida : str
        Caminho do Excel de saída (entrada da etapa 2).
    caminho_rgjson : str
        Caminho do JSON de ranges terrestres padronizados.
    outlier_method : {'iqr', 'zscore', 'mad', 'none'}, opcional
        Método de detecção de outliers. Padrão ``'iqr'``.
    outlier_threshold : float, opcional
        Threshold do método. Se ``None``, usa o padrão de cada método.
    remover_outliers : bool, opcional
        Compatibilidade com API antiga. ``True`` = ``'iqr'``, ``False`` = ``'none'``.
    save_dir : str, opcional
        Diretório para salvar figuras (PNG).
    show : bool, opcional
        Se ``True``, exibe figuras interativamente.

    Retorna
    -------
    dict
        Dicionário com ``df_to_hc``, ``df_sem_outliers``, ``df_original``,
        ``scaler``, ``pca_std``, ``colunas_numericas``.
    """
    if remover_outliers is not None:
        outlier_method = "iqr" if remover_outliers else "none"

    label_method = {
        "iqr": "IQR (k=1.5)",
        "zscore": "Z-score (k=3)",
        "mad": "MAD (k=3.5)",
        "none": "nenhum filtro",
    }.get(outlier_method, outlier_method)

    print("=" * 60)
    print(f"ETAPA 1 — PRÉ-PROCESSAMENTO  [outliers: {label_method}]")
    print("=" * 60)

    df = carregar_arquivo(caminho_entrada)
    print(f"Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")

    df_columns_filter = limpeza_dados(df)
    print(f"Após seleção de colunas: {df_columns_filter.shape}")

    df_lim = def_lines_limits(df_columns_filter)
    print(f"Após filtro de limites: {df_lim.shape}")

    df_inc_rm = remocao_incertezas(df_lim)
    print(f"Após remoção de incertezas: {df_inc_rm.shape}")

    id_outliers(df_inc_rm, save_dir=save_dir, show=show)

    df_sem_outliers = rm_outliers(df_inc_rm, method=outlier_method,
                                  threshold=outlier_threshold)
    print(f"Após rm_outliers ({outlier_method}): {df_sem_outliers.shape}")

    (df_to_hc, pca_std, scaler,
     colunas_numericas, var_pc1, var_pc2) = pca_standard_scaler(df_sem_outliers)

    df_loadings, top3_pc1, top3_pc2, soma_pc1, soma_pc2 = analise_cargas(
        pca_std, colunas_numericas
    )

    ranges_terrestres(scaler, colunas_numericas, caminho_rgjson)

    pca_plot_2D(df_to_hc, var_pc1, var_pc2,
                top3_pc1, top3_pc2, soma_pc1, soma_pc2,
                save_dir=save_dir, show=show)

    df_to_hc["pl_name"] = df.loc[df_to_hc.index, "pl_name"]

    Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
    df_to_hc.to_excel(caminho_saida, index=True)
    print(f"\nSalvo: {caminho_saida}")

    return {
        "df_to_hc": df_to_hc,
        "df_sem_outliers": df_sem_outliers,
        "df_original": df,
        "scaler": scaler,
        "pca_std": pca_std,
        "colunas_numericas": colunas_numericas,
    }


if __name__ == "__main__":
    main(
        caminho_entrada="../data/PSCompData.xlsx",
        caminho_saida="../outputs/DFHierarchicalClustering.xlsx",
        caminho_rgjson="../outputs/ranges_terrestres.json",
        save_dir="../outputs/figs",
        show=False,
    )
