# -*- coding: utf-8 -*-
"""
Módulo Hierarchical Clustering
==============================

Aplicação do algoritmo de clusterização hierárquica em dados de exoplanetas,
com base nos componentes principais obtidos na etapa de pré-processamento.

Este módulo realiza a formação de clusters por meio do método hierárquico,
sua visualização no espaço PCA e a análise da similaridade dos agrupamentos
em relação às características da Terra.

As etapas são organizadas de forma modular, permitindo a extensão gradual
das análises ao longo do desenvolvimento do projeto.

Inclui
------
- Aplicação do algoritmo de clusterização hierárquica (linkage)
- Extração dos rótulos dos clusters
- Plot do dendrograma
- Visualização dos clusters no espaço PCA (scatter plot)
- Avaliação dos clusters em relação a planetas terrestres
- Análise de proximidade aos parâmetros terrestres
- Cálculo da distância de cada cluster ao centro terrestre (Manhattan ou Euclidiana)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage


COLUNAS_NUMERICAS = ["pl_orbper", "pl_orbsmax", "pl_rade", "pl_bmasse", "pl_orbeccen"]


def _save_and_show(fig_name, save_dir, show):
    """Helper interno: salva figura (se save_dir) e exibe (se show)."""
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_dir) / fig_name, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def hc_application(df_to_hc, method="ward", n_clusters=4):
    """
    3.0 - Aplicação do Hierarchical Clustering.

    Aplica o algoritmo de agrupamento hierárquico usando o método de linkage
    especificado sobre as variáveis físicas padronizadas.

    Parâmetros
    ----------
    df_to_hc : pandas.DataFrame
        Dados de entrada (contendo as 5 variáveis físicas padronizadas).
    method : str, opcional
        Método de linkage (default: ``'ward'``).
    n_clusters : int, opcional
        Número de clusters desejados (default: 4).

    Retorna
    -------
    Z : numpy.ndarray
        Matriz de linkage.
    clusters : numpy.ndarray
        Rótulos dos clusters atribuídos a cada amostra.
    dist_corte : float
        Distância de corte correspondente à formação dos clusters.
    """
    X = df_to_hc[COLUNAS_NUMERICAS].to_numpy()
    Z = linkage(X, method=method)
    clusters = fcluster(Z, n_clusters, criterion="maxclust")
    dist_corte = np.sort(Z[:, 2])[-(n_clusters - 1)]
    return Z, clusters, dist_corte


def plot_dendrogram(Z, dist_corte=None, save_dir=None, show=True):
    """
    3.2 - Dendrograma do Agrupamento Hierárquico.

    Plota o dendrograma correspondente à matriz de linkage obtida
    pelo algoritmo de clusterização hierárquica. Quando fornecida, a
    distância de corte é utilizada para destacar visualmente a separação
    dos clusters no dendrograma.

    Parâmetros
    ----------
    Z : numpy.ndarray
        Matriz de linkage gerada pelo algoritmo de clusterização.
    dist_corte : float, opcional
        Distância de corte utilizada para visualização da formação dos
        clusters (default: None).
    save_dir : str, opcional
        Se fornecido, salva o dendrograma como PNG neste diretório.
    show : bool, opcional
        Se ``True`` (padrão), exibe interativamente.

    Retorna
    -------
    None
    """
    plt.figure(figsize=(18, 7))
    dendrogram(Z, color_threshold=dist_corte,
               above_threshold_color="gray", no_labels=True)

    if dist_corte is not None:
        plt.axhline(y=dist_corte, color="crimson", linestyle="--",
                    linewidth=2, label=f"Corte = {dist_corte:.2f}")
        plt.legend()

    plt.title("Dendrograma – Agrupamento Hierárquico (Ward)",
              fontsize=13, fontweight="bold")
    plt.xlabel("Amostras")
    plt.ylabel("Distância Euclidiana")
    plt.tight_layout()
    _save_and_show("dendrograma.png", save_dir, show)


def scatter_plot_PCA(X_pca, clusters, pca_model=None, save_dir=None, show=True):
    """
    Scatter dos clusters no espaço PCA 2D.

    Plota os pontos coloridos por cluster no espaço formado por PC1 e PC2,
    permitindo visualização da separação das classes obtidas pelo
    clustering hierárquico.

    Parâmetros
    ----------
    X_pca : numpy.ndarray ou pandas.DataFrame
        Coordenadas no espaço PCA (2 dimensões).
    clusters : array-like
        Rótulos de cluster de cada ponto.
    pca_model : sklearn.decomposition.PCA, opcional
        Modelo PCA ajustado (para exibir % variância nos eixos).
    save_dir : str, opcional
        Se fornecido, salva como ``clusters_pca.png``.
    show : bool, opcional
        Se ``True``, exibe interativamente.

    Retorna
    -------
    None
    """
    if isinstance(X_pca, pd.DataFrame):
        X_pca = X_pca[["PC1", "PC2"]].to_numpy()

    if pca_model is not None:
        var_pc1 = pca_model.explained_variance_ratio_[0] * 100
        var_pc2 = pca_model.explained_variance_ratio_[1] * 100
        xlabel = f"PC1 ({var_pc1:.1f}% var.)"
        ylabel = f"PC2 ({var_pc2:.1f}% var.)"
    else:
        xlabel, ylabel = "PC1", "PC2"

    plt.figure(figsize=(8, 6))
    plt.title("Clusters de Exoplanetas no Espaço PCA\n(Agrupamento Hierárquico - Ward)",
              fontsize=12, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for c in sorted(np.unique(clusters)):
        mask = clusters == c
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=f"Cluster {c} (n={np.sum(mask)})", alpha=0.7, s=50)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save_and_show("clusters_pca.png", save_dir, show)

    print("\nEstatísticas dos clusters:")
    for c in sorted(np.unique(clusters)):
        n = np.sum(clusters == c)
        perc = (n / len(clusters)) * 100
        print(f"  Cluster {c}: {n} planetas ({perc:.1f}%)")


def extr_rot_clusters(df_to_hc, clusters):
    """
    3.3 - Extração dos rótulos dos clusters.

    Constrói um DataFrame com o nome (ou índice) de cada planeta e o
    cluster ao qual foi atribuído.

    Parâmetros
    ----------
    df_to_hc : pandas.DataFrame
        DataFrame com os dados tratados.
    clusters : array-like
        Rótulos dos clusters.

    Retorna
    -------
    rotulos_clusters : pandas.DataFrame
        DataFrame com colunas ``Planeta`` (índice do df) e ``Cluster``.
    """
    return pd.DataFrame({
        "Planeta": df_to_hc.index,
        "Cluster": clusters,
    })


def carregar_ranges_padronizados(caminho):
    """
    Carrega o JSON de ranges terrestres do disco.

    Parâmetros
    ----------
    caminho : str
        Caminho do arquivo JSON produzido por
        :func:`pipeline.preprocessing.ranges_terrestres`.

    Retorna
    -------
    dict
        Dicionário ``{variavel: [z_min, z_max]}`` com faixas terrestres
        em espaço padronizado.
    """
    with open(caminho, "r") as f:
        return json.load(f)


def analise_proximidade_terra(df_to_hc, clusters, ranges_padronizados,
                              colunas_numericas, metrica="manhattan"):
    """
    3.4 - Análise de Proximidade aos Parâmetros Terrestres.

    Calcula a distância entre o centro médio de cada cluster e o centro
    terrestre definido no espaço padronizado (z-score), identificando o
    cluster mais próximo (candidato TT) e o mais distante (candidato NT).

    Métricas disponíveis
    --------------------
    - **manhattan** (padrão, usada no pipeline original): soma dos módulos
      das diferenças, ``d = Σ |μ_c[v] - centro[v]|``. Menos sensível a
      diferenças grandes em uma única variável.
    - **euclidiana** (L2): raiz da soma dos quadrados,
      ``d = √(Σ (μ_c[v] - centro[v])²)``. Pune mais fortemente diferenças
      grandes em qualquer dimensão.

    Método
    ------
    - Para cada cluster ``c``, calcula a média das 5 variáveis físicas
      padronizadas.
    - Calcula a distância (Manhattan ou Euclidiana) dessa média ao centro
      terrestre (média das faixas em z-space).
    - Ordena os clusters pela distância ascendente.
    - Cluster com menor distância → TT (mais próximo).
    - Cluster com maior distância → NT (mais distante).

    Parâmetros
    ----------
    df_to_hc : pandas.DataFrame
        DataFrame contendo as variáveis físicas padronizadas.
    clusters : array-like
        Vetor de rótulos de cluster (mesma ordem de df_to_hc).
    ranges_padronizados : dict
        Faixas terrestres em espaço padronizado.
    colunas_numericas : list of str
        Nomes das variáveis físicas.
    metrica : {'manhattan', 'euclidiana'}, opcional
        Métrica de distância. Padrão ``'manhattan'`` (preserva
        comportamento do pipeline original).

    Retorna
    -------
    distancias : pandas.Series
        Distâncias de cada cluster ao centro terrestre, ordenadas
        ascendente.
    cluster_mais_proximo : int
        Cluster identificado como TT (menor distância).
    cluster_mais_distante : int
        Cluster identificado como NT (maior distância).

    Raises
    ------
    ValueError
        Se o número de amostras não coincide com o número de clusters, ou
        se ``metrica`` for inválida.
    """
    if len(df_to_hc) != len(clusters):
        raise ValueError("Número de amostras diferente do número de clusters.")

    if metrica not in ("manhattan", "euclidiana"):
        raise ValueError(
            f"metrica='{metrica}' inválida. Use 'manhattan' ou 'euclidiana'."
        )

    df_clusters = df_to_hc.copy()
    df_clusters["Cluster"] = clusters

    centro_terrestre = {
        var: (lo + hi) / 2
        for var, (lo, hi) in ranges_padronizados.items()
        if var in colunas_numericas
    }

    dist_por_cluster = {}
    for c in sorted(np.unique(clusters)):
        medias = df_clusters[df_clusters["Cluster"] == c][colunas_numericas].mean()
        if metrica == "manhattan":
            d = sum(
                abs(medias[var] - centro_terrestre[var])
                for var in centro_terrestre
                if var in medias.index
            )
        else:  # euclidiana
            soma_q = sum(
                (medias[var] - centro_terrestre[var]) ** 2
                for var in centro_terrestre
                if var in medias.index
            )
            d = float(np.sqrt(soma_q))
        dist_por_cluster[c] = float(d)

    distancias = pd.Series(dist_por_cluster).sort_values()
    cluster_mais_proximo = int(distancias.index[0])
    cluster_mais_distante = int(distancias.index[-1])

    print(f"\nDistância ao centro terrestre ({metrica}):")
    for cluster, dist in distancias.items():
        print(f"  Cluster {cluster}: {dist:.5f}")
    print(f"\n  → Cluster mais próximo:  {cluster_mais_proximo}")
    print(f"  → Cluster mais distante: {cluster_mais_distante}")

    return distancias, cluster_mais_proximo, cluster_mais_distante


def cluster_info_json(cluster_mais_proximo, cluster_mais_distante, caminho_saida):
    """
    Exporta informações de proximidade dos clusters em formato JSON.

    Persiste as informações sobre qual cluster é considerado TT (mais próximo
    da Terra) e qual é NT (mais distante), para uso pela etapa de Label
    Propagation.

    Parâmetros
    ----------
    cluster_mais_proximo : int
        Cluster cujo centro está mais próximo dos parâmetros terrestres (TT).
    cluster_mais_distante : int
        Cluster cujo centro está mais distante dos parâmetros terrestres (NT).
    caminho_saida : str
        Caminho do arquivo JSON a ser salvo.

    Retorna
    -------
    dict
        Dicionário contendo as informações salvas
        (``cluster_mais_proximo``, ``cluster_mais_distante``).
    """
    info = {
        "cluster_mais_proximo": int(cluster_mais_proximo),
        "cluster_mais_distante": int(cluster_mais_distante),
    }
    Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_saida, "w") as f:
        json.dump(info, f, indent=4)
    return info


def main(caminho_entrada, caminho_ranges, caminho_saida, caminho_info_json,
         n_clusters=4, metodo_linkage="ward",
         metrica_proximidade="manhattan",
         save_dir=None, show=False):
    """
    Executa toda a etapa 2 (hierarchical clustering) do pipeline.

    Encadeia:
    :func:`hc_application` → :func:`plot_dendrogram` →
    :func:`scatter_plot_PCA` → :func:`carregar_ranges_padronizados` →
    :func:`analise_proximidade_terra` → :func:`cluster_info_json`.

    Parâmetros
    ----------
    caminho_entrada : str
        Caminho do Excel produzido pela etapa 1 (preprocessing).
    caminho_ranges : str
        Caminho do JSON com as faixas terrestres padronizadas.
    caminho_saida : str
        Caminho do Excel de saída (entrada da etapa 3).
    caminho_info_json : str
        Caminho do JSON com identificação dos clusters TT/NT.
    n_clusters : int, opcional
        Número de clusters a formar (padrão 4).
    metodo_linkage : str, opcional
        Método de linkage (padrão ``'ward'``).
    metrica_proximidade : {'manhattan', 'euclidiana'}, opcional
        Métrica para identificar TT/NT. Padrão ``'manhattan'``.
    save_dir : str, opcional
        Diretório para salvar figuras.
    show : bool, opcional
        Se ``True``, exibe figuras interativamente.

    Retorna
    -------
    dict
        Chaves ``df_lp``, ``clusters``, ``distancias``,
        ``cluster_mais_proximo``, ``cluster_mais_distante``, ``Z``.
    """
    print("=" * 60)
    print(f"ETAPA 2 — HIERARCHICAL CLUSTERING  [métrica: {metrica_proximidade}]")
    print("=" * 60)

    df = pd.read_excel(caminho_entrada, index_col=0)
    print(f"Dataset carregado: {df.shape}")

    Z, clusters, dist_corte = hc_application(
        df, method=metodo_linkage, n_clusters=n_clusters
    )
    print(f"\nClusters formados: {n_clusters} | Distância de corte: {dist_corte:.4f}")

    plot_dendrogram(Z, dist_corte, save_dir=save_dir, show=show)

    X_pca = df[["PC1", "PC2"]].values
    scatter_plot_PCA(X_pca, clusters, pca_model=None,
                     save_dir=save_dir, show=show)

    ranges_padronizados = carregar_ranges_padronizados(caminho_ranges)

    distancias, cluster_prox, cluster_dist = analise_proximidade_terra(
        df[COLUNAS_NUMERICAS], clusters, ranges_padronizados,
        COLUNAS_NUMERICAS, metrica=metrica_proximidade
    )

    df_lp = df.copy()
    df_lp["cluster_hc"] = clusters

    Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
    df_lp.to_excel(caminho_saida, index=True)
    print(f"\nSalvo: {caminho_saida}")

    cluster_info_json(cluster_prox, cluster_dist, caminho_info_json)
    print(f"Salvo: {caminho_info_json}")

    return {
        "df_lp": df_lp,
        "clusters": clusters,
        "distancias": distancias,
        "cluster_mais_proximo": cluster_prox,
        "cluster_mais_distante": cluster_dist,
        "Z": Z,
    }


if __name__ == "__main__":
    main(
        caminho_entrada="../outputs/DFHierarchicalClustering.xlsx",
        caminho_ranges="../outputs/ranges_terrestres.json",
        caminho_saida="../outputs/DFLabelPropagation.xlsx",
        caminho_info_json="../outputs/info_proximidade.json",
        save_dir="../outputs/figs",
        show=False,
    )
