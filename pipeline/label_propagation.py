# -*- coding: utf-8 -*-
"""
Módulo Label Propagation
========================

Etapa semi-supervisionada do pipeline. Cria rótulos iniciais a partir da
identificação de cluster TT (Tipo Terrestre) e cluster NT (Não Terrestre)
realizada na etapa 2, deixando os demais planetas como não rotulados (-1).
Em seguida, propaga esses rótulos pelo restante do dataset usando o
algoritmo LabelPropagation do scikit-learn com kernel k-NN.

Convenção de rótulos
--------------------
- ``-1`` : não rotulado
- ``1`` : TT (Tipo Terrestre)
- ``0`` : NT (Não Terrestre)

Inclui
------
- Criação dos rótulos semi-supervisionados iniciais (y_semi)
- Aplicação do algoritmo Label Propagation
- Visualização dos rótulos propagados no espaço PCA
- Exportação do dataset final para uso pelo SVM
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.semi_supervised import LabelPropagation


def _save_and_show(fig_name, save_dir, show):
    """Helper interno: salva figura (se save_dir) e exibe (se show)."""
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_dir) / fig_name, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def carregar_info_proximidade(caminho):
    """
    Carrega o JSON com identificação dos clusters TT e NT.

    Lê o arquivo produzido por
    :func:`pipeline.hierarchical_clustering.cluster_info_json`.

    Parâmetros
    ----------
    caminho : str
        Caminho do arquivo JSON.

    Retorna
    -------
    dict
        Dicionário com chaves ``cluster_mais_proximo`` (TT) e
        ``cluster_mais_distante`` (NT).
    """
    with open(caminho, "r") as f:
        return json.load(f)


def label_creation(clusters, cluster_mais_proximo, cluster_mais_distante):
    """
    4.0 - Criação dos Rótulos Semi-Supervisionados (y_semi).

    Constrói o vetor inicial de rótulos com a convenção:

    - Cluster mais próximo do centro terrestre → rótulo ``1`` (TT)
    - Cluster mais distante do centro terrestre → rótulo ``0`` (NT)
    - Demais clusters → rótulo ``-1`` (não rotulado, será preenchido pelo LP)

    Parâmetros
    ----------
    clusters : array-like
        Rótulos de cluster do hierarchical clustering.
    cluster_mais_proximo : int
        Índice do cluster identificado como TT.
    cluster_mais_distante : int
        Índice do cluster identificado como NT.

    Retorna
    -------
    y_semi : numpy.ndarray
        Vetor de rótulos iniciais (-1, 0 ou 1).
    """
    print("Clusters únicos:", np.unique(clusters))

    y_semi = np.full(len(clusters), -1, dtype=int)
    y_semi[clusters == cluster_mais_proximo] = 1
    y_semi[clusters == cluster_mais_distante] = 0

    print("\nRótulos iniciais (contagem):")
    print(pd.Series(y_semi).value_counts(sort=False))

    return y_semi


def label_propagation(X, y_semi, clusters, n_neighbors=7, max_iter=1000):
    """
    4.1 - Aplicação do Label Propagation.

    Aplica o algoritmo :class:`sklearn.semi_supervised.LabelPropagation`
    com kernel k-NN sobre os dados padronizados, propagando os rótulos
    conhecidos (TT/NT) para os planetas inicialmente não rotulados.

    Valida também a acurácia nos rótulos originais (deve ser próxima de 100%
    porque LP idealmente respeita as seeds).

    Parâmetros
    ----------
    X : array-like
        Dados de entrada (5 variáveis físicas padronizadas).
    y_semi : numpy.ndarray
        Vetor de rótulos iniciais (com -1 nos não rotulados).
    clusters : array-like
        Rótulos de cluster do HC (usado para crosstab de diagnóstico).
    n_neighbors : int, opcional
        Número de vizinhos no kernel k-NN (padrão 7).
    max_iter : int, opcional
        Iterações máximas do algoritmo (padrão 1000).

    Retorna
    -------
    labels_final : numpy.ndarray
        Vetor de rótulos após propagação (todos preenchidos com 0 ou 1).
    """
    lp = LabelPropagation(kernel="knn", n_neighbors=n_neighbors, max_iter=max_iter)
    lp.fit(X, y_semi)
    labels_final = lp.transduction_

    print("\nRótulos finais (contagem):")
    print(pd.Series(labels_final).value_counts())

    mask_known = y_semi != -1
    if mask_known.sum() > 0:
        acc = (labels_final[mask_known] == y_semi[mask_known]).mean()
        print(f"\nAcurácia nos rótulos iniciais: {acc:.3f} "
              f"({mask_known.sum()} instâncias verificadas)")

    ct = pd.crosstab(pd.Series(clusters, name="cluster_init"),
                     pd.Series(labels_final, name="label_propagado"))
    print("\nCrosstab (cluster inicial x label propagado):")
    print(ct)

    return labels_final


def plot_pca(X_pca, labels_final, y_semi, save_dir=None, show=True):
    """
    4.2 - Visualização dos rótulos propagados no espaço PCA 2D.

    Plota o scatter dos planetas coloridos por rótulo final (TT/NT), com
    destaque circular nos exemplos inicialmente rotulados (seeds).

    Parâmetros
    ----------
    X_pca : pandas.DataFrame
        DataFrame contendo colunas PC1 e PC2.
    labels_final : array-like
        Rótulos finais após propagação (0 ou 1).
    y_semi : array-like
        Rótulos iniciais (com -1 nos não rotulados).
    save_dir : str, opcional
        Se fornecido, salva como ``label_propagation_pca.png``.
    show : bool, opcional
        Se ``True``, exibe interativamente.

    Retorna
    -------
    None
    """
    var1 = np.var(X_pca["PC1"])
    var2 = np.var(X_pca["PC2"])
    total = var1 + var2
    perc1 = 100 * var1 / total
    perc2 = 100 * var2 / total

    label_names = {1: "TT — Tipo Terrestre", 0: "NT — Não Terrestre"}

    plt.figure(figsize=(10, 7))
    for lab in np.unique(labels_final):
        mask = labels_final == lab
        plt.scatter(X_pca.loc[mask, "PC1"], X_pca.loc[mask, "PC2"],
                    s=50, alpha=0.75,
                    label=f"{label_names.get(lab, str(lab))} (n={np.sum(mask)})")

    mask_known = y_semi != -1
    plt.scatter(X_pca.loc[mask_known, "PC1"], X_pca.loc[mask_known, "PC2"],
                facecolors="none", edgecolors="k", linewidths=0.8, s=80,
                label="Exemplos inicialmente rotulados")

    plt.xlabel(f"PC1 ({perc1:.1f}%)")
    plt.ylabel(f"PC2 ({perc2:.1f}%)")
    plt.title("Distribuição dos Exoplanetas após Label Propagation (PCA 2D)")
    plt.legend(loc="best", fontsize=10)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    _save_and_show("label_propagation_pca.png", save_dir, show)


def export_svm_dataset(df_base, clusters, labels_lp, caminho_saida):
    """
    4.3 - Exportação do dataset final para uso pelo SVM.

    Constrói o DataFrame com:

    - ``pl_name``: nome do planeta
    - as 5 variáveis físicas padronizadas
    - ``cluster_hc``: rótulo do hierarchical clustering
    - ``label_lp``: rótulo final após label propagation (0 ou 1)

    Parâmetros
    ----------
    df_base : pandas.DataFrame
        DataFrame com colunas ``pl_name``, as 5 features físicas e
        ``cluster_hc``.
    clusters : array-like
        Rótulos de cluster do HC.
    labels_lp : array-like
        Rótulos após label propagation.
    caminho_saida : str
        Caminho do Excel de saída (entrada da etapa 4).

    Retorna
    -------
    DFsvm : pandas.DataFrame
        DataFrame exportado.

    Raises
    ------
    ValueError
        Se os tamanhos de ``clusters`` ou ``labels_lp`` não coincidem com
        o de ``df_base``.
    """
    n = len(df_base)
    if len(clusters) != n:
        raise ValueError("Clusters e DataFrame com tamanhos diferentes")
    if len(labels_lp) != n:
        raise ValueError("Rótulos LP e DataFrame com tamanhos diferentes")

    colunas_excluir = ["pl_name", "cluster_hc", "PC1", "PC2"]
    colunas_numericas = [c for c in df_base.columns if c not in colunas_excluir]

    DFsvm = df_base[["pl_name"] + colunas_numericas].copy()
    DFsvm["cluster_hc"] = clusters
    DFsvm["label_lp"] = labels_lp

    Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
    DFsvm.to_excel(caminho_saida, index=True)

    print(f"\nSalvo: {caminho_saida}")
    print("Distribuição dos rótulos:")
    print(DFsvm["label_lp"].value_counts())

    return DFsvm


def main(caminho_entrada, caminho_info_proximidade, caminho_saida,
         save_dir=None, show=False):
    """
    Executa toda a etapa 3 (label propagation) do pipeline.

    Encadeia:
    :func:`carregar_info_proximidade` → :func:`label_creation` →
    :func:`label_propagation` → :func:`plot_pca` →
    :func:`export_svm_dataset`.

    Parâmetros
    ----------
    caminho_entrada : str
        Caminho do Excel produzido pela etapa 2.
    caminho_info_proximidade : str
        Caminho do JSON com identificação TT/NT.
    caminho_saida : str
        Caminho do Excel de saída (entrada da etapa 4).
    save_dir : str, opcional
        Diretório para salvar figuras.
    show : bool, opcional
        Se ``True``, exibe figuras interativamente.

    Retorna
    -------
    dict
        Chaves ``DFsvm``, ``labels_final``, ``y_semi``.
    """
    print("=" * 60)
    print("ETAPA 3 — LABEL PROPAGATION")
    print("=" * 60)

    df = pd.read_excel(caminho_entrada, index_col=0)
    print(f"Dataset carregado: {df.shape}")

    if "cluster_hc" not in df.columns:
        raise ValueError("Coluna 'cluster_hc' não encontrada.")

    clusters = df["cluster_hc"].values

    colunas_excluir = ["pl_name", "cluster_hc", "PC1", "PC2"]
    colunas_numericas = [c for c in df.columns if c not in colunas_excluir]

    info = carregar_info_proximidade(caminho_info_proximidade)
    cluster_prox = info["cluster_mais_proximo"]
    cluster_dist = info["cluster_mais_distante"]
    print(f"Cluster TT (mais próximo): {cluster_prox}")
    print(f"Cluster NT (mais distante): {cluster_dist}")

    y_semi = label_creation(clusters, cluster_prox, cluster_dist)
    labels_final = label_propagation(df[colunas_numericas], y_semi, clusters)

    plot_pca(df[["PC1", "PC2"]], labels_final, y_semi,
             save_dir=save_dir, show=show)

    DFsvm = export_svm_dataset(df, clusters, labels_final, caminho_saida)

    return {"DFsvm": DFsvm, "labels_final": labels_final, "y_semi": y_semi}


if __name__ == "__main__":
    main(
        caminho_entrada="../outputs/DFLabelPropagation.xlsx",
        caminho_info_proximidade="../outputs/info_proximidade.json",
        caminho_saida="../outputs/DFsvm.xlsx",
        save_dir="../outputs/figs",
        show=False,
    )
