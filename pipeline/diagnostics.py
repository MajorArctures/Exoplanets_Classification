# -*- coding: utf-8 -*-
"""
Módulo de Diagnósticos
======================

Ferramentas de diagnóstico metodológico do pipeline.

Investiga duas questões críticas:

1. **Circularidade**: quanto o SVM "aprende" além do que já estava no
   agrupamento hierárquico e no Label Propagation?

   - Se ``agreement(SVM, LP) ≈ 100%``, o SVM está apenas reproduzindo o LP.
   - Se ``agreement(baseline_cluster, LP) ≈ 100%``, o LP está apenas
     propagando a atribuição do cluster.

2. **Validação externa**: os rótulos TT/NT do pipeline correspondem à
   intuição física esperada para corpos conhecidos? Testa o modelo em
   planetas do Sistema Solar e exoplanetas de referência que NUNCA
   estiveram no treino.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

COLUNAS_NUMERICAS = ["pl_orbper", "pl_orbsmax", "pl_rade", "pl_bmasse", "pl_orbeccen"]


# Corpos de referência para validação externa
# Fonte: NASA Fact Sheets (Sistema Solar) e NASA Exoplanet Archive (exoplanetas)
# Convenção esperada: TT (1) para planetas rochosos parecidos com a Terra,
# NT (0) para gigantes gasosos/de gelo.
CORPOS_REFERENCIA = pd.DataFrame([
    # Sistema Solar — planetas rochosos
    {"nome": "Mercúrio", "tipo_esperado": "Rochoso (fora do range TT do pipeline)",
     "pl_orbper": 87.97,   "pl_orbsmax": 0.387, "pl_rade": 0.383, "pl_bmasse": 0.055, "pl_orbeccen": 0.2056},
    {"nome": "Vênus",    "tipo_esperado": "Rochoso Earth-like",
     "pl_orbper": 224.70,  "pl_orbsmax": 0.723, "pl_rade": 0.949, "pl_bmasse": 0.815, "pl_orbeccen": 0.0068},
    {"nome": "Terra",    "tipo_esperado": "Rochoso (referência)",
     "pl_orbper": 365.25,  "pl_orbsmax": 1.000, "pl_rade": 1.000, "pl_bmasse": 1.000, "pl_orbeccen": 0.0167},
    {"nome": "Marte",    "tipo_esperado": "Rochoso (fora do range TT do pipeline)",
     "pl_orbper": 686.97,  "pl_orbsmax": 1.524, "pl_rade": 0.532, "pl_bmasse": 0.107, "pl_orbeccen": 0.0934},
    # Sistema Solar — gigantes
    {"nome": "Júpiter",  "tipo_esperado": "Gigante gasoso",
     "pl_orbper": 4332.6, "pl_orbsmax":  5.20, "pl_rade": 11.21, "pl_bmasse": 317.83, "pl_orbeccen": 0.0489},
    {"nome": "Saturno",  "tipo_esperado": "Gigante gasoso",
     "pl_orbper": 10759,  "pl_orbsmax":  9.58, "pl_rade":  9.45, "pl_bmasse":  95.16, "pl_orbeccen": 0.0565},
    {"nome": "Urano",    "tipo_esperado": "Gigante de gelo",
     "pl_orbper": 30687,  "pl_orbsmax": 19.20, "pl_rade":  4.01, "pl_bmasse":  14.54, "pl_orbeccen": 0.0457},
    {"nome": "Netuno",   "tipo_esperado": "Gigante de gelo",
     "pl_orbper": 60190,  "pl_orbsmax": 30.05, "pl_rade":  3.88, "pl_bmasse":  17.15, "pl_orbeccen": 0.0113},
])


def relatorio_circularidade(df_svm, modelo_svm,
                            cluster_mais_proximo, cluster_mais_distante):
    """
    Investigação da circularidade Cluster → LP → SVM.

    Compara três atribuições sobre TODO o dataset (sem split):

    - **A) Baseline trivial**: cada cluster recebe o rótulo majoritário do
      LP dentro dele. Aproximação do que o LP faria sozinho a partir do
      clustering.
    - **B) Label Propagation**: rótulos armazenados na coluna ``label_lp``.
    - **C) SVM**: previsões do modelo treinado.

    Se ``agreement(A, B) ≈ 100%``, o LP agrega pouca informação além do
    clustering. Se ``agreement(C, B) ≈ 100%``, o SVM apenas reproduz o LP.

    Parâmetros
    ----------
    df_svm : pandas.DataFrame
        DataFrame carregado de ``DFsvm.xlsx`` (com ``cluster_hc``,
        ``label_lp`` e features).
    modelo_svm : sklearn.svm.SVC
        Modelo SVM treinado.
    cluster_mais_proximo : int
        Cluster usado como seed TT.
    cluster_mais_distante : int
        Cluster usado como seed NT.

    Retorna
    -------
    dict
        Chaves ``acc_baseline_vs_lp``, ``acc_svm_vs_lp``,
        ``acc_svm_vs_baseline``.
    """
    print("=" * 60)
    print("DIAGNÓSTICO DE CIRCULARIDADE")
    print("=" * 60)

    features = [c for c in df_svm.columns
                if c not in ["pl_name", "cluster_hc", "PC1", "PC2", "label_lp"]]
    X_all = df_svm[features].values
    y_lp = df_svm["label_lp"].values
    clusters = df_svm["cluster_hc"].values

    y_baseline = np.full(len(clusters), -1, dtype=int)
    for c in np.unique(clusters):
        mask = clusters == c
        if c == cluster_mais_proximo:
            y_baseline[mask] = 1
        elif c == cluster_mais_distante:
            y_baseline[mask] = 0
        else:
            maioria = pd.Series(y_lp[mask]).mode().iloc[0]
            y_baseline[mask] = maioria

    y_svm = modelo_svm.predict(X_all)

    acc_base_vs_lp = accuracy_score(y_lp, y_baseline)
    acc_svm_vs_lp = accuracy_score(y_lp, y_svm)
    acc_base_vs_svm = accuracy_score(y_svm, y_baseline)

    print(f"\nAmostras: {len(df_svm)}")
    print(f"\n[A] Baseline trivial (majoritário por cluster) vs Label Propagation:")
    print(f"    Concordância: {acc_base_vs_lp:.3f}  ({int(acc_base_vs_lp*len(df_svm))}/{len(df_svm)})")

    print(f"\n[B] SVM vs Label Propagation:")
    print(f"    Concordância: {acc_svm_vs_lp:.3f}  ({int(acc_svm_vs_lp*len(df_svm))}/{len(df_svm)})")

    print(f"\n[C] SVM vs Baseline trivial:")
    print(f"    Concordância: {acc_base_vs_svm:.3f}  ({int(acc_base_vs_svm*len(df_svm))}/{len(df_svm)})")

    print("\nCrosstab (LP × SVM em todo o dataset):")
    print(pd.crosstab(pd.Series(y_lp, name="LP"),
                      pd.Series(y_svm, name="SVM")))

    print("\nCrosstab (Cluster HC × LP):")
    print(pd.crosstab(pd.Series(clusters, name="cluster_hc"),
                      pd.Series(y_lp, name="label_lp")))

    return {
        "acc_baseline_vs_lp": acc_base_vs_lp,
        "acc_svm_vs_lp": acc_svm_vs_lp,
        "acc_svm_vs_baseline": acc_base_vs_svm,
    }


def validacao_externa(modelo_svm, scaler, corpos=None):
    """
    Validação externa em corpos conhecidos (Sistema Solar).

    Aplica o modelo SVM em planetas do Sistema Solar (e opcionalmente
    exoplanetas de referência). Nenhum destes corpos esteve no treino, então
    é uma validação genuína out-of-sample.

    Parâmetros
    ----------
    modelo_svm : sklearn.svm.SVC
        Modelo SVM treinado.
    scaler : sklearn.preprocessing.StandardScaler
        Scaler ajustado na etapa 1 (o MESMO usado no treino).
    corpos : pandas.DataFrame, opcional
        DataFrame customizado. Default = :data:`CORPOS_REFERENCIA`.

    Retorna
    -------
    pandas.DataFrame
        DataFrame com colunas ``nome``, ``tipo_esperado``, ``predicao_svm``,
        ``|z|_max`` (medida de extrapolação) e ``ood?``.
    """
    print("=" * 60)
    print("VALIDAÇÃO EXTERNA (Sistema Solar + exoplanetas de referência)")
    print("=" * 60)

    corpos = corpos if corpos is not None else CORPOS_REFERENCIA
    X_df = corpos[COLUNAS_NUMERICAS]
    X_std = scaler.transform(X_df)
    y_pred = modelo_svm.predict(X_std)

    z_max = np.abs(X_std).max(axis=1)

    resultado = corpos[["nome", "tipo_esperado"]].copy()
    resultado["predicao_svm"] = [
        "TT (Tipo Terrestre)" if p == 1 else "NT (Não Terrestre)"
        for p in y_pred
    ]
    resultado["|z|_max"] = z_max.round(2)
    resultado["ood?"] = ["SIM" if z > 4 else "não" for z in z_max]

    print(resultado.to_string(index=False))

    return resultado
