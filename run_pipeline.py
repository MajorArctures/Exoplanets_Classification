#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orquestrador do pipeline de Classificação de Exoplanetas
=========================================================

Roda todas as 5 etapas em sequência:
    1. Pré-processamento (limpeza + PCA + ranges terrestres)
    2. Hierarchical Clustering (Ward, 4 clusters)
    3. Label Propagation (TT/NT semi-supervisionado)
    4. SVM (classificação supervisionada)
    5. Análise de similaridade com a Terra

Uso:
    python run_pipeline.py
    python run_pipeline.py --show          # abre janelas dos gráficos
    python run_pipeline.py --data caminho.xlsx --out ./meus_resultados
"""

import argparse
from pathlib import Path

from pipeline import (hierarchical_clustering, label_propagation,
                      preprocessing, similaridade_terra, svm)


def run(data_path, out_dir, show=False, outlier_method="iqr",
        outlier_threshold=None, metrica_proximidade="manhattan"):
    out = Path(out_dir)
    figs = out / "figs"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    df_hc_path = out / "DFHierarchicalClustering.xlsx"
    ranges_path = out / "ranges_terrestres.json"
    df_lp_path = out / "DFLabelPropagation.xlsx"
    info_prox_path = out / "info_proximidade.json"
    df_svm_path = out / "DFsvm.xlsx"
    resultados_csv = out / "resultados_svm.csv"
    similaridade_csv = out / "similaridade_exoplanetas_terra.csv"

    # 1. Pré-processamento
    result_pp = preprocessing.main(
        caminho_entrada=str(data_path),
        caminho_saida=str(df_hc_path),
        caminho_rgjson=str(ranges_path),
        outlier_method=outlier_method,
        outlier_threshold=outlier_threshold,
        save_dir=str(figs),
        show=show,
    )

    # 2. Hierarchical Clustering
    hierarchical_clustering.main(
        caminho_entrada=str(df_hc_path),
        caminho_ranges=str(ranges_path),
        caminho_saida=str(df_lp_path),
        caminho_info_json=str(info_prox_path),
        metrica_proximidade=metrica_proximidade,
        save_dir=str(figs),
        show=show,
    )

    # 3. Label Propagation
    label_propagation.main(
        caminho_entrada=str(df_lp_path),
        caminho_info_proximidade=str(info_prox_path),
        caminho_saida=str(df_svm_path),
        save_dir=str(figs),
        show=show,
    )

    # 4. SVM
    result_svm = svm.main(
        caminho_entrada=str(df_svm_path),
        caminho_saida_csv=str(resultados_csv),
        save_dir=str(figs),
        show=show,
    )

    # 5. Similaridade — usa scaler e df_sem_outliers do estágio 1
    similaridade_terra.main(
        df_sem_outliers=result_pp["df_sem_outliers"].assign(
            pl_name=result_pp["df_original"].loc[
                result_pp["df_sem_outliers"].index, "pl_name"
            ]
        ),
        scaler=result_pp["scaler"],
        colunas_numericas=result_pp["colunas_numericas"],
        caminho_saida_csv=str(similaridade_csv),
    )

    print("\n" + "=" * 60)
    print("PIPELINE CONCLUÍDO")
    print("=" * 60)
    print(f"Diretório de saída: {out.resolve()}")
    if result_svm["acuracia"] is not None:
        print(f"Acurácia final SVM: {result_svm['acuracia']:.1%}")
    else:
        print("SVM saltado (distribuição degenerada de rótulos).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de classificação de exoplanetas")
    parser.add_argument("--data", default="data/PSCompData.xlsx",
                        help="Caminho do dataset (.xlsx ou .csv)")
    parser.add_argument("--out", default="outputs",
                        help="Diretório de saída")
    parser.add_argument("--show", action="store_true",
                        help="Exibe as janelas dos gráficos (padrão: só salva PNG)")
    parser.add_argument("--outlier-method", dest="outlier_method",
                        choices=["iqr", "zscore", "mad", "none"],
                        default="iqr",
                        help="Método de remoção de outliers (default: iqr)")
    parser.add_argument("--outlier-threshold", dest="outlier_threshold",
                        type=float, default=None,
                        help="Threshold do método (default: 1.5/3.0/3.5 para iqr/zscore/mad)")
    parser.add_argument("--no-outliers", dest="no_outliers",
                        action="store_true",
                        help="Atalho para --outlier-method none")
    parser.add_argument("--metrica", dest="metrica_proximidade",
                        choices=["manhattan", "euclidiana"],
                        default="manhattan",
                        help="Métrica de proximidade ao centro terrestre "
                             "(default: manhattan, como no pipeline original)")
    args = parser.parse_args()

    method = "none" if args.no_outliers else args.outlier_method
    run(args.data, args.out, show=args.show,
        outlier_method=method, outlier_threshold=args.outlier_threshold,
        metrica_proximidade=args.metrica_proximidade)
