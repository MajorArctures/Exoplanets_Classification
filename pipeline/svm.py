# -*- coding: utf-8 -*-
"""
Módulo Support Vector Machine (SVM)
====================================

Etapa supervisionada final do pipeline. Treina um classificador SVM (kernel
RBF) sobre os pseudo-rótulos gerados pelo Label Propagation e avalia seu
desempenho em conjuntos de ajuste (15%) e teste final (10%) estratificados.

Inclui
------
- Separação estratificada dos dados em treino (75%), ajuste (15%) e teste (10%)
- Treinamento do SVM
- Avaliação com matriz de confusão, precision, recall, F1 e acurácia
- Captura dos nomes dos exoplanetas do conjunto de teste
- Análise detalhada dos erros (VP/VN/FP/FN)
- Exportação dos resultados em CSV
- Tratamento gracioso do caso degenerado (LP produz apenas uma classe)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def _save_and_show(fig_name, save_dir, show):
    """Helper interno: salva figura (se save_dir) e exibe (se show)."""
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_dir) / fig_name, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def separacao_de_dados(df, rotulos, random_state=57):
    """
    5.0 - Separação estratificada dos dados (75 / 15 / 10).

    Divide o dataset em três subconjuntos preservando a proporção de classes:

    - Treino: 75% (para ajuste dos parâmetros do SVM)
    - Ajuste: 15% (para validação durante desenvolvimento)
    - Teste final: 10% (para avaliação final em dados nunca vistos)

    Os índices originais também são propagados para permitir recuperar os
    nomes dos planetas do conjunto de teste.

    Parâmetros
    ----------
    df : array-like ou pandas.DataFrame
        Features das amostras.
    rotulos : array-like
        Vetor de rótulos (0 ou 1).
    random_state : int, opcional
        Semente para reprodutibilidade (padrão 57).

    Retorna
    -------
    X_train, X_adj, X_test : array-like
        Features para cada conjunto.
    y_train, y_adj, y_test : array-like
        Rótulos correspondentes.
    idx_test : numpy.ndarray
        Índices originais das amostras do conjunto de teste.

    Raises
    ------
    ValueError
        Se alguma classe tiver menos de 2 amostras (impossibilita split
        estratificado).
    """
    print("Preparando dados para SVM:")
    print(f"  Shape: {df.shape}")
    print(f"  Distribuição rótulos: {pd.Series(rotulos).value_counts().to_dict()}")

    contagem = pd.Series(rotulos).value_counts()
    if (contagem < 2).any():
        raise ValueError(
            f"Distribuição degenerada: classes {contagem[contagem < 2].index.tolist()} "
            f"têm menos de 2 amostras. SVM supervisionado inviável."
        )

    indices = np.arange(len(df))

    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        df, rotulos, indices,
        test_size=0.25, random_state=random_state, stratify=rotulos
    )
    X_adj, X_test, y_adj, y_test, idx_adj, idx_test = train_test_split(
        X_temp, y_temp, idx_temp,
        test_size=0.4, random_state=random_state, stratify=y_temp
    )

    print(f"\nDivisão 75-15-10: Treino={len(X_train)} | Ajuste={len(X_adj)} | Teste={len(X_test)}")
    return X_train, X_adj, X_test, y_train, y_adj, y_test, idx_test


def treinamento_modelo_svm(X_train, y_train, C=1.0, gamma="scale", random_state=57):
    """
    5.1 - Treinamento do SVM.

    Treina um classificador SVM com kernel RBF sobre o conjunto de treino.

    Parâmetros
    ----------
    X_train : array-like
        Features do conjunto de treino.
    y_train : array-like
        Rótulos do conjunto de treino.
    C : float, opcional
        Parâmetro de regularização (padrão 1.0).
    gamma : str ou float, opcional
        Coeficiente do kernel RBF (padrão ``'scale'``).
    random_state : int, opcional
        Semente para reprodutibilidade (padrão 57).

    Retorna
    -------
    modelo : sklearn.svm.SVC
        Modelo SVM treinado.
    """
    modelo = SVC(kernel="rbf", C=C, gamma=gamma, random_state=random_state)
    modelo.fit(X_train, y_train)
    return modelo


def avaliacao_modelo(modelo, X, y, nome_conjunto, save_dir=None, show=True):
    """
    5.2 - Avaliação do modelo em um conjunto de dados.

    Aplica o modelo, gera matriz de confusão, imprime o classification_report
    e retorna a acurácia. Aceita conjuntos de ajuste ou teste final.

    Parâmetros
    ----------
    modelo : sklearn.svm.SVC
        Modelo SVM treinado.
    X : array-like
        Features do conjunto a avaliar.
    y : array-like
        Rótulos reais.
    nome_conjunto : str
        Nome descritivo (aparece no título e nome do PNG).
    save_dir : str, opcional
        Se fornecido, salva matriz de confusão como
        ``matriz_confusao_<nome>.png``.
    show : bool, opcional
        Se ``True``, exibe interativamente.

    Retorna
    -------
    previsoes : numpy.ndarray
        Vetor de previsões do modelo.
    matriz : numpy.ndarray
        Matriz de confusão 2x2.
    acuracia : float
        Acurácia calculada.
    """
    print(f"\nAvaliação — {nome_conjunto.upper()}")

    previsoes = modelo.predict(X)
    matriz = confusion_matrix(y, previsoes)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
                xticklabels=["NT", "TT"], yticklabels=["NT", "TT"])
    plt.title(f"Matriz de Confusão - SVM ({nome_conjunto})")
    plt.ylabel("Real")
    plt.xlabel("Predito")
    plt.tight_layout()
    _save_and_show(f"matriz_confusao_{nome_conjunto.lower().replace(' ', '_')}.png",
                   save_dir, show)

    print(f"\nRelatório de classificação ({nome_conjunto}):")
    print(classification_report(y, previsoes,
                                target_names=["Não-Terrestre", "Terrestre"]))
    acuracia = accuracy_score(y, previsoes)
    print(f"Acurácia: {acuracia:.3f}")

    return previsoes, matriz, acuracia


def captura_nomes_exoplanetas(df, idx_teste, y_test, previsoes):
    """
    5.3 - Captura dos nomes dos exoplanetas do conjunto de teste.

    Constrói uma tabela relacionando o nome de cada planeta do conjunto de
    teste, seu rótulo real, sua previsão e se a classificação foi correta.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame com coluna ``pl_name``.
    idx_teste : array-like
        Índices posicionais (via ``iloc``) das amostras do conjunto de teste.
    y_test : array-like
        Rótulos reais do conjunto de teste.
    previsoes : array-like
        Previsões do modelo.

    Retorna
    -------
    tabela : pandas.DataFrame
        DataFrame com colunas ``nome_planeta``, ``rotulo_original``,
        ``rotulo_predito``, ``previsao_correta``, ``categoria_original``,
        ``categoria_predita``.
    """
    nomes = df.iloc[idx_teste]["pl_name"].values

    tabela = pd.DataFrame({
        "nome_planeta": nomes,
        "rotulo_original": y_test,
        "rotulo_predito": previsoes,
        "previsao_correta": y_test == previsoes,
    })
    tabela["categoria_original"] = tabela["rotulo_original"].map(
        {0: "Não-Terrestre", 1: "Terrestre"}
    )
    tabela["categoria_predita"] = tabela["rotulo_predito"].map(
        {0: "Não-Terrestre", 1: "Terrestre"}
    )

    print(f"\nResultados finais: {len(tabela)} planetas | "
          f"Corretos: {tabela['previsao_correta'].sum()} | "
          f"Acurácia: {tabela['previsao_correta'].mean():.3f}")

    return tabela


def analise_detalhada_erros(tabela, matriz):
    """
    5.4 - Análise detalhada dos erros.

    Lista os planetas classificados incorretamente e imprime estatísticas
    dos verdadeiros/falsos positivos/negativos.

    Parâmetros
    ----------
    tabela : pandas.DataFrame
        Tabela retornada por :func:`captura_nomes_exoplanetas`.
    matriz : numpy.ndarray
        Matriz de confusão 2x2.

    Retorna
    -------
    None
    """
    erros = tabela[~tabela["previsao_correta"]]

    if len(erros) > 0:
        print(f"\nErros ({len(erros)} planetas):")
        for _, row in erros.iterrows():
            print(f"  {row['nome_planeta']:30} | Predito={row['categoria_predita']} "
                  f"| Real={row['categoria_original']}")
    else:
        print("\nNenhum erro no conjunto de teste.")

    vn, fp = matriz[0, 0], matriz[0, 1]
    fn, vp = matriz[1, 0], matriz[1, 1]
    print(f"\nVP={vp} | VN={vn} | FP={fp} | FN={fn}")


def salvar_resultados_completos(tabela, acuracia_final, caminho_saida):
    """
    5.9 - Exportação dos resultados detalhados em CSV.

    Salva um CSV contendo nome do planeta, categoria real, categoria
    predita e se a previsão foi correta.

    Parâmetros
    ----------
    tabela : pandas.DataFrame
        Tabela retornada por :func:`captura_nomes_exoplanetas`.
    acuracia_final : float
        Acurácia obtida no conjunto de teste final (para reporte).
    caminho_saida : str
        Caminho do arquivo CSV a salvar.

    Retorna
    -------
    tabela : pandas.DataFrame
        A mesma tabela recebida (para encadeamento).
    """
    Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
    tabela[["nome_planeta", "categoria_original", "categoria_predita",
            "previsao_correta"]].to_csv(caminho_saida, index=False)
    print(f"\nSalvo: {caminho_saida}")
    print(f"Acurácia final em dados nunca vistos: {acuracia_final:.1%}")
    return tabela


def preparar_features(df):
    """
    Remove colunas não-numéricas ou de rótulo para preparar as features do SVM.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame carregado da etapa 3.

    Retorna
    -------
    pandas.DataFrame
        DataFrame apenas com as features padronizadas para o SVM.
    """
    excluir = ["pl_name", "cluster_hc", "PC1", "PC2", "label_lp"]
    return df[[c for c in df.columns if c not in excluir]]


def executar_svm(df_features, df_original, caminho_saida_csv,
                 save_dir=None, show=False):
    """
    5.10 - Pipeline completo do SVM.

    Encadeia todas as sub-etapas do módulo:
    :func:`separacao_de_dados` → :func:`treinamento_modelo_svm` →
    :func:`avaliacao_modelo` (ajuste) → :func:`avaliacao_modelo` (teste) →
    :func:`captura_nomes_exoplanetas` → :func:`analise_detalhada_erros` →
    :func:`salvar_resultados_completos`.

    Parâmetros
    ----------
    df_features : pandas.DataFrame
        DataFrame apenas com as features numéricas do SVM.
    df_original : pandas.DataFrame
        DataFrame completo com ``pl_name`` e ``label_lp``.
    caminho_saida_csv : str
        Caminho do CSV de resultados finais.
    save_dir : str, opcional
        Diretório para salvar as matrizes de confusão.
    show : bool, opcional
        Se ``True``, exibe interativamente.

    Retorna
    -------
    modelo : sklearn.svm.SVC
        Modelo treinado.
    tabela : pandas.DataFrame
        Resultados detalhados do conjunto de teste.
    acuracia_final : float
        Acurácia no conjunto de teste.
    """
    if "label_lp" not in df_original.columns:
        raise ValueError("DataFrame precisa conter a coluna 'label_lp'.")

    rotulos = df_original["label_lp"].values

    X_train, X_adj, X_test, y_train, y_adj, y_test, idx_test = separacao_de_dados(
        df_features.values, rotulos
    )
    modelo = treinamento_modelo_svm(X_train, y_train)

    avaliacao_modelo(modelo, X_adj, y_adj, "Ajuste",
                     save_dir=save_dir, show=show)
    prev_test, matriz_test, acc_final = avaliacao_modelo(
        modelo, X_test, y_test, "Teste Final",
        save_dir=save_dir, show=show
    )

    tabela = captura_nomes_exoplanetas(df_original, idx_test, y_test, prev_test)
    analise_detalhada_erros(tabela, matriz_test)
    salvar_resultados_completos(tabela, acc_final, caminho_saida_csv)

    return modelo, tabela, acc_final


def main(caminho_entrada, caminho_saida_csv,
         save_dir=None, show=False):
    """
    Executa toda a etapa 4 (SVM) do pipeline.

    Trata graciosamente o caso em que o Label Propagation produziu uma
    distribuição degenerada de rótulos (uma das classes com < 2 amostras),
    retornando ``{"acuracia": None, "erro": ...}`` em vez de crashar.

    Parâmetros
    ----------
    caminho_entrada : str
        Caminho do Excel produzido pela etapa 3.
    caminho_saida_csv : str
        Caminho do CSV de resultados finais.
    save_dir : str, opcional
        Diretório para salvar figuras.
    show : bool, opcional
        Se ``True``, exibe figuras interativamente.

    Retorna
    -------
    dict
        Chaves ``modelo``, ``tabela``, ``acuracia`` (ou ``None`` + ``erro``
        se degenerado).
    """
    print("=" * 60)
    print("ETAPA 4 — SUPPORT VECTOR MACHINE")
    print("=" * 60)

    df = pd.read_excel(caminho_entrada, index_col=0)
    print(f"Dataset carregado: {df.shape}")

    df_features = preparar_features(df)

    try:
        modelo, tabela, acc = executar_svm(
            df_features, df, caminho_saida_csv,
            save_dir=save_dir, show=show
        )
    except ValueError as e:
        print(f"\n[SVM SALTADO] {e}")
        print("Isso indica que o pipeline anterior produziu uma distribuição "
              "de rótulos que não permite classificação supervisionada.")
        return {"modelo": None, "tabela": None, "acuracia": None,
                "erro": str(e)}

    return {"modelo": modelo, "tabela": tabela, "acuracia": acc}


if __name__ == "__main__":
    main(
        caminho_entrada="../outputs/DFsvm.xlsx",
        caminho_saida_csv="../outputs/resultados_svm.csv",
        save_dir="../outputs/figs",
        show=False,
    )
