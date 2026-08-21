# -*- coding: utf-8 -*-
"""
Módulo Análise de Similaridade Terrestre
=========================================

Etapa complementar do pipeline. Calcula um índice contínuo (0-100%) de
similaridade de cada exoplaneta com a Terra, no espaço padronizado
(z-score), usando pesos por variável.

Inclui
------
- Cálculo da similaridade individual (planeta vs Terra)
- Aplicação em todo o dataset e ordenação dos mais similares
- Sanity check (Terra vs Terra deve retornar 100%)
- Exportação em CSV
"""

from pathlib import Path

import numpy as np
import pandas as pd


PESOS_PADRAO = {
    "pl_rade": 0.30,       # Raio
    "pl_bmasse": 0.30,     # Massa
    "pl_orbsmax": 0.20,    # Distância à estrela
    "pl_orbper": 0.10,     # Período
    "pl_orbeccen": 0.10,   # Excentricidade
}

TERRA_FISICO = {
    "pl_orbper": 365.25,
    "pl_orbsmax": 1.00,
    "pl_rade": 1.00,
    "pl_bmasse": 1.00,
    "pl_orbeccen": 0.0167,
}


def calcular_similaridade_terrestre(parametros_planeta, scaler,
                                    colunas_numericas, pesos=None):
    """
    Similaridade de um planeta com a Terra no espaço padronizado.

    Assume que, no espaço z-score, distâncias de ~2-3 já são "muito diferentes":

    - distância 0 → 100% similar (mesmo ponto)
    - distância 3 → ~0% similar (muito diferente)
    - similaridade_var = ``max(0, 100 - distancia * 33.33)`` por variável
    - similaridade final = média ponderada pelos pesos

    Parâmetros
    ----------
    parametros_planeta : dict
        Valores físicos do planeta (chaves em ``colunas_numericas``).
    scaler : sklearn.preprocessing.StandardScaler
        Scaler ajustado na etapa 1 (deve ser o MESMO usado no treino).
    colunas_numericas : list of str
        Nomes das variáveis físicas.
    pesos : dict, opcional
        Pesos por variável. Padrão :data:`PESOS_PADRAO`.

    Retorna
    -------
    float
        Similaridade de 0 a 100%.
    """
    pesos = pesos or PESOS_PADRAO

    df_terra = pd.DataFrame([TERRA_FISICO])[colunas_numericas]
    valores_terra = dict(zip(colunas_numericas, scaler.transform(df_terra)[0]))

    df_planeta = pd.DataFrame([parametros_planeta])[colunas_numericas]
    valores_planeta = dict(zip(colunas_numericas, scaler.transform(df_planeta)[0]))

    sim_total, peso_total = 0.0, 0.0
    for var, peso in pesos.items():
        if var in valores_planeta and var in valores_terra:
            distancia = abs(valores_planeta[var] - valores_terra[var])
            sim_var = max(0, 100 - (distancia * 33.33))
            sim_total += sim_var * peso
            peso_total += peso

    return min(100, max(0, sim_total / peso_total)) if peso_total > 0 else 0.0


def main(df_sem_outliers, scaler, colunas_numericas, caminho_saida_csv):
    """
    Executa a etapa 5 (similaridade com a Terra) do pipeline.

    Aplica :func:`calcular_similaridade_terrestre` a todos os planetas do
    dataset e exporta um CSV ordenado por similaridade decrescente.

    Também imprime as estatísticas gerais (média, mediana, máxima, mínima)
    e um sanity check com a própria Terra (deve retornar 100%).

    Parâmetros
    ----------
    df_sem_outliers : pandas.DataFrame
        DataFrame com valores físicos originais (contendo ``pl_name`` e
        as 5 variáveis físicas).
    scaler : sklearn.preprocessing.StandardScaler
        Scaler ajustado na etapa 1.
    colunas_numericas : list of str
        Nomes das variáveis físicas.
    caminho_saida_csv : str
        Caminho do CSV de saída.

    Retorna
    -------
    df_sim : pandas.DataFrame
        DataFrame com colunas ``nome_planeta``, ``similaridade_terra``,
        ``pl_rade``, ``pl_bmasse``, ``pl_orbsmax``.
    """
    print("=" * 60)
    print("ETAPA 5 — ANÁLISE DE SIMILARIDADE COM A TERRA")
    print("=" * 60)

    resultados = []
    for _, row in df_sem_outliers.iterrows():
        parametros = {col: row.get(col) for col in colunas_numericas}
        if all(v is not None and not pd.isna(v) for v in parametros.values()):
            sim = calcular_similaridade_terrestre(
                parametros, scaler, colunas_numericas
            )
            resultados.append({
                "nome_planeta": row["pl_name"],
                "similaridade_terra": sim,
                **{k: parametros[k] for k in ("pl_rade", "pl_bmasse", "pl_orbsmax")},
            })

    df_sim = pd.DataFrame(resultados)
    print(f"\nPlanetas analisados: {len(df_sim)}")

    print("\nTOP 10 planetas mais similares à Terra:")
    for _, row in df_sim.nlargest(10, "similaridade_terra").iterrows():
        print(f"  {row['nome_planeta']:30} | {row['similaridade_terra']:5.1f}% | "
              f"Raio: {row['pl_rade']:.2f} | Massa: {row['pl_bmasse']:.2f}")

    print("\nEstatísticas:")
    print(f"  Média:   {df_sim['similaridade_terra'].mean():.1f}%")
    print(f"  Mediana: {df_sim['similaridade_terra'].median():.1f}%")
    print(f"  Máxima:  {df_sim['similaridade_terra'].max():.1f}%")
    print(f"  Mínima:  {df_sim['similaridade_terra'].min():.1f}%")

    sim_terra = calcular_similaridade_terrestre(TERRA_FISICO, scaler, colunas_numericas)
    print(f"\n  Sanity check — Terra vs Terra: {sim_terra:.1f}% (deve ser 100%)")

    Path(caminho_saida_csv).parent.mkdir(parents=True, exist_ok=True)
    df_sim.to_csv(caminho_saida_csv, index=False)
    print(f"\nSalvo: {caminho_saida_csv}")

    return df_sim
