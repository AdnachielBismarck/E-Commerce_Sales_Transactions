"""
plots.py - Visualizaciones consistentes para el dashboard
Autor: Adnachiel Bismarck Avendaño Chavez
Ubicación: src/plots.py

Centraliza todos los gráficos del dashboard:
- Barras, líneas, dispersión, heatmaps, pastel
- Estilo consistente con las paletas de colores
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from config import (
    CUSTOM_PALETTE_1, CUSTOM_PALETTE_2, CUSTOM_PALETTE_3,
    CUSTOM_PALETTE_4, CUSTOM_PALETTE_5, CUSTOM_PALETTE_6,
    DAYS_IN_SPANISH, MONTHS_IN_SPANISH, setup_plot_style
)


def plot_bar(data, title, xlabel, ylabel, color_palette=CUSTOM_PALETTE_1, rotate_xticks=0):
    """
    Gráfico de barras simple
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    data.plot(kind='bar', color=color_palette[:len(data)], ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=rotate_xticks)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_line(data, title, ylabel, color=CUSTOM_PALETTE_1[0], marker='o'):
    """
    Gráfico de línea para series temporales
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data.values, marker=marker, color=color, linewidth=2, markersize=6)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Período')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_pie(data, title, color_palette=CUSTOM_PALETTE_5):
    """
    Gráfico de pastel
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(data.values, labels=data.index, autopct='%1.1f%%', 
            startangle=140, colors=color_palette[:len(data)])
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('equal')
    plt.tight_layout()
    return fig


def plot_stacked_bar(data, title, xlabel, ylabel, color_palette=CUSTOM_PALETTE_1):
    """
    Gráfico de barras apiladas
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    data.plot(kind='bar', stacked=True, ax=ax, color=color_palette[:len(data.columns)])
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Categoría', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_scatter(x, y, hue, title, xlabel, ylabel, palette=CUSTOM_PALETTE_1):
    """
    Gráfico de dispersión con color por cluster
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(x, y, c=hue, cmap='viridis', alpha=0.6, s=50)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(scatter, label='Cluster')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_heatmap(data, title, fmt='.0%', cmap='Blues'):
    """
    Mapa de calor (para matriz de retención)
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap, ax=ax, 
                cbar_kws={'label': 'Tasa'})
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Número de Periodo')
    ax.set_ylabel('Mes de Cohorte')
    plt.tight_layout()
    return fig


def plot_boxplot(data, x, y, title, xlabel, ylabel, palette=CUSTOM_PALETTE_1):
    """
    Diagrama de caja (boxplot)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=x, y=y, data=data, palette=palette[:len(data[x].unique())], ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_prediction(historical_data, prediction_df, title, ylabel, color_hist=CUSTOM_PALETTE_5[0], color_pred=CUSTOM_PALETTE_5[4]):
    """
    Gráfico de histórico + predicción
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Datos históricos (últimos 12 meses para mejor visualización)
    hist_recent = historical_data.iloc[-12:] if len(historical_data) > 12 else historical_data
    
    # Convertir a fechas para el eje X
    x_hist = range(len(hist_recent))
    x_pred = range(len(hist_recent), len(hist_recent) + len(prediction_df))
    
    ax.plot(x_hist, hist_recent['ventas_totales'].values, 
            marker='o', label='Histórico', color=color_hist, linewidth=2)
    ax.plot(x_pred, prediction_df['Ventas_Predichas'].values, 
            marker='s', label='Predicción', color=color_pred, linewidth=2, linestyle='--')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Período (meses)')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_weekly_sales(ventas_por_dia, title="Ventas por Día de la Semana"):
    """
    Gráfico de ventas por día de semana (ordenado: Lunes a Domingo)
    """
    # Ordenar días correctamente
    dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ventas_ordenadas = ventas_por_dia.reindex(dias_orden)
    
    # Renombrar a español
    ventas_ordenadas.index = [DAYS_IN_SPANISH[d] for d in ventas_ordenadas.index]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ventas_ordenadas.plot(kind='bar', color=CUSTOM_PALETTE_3, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Día de la Semana')
    ax.set_ylabel('Ventas ($)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_monthly_seasonality(avg_monthly_sales, title="Estacionalidad de Ventas por Mes"):
    """
    Gráfico de estacionalidad mensual (promedio histórico por mes)
    """
    # Renombrar meses a español
    meses_nombres = [MONTHS_IN_SPANISH[m] for m in range(1, 13)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(1, 13), avg_monthly_sales.values, marker='o', 
            color=CUSTOM_PALETTE_3[0], linewidth=2, markersize=8)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Mes')
    ax.set_ylabel('Ventas Promedio ($)')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(meses_nombres, rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_cluster_scatter_matrix(rfm_with_clusters):
    """
    Matriz de 3 gráficos de dispersión para clusters:
    1. Recencia vs Gasto
    2. Frecuencia vs Gasto
    3. Edad vs Gasto
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    clusters = rfm_with_clusters['cluster_ranked'] if 'cluster_ranked' in rfm_with_clusters.columns else rfm_with_clusters['cluster']
    
    # Gráfico 1: Recencia vs Gasto
    scatter1 = axes[0].scatter(rfm_with_clusters['recencia'], 
                                rfm_with_clusters['gasto_cliente'], 
                                c=clusters, cmap='viridis', alpha=0.6, s=50)
    axes[0].set_title('Recencia vs Gasto', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Recencia (días)')
    axes[0].set_ylabel('Gasto Total ($)')
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico 2: Frecuencia vs Gasto
    scatter2 = axes[1].scatter(rfm_with_clusters['frecuencia_compra'], 
                                rfm_with_clusters['gasto_cliente'], 
                                c=clusters, cmap='viridis', alpha=0.6, s=50)
    axes[1].set_title('Frecuencia vs Gasto', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Frecuencia de Compra')
    axes[1].set_ylabel('Gasto Total ($)')
    axes[1].grid(True, alpha=0.3)
    
    # Gráfico 3: Edad vs Gasto
    scatter3 = axes[2].scatter(rfm_with_clusters['edad_ultima_compra'], 
                                rfm_with_clusters['gasto_cliente'], 
                                c=clusters, cmap='viridis', alpha=0.6, s=50)
    axes[2].set_title('Edad vs Gasto', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Edad (años)')
    axes[2].set_ylabel('Gasto Total ($)')
    axes[2].grid(True, alpha=0.3)
    
    # Colorbar común
    cbar = fig.colorbar(scatter1, ax=axes, orientation='vertical', aspect=40, pad=0.02)
    cbar.set_label('Cluster')
    
    plt.tight_layout()
    return fig