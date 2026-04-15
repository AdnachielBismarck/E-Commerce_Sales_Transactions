"""
Módulo para crear visualizaciones consistentes
Basado en el estilo de gráficos del notebook original
"""
import pandas as pd  # ← ESTA LÍNEA FALTA
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de colores (de tu notebook)
custom_palette_1 = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']
custom_palette_2 = ['#f94144', '#f3722c', '#f8961e', '#f9844a', '#f9c74f', '#90be6d', '#43aa8b', '#4d908e', '#577590', '#277da1']
custom_palette_5 = ['#ff7b00', '#ff8800', '#ff9500', '#ffa200', '#ffaa00', '#ffb700', '#ffc300', '#ffd000', '#ffdd00', '#ffea00']

def setup_style():
    """Configura el estilo visual (de tu notebook)"""
    sns.set_style("darkgrid")
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['figure.dpi'] = 250
    sns.set_palette(custom_palette_1)

def plot_margenes(data, title, xlabel, ylabel="Margen (%)", color_palette=custom_palette_5):
    """Gráfico de barras para márgenes (estilo de tu notebook)"""
    fig, ax = plt.subplots(figsize=(10, 5))
    data.plot(kind='bar', color=color_palette[:len(data)], ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_series_temporal(data, title, ylabel, color=custom_palette_5[0]):
    """Gráfico de línea para series temporales"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data.index, data.values, marker='o', color=color, linewidth=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Mes")
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    return fig

def plot_pie_gastos_vs_margen(gastos_totales, margen_total):
    """Gráfico de pastel: Gastos vs Margen (de tu notebook)"""
    fig, ax = plt.subplots(figsize=(8, 8))
    proporciones = pd.Series({
        'Gastos (Envío + Devoluciones)': gastos_totales,
        'Margen de Beneficio Total': margen_total
    })
    ax.pie(proporciones, labels=proporciones.index, autopct='%1.1f%%', startangle=140, colors=custom_palette_5)
    ax.set_title('Proporción de Gastos y Margen de Beneficio')
    return fig