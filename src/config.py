"""
config.py - Configuración global del dashboard
Autor: Adnachiel Bismarck Avendaño Chavez
Ubicación: src/config.py
"""

import matplotlib.pyplot as plt
import seaborn as sns

# ==================== RUTAS ====================
DATA_PATH = 'data/ecommerce_sales_34500.csv'
PROCESSED_DATA_PATH = 'data/processed_data.parquet'

# ==================== PALETAS DE COLORES ====================
# (Tomadas directamente de tu notebook original)
CUSTOM_PALETTE_1 = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']
CUSTOM_PALETTE_2 = ['#f94144', '#f3722c', '#f8961e', '#f9844a', '#f9c74f', '#90be6d', '#43aa8b', '#4d908e', '#577590', '#277da1']
CUSTOM_PALETTE_3 = ['#d9ed92', '#b5e48c', '#99d98c', '#76c893', '#52b69a', '#34a0a4', '#168aad', '#1a759f', '#1e6091', '#184e77']
CUSTOM_PALETTE_4 = ['#fbf8cc', '#fde4cf', '#ffcfd2', '#f1c0e8', '#cfbaf0', '#a3c4f3', '#90dbf4', '#8eecf5', '#98f5e1', '#b9fbc0']
CUSTOM_PALETTE_5 = ['#ff7b00', '#ff8800', '#ff9500', '#ffa200', '#ffaa00', '#ffb700', '#ffc300', '#ffd000', '#ffdd00', '#ffea00']
CUSTOM_PALETTE_6 = ['#f72585', '#b5179e', '#7209b7', '#560bad', '#480ca8', '#3a0ca3', '#3f37c9', '#4361ee', '#4895ef', '#4cc9f0']

# Paleta por defecto
DEFAULT_PALETTE = CUSTOM_PALETTE_1

# ==================== CONFIGURACIÓN DE GRÁFICOS ====================
def setup_plot_style():
    """Configura el estilo visual de todos los gráficos (igual que tu notebook)"""
    sns.set_style("darkgrid")
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['figure.dpi'] = 250
    sns.set_palette(DEFAULT_PALETTE)

# ==================== NOMBRES DE MESES (español) ====================
MONTHS_IN_SPANISH = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}

# ==================== NOMBRES DE DÍAS (español) ====================
DAYS_IN_SPANISH = {
    'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
    'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado',
    'Sunday': 'Domingo'
}