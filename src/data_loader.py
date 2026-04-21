"""
data_loader.py - Carga y limpieza de datos del e-commerce
Autor: Adnachiel Bismarck Avendaño Chavez
Ubicación: src/data_loader.py
"""

import pandas as pd


def load_data(file_path='data/ecommerce_sales_34500.csv'):
    """
    Carga el dataset y realiza la limpieza inicial
    Basado en el código original del notebook
    """
    # Cargar datos
    data = pd.read_csv(file_path)
    
    # Convertir fechas
    data['order_date'] = pd.to_datetime(data['order_date'])
    
    # Crear grupos de edad
    data['age_group'] = pd.cut(
        data['customer_age'], 
        bins=[0, 18, 25, 35, 50, 100],
        labels=['<18', '18-25', '25-35', '35-50', '>50']
    )
    
    # Crear rangos de descuento
    data['rango_de_descuento'] = pd.cut(
        data['discount'],
        bins=[-0.01, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.31],
        labels=['0%', '1-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%'],
        right=False
    )
    
    # Calcular venta real (precio con descuento aplicado)
    data['venta_real'] = data['price'] * data['quantity'] * (1 - data['discount'])
    
    # NUEVO: Día de la semana (para análisis comercial)
    data['day_of_week'] = data['order_date'].dt.day_name()
    
    # NUEVO: Mes y año (para filtros)
    data['year'] = data['order_date'].dt.year
    data['month_num'] = data['order_date'].dt.month
    data['month_name'] = data['order_date'].dt.month_name()
    
    # NUEVO: Mes de pedido y cohorte (para matriz de retención)
    data['mes_de_pedido'] = data['order_date'].dt.to_period('M')
    data['mes_de_cohorte'] = data.groupby('customer_id')['mes_de_pedido'].transform('min')
    
    return data


def get_basic_info(df):
    """Obtiene información básica del dataset"""
    ventas_total = (df['price'] * df['quantity'] * (1 - df['discount'])).sum()
    venta_promedio = df['total_amount'].mean()
    
    return {
        'total_ventas': ventas_total,
        'ticket_promedio': venta_promedio,
        'total_clientes': df['customer_id'].nunique(),
        'total_ordenes': df['order_id'].nunique(),
        'periodo_inicio': df['order_date'].min(),
        'periodo_fin': df['order_date'].max(),
    }