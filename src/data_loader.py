"""
Módulo para la carga y limpieza de datos del e-commerce
Basado en el análisis original de Adnachiel Bismarck Avendaño Chavez
"""
import pandas as pd

def load_data(file_path='data/ecommerce_sales_34500.csv'):
    """
    Carga el dataset y realiza la limpieza inicial
    (Basado en el código original del notebook)
    """
    # Cargar datos
    data = pd.read_csv(file_path)
    
    # Convertir fechas (de tu notebook)
    data['order_date'] = pd.to_datetime(data['order_date'])
    
    # Crear grupos de edad (de tu notebook)
    data['age_group'] = pd.cut(data['customer_age'], 
                                bins=[0, 18, 25, 35, 50, 100],
                                labels=['<18', '18-25', '25-35', '35-50', '>50'])
    
    # Crear rangos de descuento (de tu notebook)
    data['rango_de_descuento'] = pd.cut(data['discount'],
                                        bins=[-0.01, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.31],
                                        labels=['0%', '1-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%'],
                                        right=False)
    
    # Calcular ventas totales (de tu notebook)
    data['total_amount_calculated'] = data['price'] * data['quantity'] * (1 - data['discount'])
    
    return data

def get_basic_info(df):
    """Obtiene información básica del dataset (de tu notebook)"""
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