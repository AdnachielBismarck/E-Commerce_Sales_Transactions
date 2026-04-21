"""
prediction.py - Predicción de ventas con promedio móvil
Autor: Adnachiel Bismarck Avendaño Chavez
Ubicación: src/prediction.py

Implementa:
- Predicción con promedio móvil (más adecuado para datos con estacionalidad)
- Selector de región (ventas totales o por región)
- Cálculo de error (MAPE) para mostrar precisión
- Manejo de casos con datos insuficientes
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def get_sales_prediction_moving_average(df, months_ahead=3, window=3, region=None):
    """
    Predice ventas futuras usando promedio móvil simple
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con datos históricos
    months_ahead : int
        Número de meses a predecir (default: 3)
    window : int
        Ventana para el promedio móvil (default: 3 meses)
    region : str or None
        Si es None, predice ventas totales.
        Si es un nombre de región, predice ventas de esa región.
    
    Retorna:
    --------
    dict con predicción, histórico, y métricas
    """
    
    # Filtrar por región si se especifica
    if region and region != 'Todas':
        df_pred = df[df['region'] == region].copy()
        region_label = region
    else:
        df_pred = df.copy()
        region_label = "Todas"
    
    # Verificar que hay datos
    if len(df_pred) == 0:
        return {
            'prediccion': pd.DataFrame({'Mes': [], 'Ventas_Predichas': []}),
            'ultimo_promedio': 0,
            'historico': pd.DataFrame(),
            'mape': None,
            'error': f"No hay datos para la región {region}"
        }
    
    # Agrupar ventas por mes
    ventas_mensuales = df_pred.groupby(df_pred['order_date'].dt.to_period('M')).agg(
        ventas_totales=('total_amount', 'sum')
    ).reset_index()
    
    ventas_mensuales['fecha'] = ventas_mensuales['order_date'].dt.to_timestamp()
    ventas_mensuales = ventas_mensuales.sort_values('fecha')
    
    # Verificar que hay suficientes datos
    if len(ventas_mensuales) < 2:
        return {
            'prediccion': pd.DataFrame({'Mes': [], 'Ventas_Predichas': []}),
            'ultimo_promedio': ventas_mensuales['ventas_totales'].iloc[-1] if len(ventas_mensuales) > 0 else 0,
            'historico': ventas_mensuales,
            'mape': None,
            'error': f"Datos insuficientes para la región {region} (solo {len(ventas_mensuales)} meses)"
        }
    
    # Calcular promedio móvil
    ventas_mensuales['promedio_movil'] = ventas_mensuales['ventas_totales'].rolling(window=min(window, len(ventas_mensuales)), min_periods=1).mean()
    
    # Predicción: usar el último promedio móvil disponible
    ultimo_promedio = ventas_mensuales['promedio_movil'].iloc[-1]
    ultima_fecha = ventas_mensuales['fecha'].iloc[-1]
    
    # Generar fechas futuras
    fechas_futuras = []
    predicciones = []
    
    for i in range(1, months_ahead + 1):
        fecha_futura = ultima_fecha + timedelta(days=30 * i)
        fechas_futuras.append(fecha_futura)
        predicciones.append(ultimo_promedio)
    
    # Calcular MAPE (error) si hay suficientes datos
    mape = None
    if len(ventas_mensuales) > window + 1:
        reales = ventas_mensuales['ventas_totales'].iloc[-window:].values
        # Usar predicción histórica (promedio móvil del período anterior)
        historico_predicho = ventas_mensuales['promedio_movil'].shift(1).iloc[-window:].values
        # Evitar división por cero
        mask = reales != 0
        if mask.any():
            mape = np.mean(np.abs((reales[mask] - historico_predicho[mask]) / reales[mask])) * 100
    
    resultado = pd.DataFrame({
        'Mes': fechas_futuras,
        'Ventas_Predichas': predicciones
    })
    
    return {
        'prediccion': resultado,
        'ultimo_promedio': ultimo_promedio,
        'historico': ventas_mensuales,
        'mape': mape,
        'region': region_label,
        'error': None
    }


def get_sales_prediction_linear(df, months_ahead=3, region=None):
    """
    Predice ventas futuras usando regresión lineal (alternativa)
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con datos históricos
    months_ahead : int
        Número de meses a predecir (default: 3)
    region : str or None
        Si es None, predice ventas totales.
        Si es un nombre de región, predice ventas de esa región.
    
    Retorna:
    --------
    dict con predicción, modelo, y métricas
    """
    from sklearn.linear_model import LinearRegression
    
    # Filtrar por región si se especifica
    if region and region != 'Todas':
        df_pred = df[df['region'] == region].copy()
        region_label = region
    else:
        df_pred = df.copy()
        region_label = "Todas"
    
    # Verificar que hay datos
    if len(df_pred) == 0:
        return {
            'prediccion': pd.DataFrame({'Mes': [], 'Ventas_Predichas': []}),
            'modelo': None,
            'r2': None,
            'mape': None,
            'historico': pd.DataFrame(),
            'error': f"No hay datos para la región {region}"
        }
    
    # Agrupar ventas por mes
    ventas_mensuales = df_pred.groupby(df_pred['order_date'].dt.to_period('M')).agg(
        ventas_totales=('total_amount', 'sum')
    ).reset_index()
    
    ventas_mensuales['fecha'] = ventas_mensuales['order_date'].dt.to_timestamp()
    ventas_mensuales = ventas_mensuales.sort_values('fecha')
    
    # Verificar que hay suficientes datos
    if len(ventas_mensuales) < 3:
        return {
            'prediccion': pd.DataFrame({'Mes': [], 'Ventas_Predichas': []}),
            'modelo': None,
            'r2': None,
            'mape': None,
            'historico': ventas_mensuales,
            'error': f"Datos insuficientes para regresión en región {region} (solo {len(ventas_mensuales)} meses)"
        }
    
    ventas_mensuales['mes_num'] = range(len(ventas_mensuales))
    
    # Entrenar modelo
    X = ventas_mensuales[['mes_num']].values
    y = ventas_mensuales['ventas_totales'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predecir
    ultimo_mes_num = ventas_mensuales['mes_num'].max()
    future_months = np.array(range(ultimo_mes_num + 1, ultimo_mes_num + months_ahead + 1)).reshape(-1, 1)
    predicciones = model.predict(future_months)
    # Asegurar que las predicciones no sean negativas
    predicciones = np.maximum(predicciones, 0)
    
    # Generar fechas futuras
    ultima_fecha = ventas_mensuales['fecha'].iloc[-1]
    fechas_futuras = [ultima_fecha + timedelta(days=30 * i) for i in range(1, months_ahead + 1)]
    
    # Calcular R² y MAPE
    r2 = model.score(X, y)
    
    # MAPE histórico
    y_pred = model.predict(X)
    y_pred = np.maximum(y_pred, 0)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    resultado = pd.DataFrame({
        'Mes': fechas_futuras,
        'Ventas_Predichas': predicciones
    })
    
    return {
        'prediccion': resultado,
        'modelo': model,
        'r2': r2,
        'mape': mape,
        'historico': ventas_mensuales,
        'region': region_label,
        'error': None
    }


def get_prediction_summary(prediction_result, region_name=None):
    """
    Genera un resumen textual de la predicción
    """
    # Verificar si hubo error
    if prediction_result.get('error'):
        return f"⚠️ {prediction_result['error']}"
    
    prediccion = prediction_result['prediccion']
    
    if len(prediccion) == 0:
        return "⚠️ No hay suficientes datos para generar predicción"
    
    mape = prediction_result.get('mape', None)
    
    ultimo_mes = prediccion['Mes'].iloc[0].strftime('%B %Y')
    valor_predicho = prediccion['Ventas_Predichas'].iloc[0]
    
    texto = f"📈 **Predicción para {ultimo_mes}:** ${valor_predicho:,.2f}"
    
    if mape:
        texto += f"\n📊 **Precisión del modelo:** MAPE = {mape:.1f}%"
    
    if region_name and region_name != 'Todas':
        texto = f"📍 **Región {region_name}**\n" + texto
    
    return texto


def get_available_regions_for_prediction(df):
    """
    Obtiene las regiones que tienen suficientes datos para predicción
    Retorna lista de regiones con al menos 3 meses de datos
    """
    regions = df['region'].unique()
    valid_regions = []
    
    for region in regions:
        df_region = df[df['region'] == region]
        meses = df_region['order_date'].dt.to_period('M').nunique()
        if meses >= 3:
            valid_regions.append(region)
    
    return sorted(valid_regions)