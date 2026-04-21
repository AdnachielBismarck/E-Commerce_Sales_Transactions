"""
metrics.py - Métricas de negocio unificadas
Autor: Adnachiel Bismarck Avendaño Chavez
Ubicación: src/metrics.py

Unifica:
- Métricas financieras (ventas, márgenes, descuentos)
- Métricas logísticas (tiempos de entrega, costos de envío)
- Métricas comerciales (top productos, métodos de pago)
- Segmentación RFM con clustering (incluyendo edad)
- CLTV histórico
- Matriz de retención de cohortes
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from operator import attrgetter


# ==================== MÉTRICAS FINANCIERAS ====================

def get_financial_metrics(df):
    """
    Calcula métricas financieras clave
    Basado en el análisis original del notebook
    """
    
    # Resumen por categoría
    resumen_categoria = df.groupby('category').agg(
        numero_clientes=('customer_id', lambda x: x.nunique()),
        cantidad_productos_vendidos=('quantity', 'sum'),
        ventas_totales=('total_amount', 'sum'),
        cantidad_productos_devueltos=('quantity', lambda x: x[df.loc[x.index, 'returned'] == 'Yes'].sum()),
        ventas_devolutas=('total_amount', lambda x: x[df.loc[x.index, 'returned'] == 'Yes'].sum()),
        margen_promedio_venta=('profit_margin', 'mean')
    ).round(2)
    
    # Margen por rango de descuento
    margen_por_descuento = df.groupby('rango_de_descuento', observed=False)['profit_margin'].mean()
    
    # Ventas temporales por mes
    datos_ventas_temporales = df.groupby(df['order_date'].dt.to_period('M')).agg(
        ventas_totales=('total_amount', 'sum'),
        margen_ganancias=('profit_margin', 'sum'),
        gasto_envio=('shipping_cost', 'sum'),
    )
    datos_ventas_temporales.index = datos_ventas_temporales.index.astype(str)
    
    # Proporciones financieras (gastos vs margen)
    costo_total_envio = df['shipping_cost'].sum()
    ventas_perdidas_devoluciones = df[df['returned'] == 'Yes']['total_amount'].sum()
    margen_total = df['profit_margin'].sum()
    gastos_totales = costo_total_envio + ventas_perdidas_devoluciones
    
    return {
        'resumen_categoria': resumen_categoria,
        'margen_por_descuento': margen_por_descuento,
        'ventas_por_mes': datos_ventas_temporales,
        'gastos_totales': gastos_totales,
        'margen_total': margen_total,
        'costo_total_envio': costo_total_envio,
        'ventas_perdidas_devoluciones': ventas_perdidas_devoluciones
    }


# ==================== MÉTRICAS LOGÍSTICAS ====================

def get_logistics_metrics(df):
    """
    Calcula métricas logísticas
    Basado en el análisis original del notebook
    """
    
    resumen_region = df.groupby('region').agg(
        numero_clientes=('customer_id', lambda x: x.nunique()),
        dias_entrega_promedio=('delivery_time_days', 'mean'),
        dias_entrega_promedio_articulos_devueltos=('delivery_time_days', lambda x: x[df.loc[x.index, 'returned'] == 'Yes'].mean()),
        dias_entrega_promedio_articulos_no_devueltos=('delivery_time_days', lambda x: x[df.loc[x.index, 'returned'] == 'No'].mean()),
        cantidad_productos_vendidos=('quantity', 'sum'),
        ventas_totales=('total_amount', 'sum'),
        cantidad_productos_devueltos=('quantity', lambda x: x[df.loc[x.index, 'returned'] == 'Yes'].sum()),
        ventas_devolutas=('total_amount', lambda x: x[df.loc[x.index, 'returned'] == 'Yes'].sum()),
        cantidad_productos_vendidos_netos=('quantity', lambda x: x[df.loc[x.index, 'returned'] == 'No'].sum()),
        ventas_netas=('total_amount', lambda x: x[df.loc[x.index, 'returned'] == 'No'].sum()),
        costo_total_envio=('shipping_cost', 'sum'),
        costo_promedio_envio=('shipping_cost', 'mean'),
        margen_promedio_venta=('profit_margin', 'mean')
    ).round(2)
    
    return {
        'resumen_region': resumen_region
    }


# ==================== MÉTRICAS COMERCIALES ====================

def get_commercial_metrics(df):
    """
    Calcula métricas comerciales
    Basado en el análisis original del notebook
    """
    
    # Resumen de productos
    resumen_productos = df.groupby('product_id').agg(
        numero_clientes=('customer_id', lambda x: x.nunique()),
        cantidad_productos_vendidos=('quantity', 'sum'),
        ventas_totales=('total_amount', 'sum')
    ).round(2)
    
    # Top 10 productos
    top_10_products = resumen_productos.sort_values('ventas_totales', ascending=False).head(10)
    
    # Métodos de pago por región
    metodo_pago_region = df.groupby(['region', 'payment_method']).size().unstack(fill_value=0)
    
    # Rentabilidad por método de pago
    rentabilidad_metodo_pago = df.groupby('payment_method')['profit_margin'].mean().sort_values(ascending=False)
    
    # Ventas por día de semana
    ventas_por_dia = df.groupby('day_of_week')['total_amount'].sum()
    
    # Ventas promedio por mes (estacionalidad)
    avg_monthly_sales = df.groupby(df['order_date'].dt.month)['total_amount'].mean()
    
    return {
        'resumen_productos': resumen_productos,
        'top_10_productos': top_10_products,
        'metodo_pago_region': metodo_pago_region,
        'rentabilidad_metodo_pago': rentabilidad_metodo_pago,
        'ventas_por_dia': ventas_por_dia,
        'avg_monthly_sales': avg_monthly_sales
    }


# ==================== SEGMENTACIÓN RFM CON CLUSTERING (incluye edad) ====================

def prepare_rfm_with_age(df):
    """
    Prepara datos RFM incluyendo edad al momento de la última compra
    Basado en el código original del notebook (que sí incluía edad)
    """
    ultima_fecha_compra = df['order_date'].max()
    
    # RFM base
    rfm = df.groupby('customer_id').agg(
        recencia=('order_date', lambda date: (ultima_fecha_compra - date.max()).days),
        frecuencia_compra=('order_id', 'nunique'),
        gasto_cliente=('total_amount', 'sum'),
        primera_fecha_compra=('order_date', 'min'),
        ultima_fecha_compra=('order_date', 'max'),
        edad_consumidor=('customer_age', 'first')  # Edad al momento del registro
    ).reset_index()
    
    # Calcular edad al momento de la última compra (como en tu notebook)
    rfm['edad_ultima_compra'] = rfm['edad_consumidor'] + (
        (rfm['ultima_fecha_compra'] - rfm['primera_fecha_compra']).dt.days / 365.25
    ).round().astype(int)
    
    return rfm


def apply_kmeans_with_age(rfm, n_clusters=4, random_state=45):
    """
    Aplica K-Means clustering a los datos RFM + Edad
    Basado en el código original del notebook (4 variables: R, F, M, Edad)
    """
    # Escalar datos (4 variables como en tu notebook)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['recencia', 'frecuencia_compra', 'gasto_cliente', 'edad_ultima_compra']])
    
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Resumen de clusters
    cluster_summary = rfm.groupby('cluster').agg(
        recencia_promedio=('recencia', 'mean'),
        frecuencia_promedio=('frecuencia_compra', 'mean'),
        gasto_promedio=('gasto_cliente', 'mean'),
        edad_promedio=('edad_ultima_compra', 'mean'),
        n_clientes=('customer_id', 'count')
    ).round(2)
    
    # Ordenar por gasto (cluster más valioso primero)
    cluster_summary = cluster_summary.sort_values('gasto_promedio', ascending=False)
    
    # Reasignar números de cluster basados en el orden (opcional, para consistencia)
    cluster_ranking = {old: new for new, old in enumerate(cluster_summary.index)}
    rfm['cluster_ranked'] = rfm['cluster'].map(cluster_ranking)
    
    return rfm, cluster_summary


def get_cltv_by_cluster(rfm_with_clusters):
    """
    Calcula el CLTV histórico por cluster
    Basado en el análisis original del notebook
    """
    # CLTV histórico = gasto_cliente (del RFM)
    rfm_with_clusters['cltv_historico'] = rfm_with_clusters['gasto_cliente']
    
    # Estadísticas por cluster
    cltv_summary = rfm_with_clusters.groupby('cluster_ranked')['cltv_historico'].agg(
        media='mean',
        mediana='median',
        minimo='min',
        maximo='max',
        std='std'
    ).round(2)
    
    return rfm_with_clusters, cltv_summary


def get_cluster_profiles(cluster_summary):
    """
    Genera perfiles descriptivos para cada cluster
    Basado en el análisis original del notebook
    """
    profiles = {}
    
    for cluster in cluster_summary.index:
        gasto = cluster_summary.loc[cluster, 'gasto_promedio']
        frecuencia = cluster_summary.loc[cluster, 'frecuencia_promedio']
        recencia = cluster_summary.loc[cluster, 'recencia_promedio']
        edad = cluster_summary.loc[cluster, 'edad_promedio']
        
        # Clasificación de valor
        if gasto > cluster_summary['gasto_promedio'].mean():
            valor = "💰 Alto valor"
        else:
            valor = "💸 Bajo valor"
        
        # Clasificación de lealtad
        if frecuencia > cluster_summary['frecuencia_promedio'].mean():
            lealtad = "⭐ Alta lealtad"
        else:
            lealtad = "🔄 Baja lealtad"
        
        # Clasificación de actividad
        if recencia < cluster_summary['recencia_promedio'].mean():
            actividad = "🔥 Activo"
        else:
            actividad = "😴 Inactivo"
        
        # Rango de edad
        if edad < 30:
            edad_desc = "👶 Joven"
        elif edad < 50:
            edad_desc = "👨‍🦱 Adulto"
        else:
            edad_desc = "👴 Senior"
        
        profiles[cluster] = f"{valor} | {lealtad} | {actividad} | {edad_desc}"
    
    return profiles


# ==================== MATRIZ DE RETENCIÓN ====================

def get_retention_matrix(df):
    """
    Calcula la matriz de retención de cohortes
    Basado en el código original del notebook
    """
    # Asegurar que las columnas de cohorte existan
    if 'mes_de_pedido' not in df.columns:
        df['mes_de_pedido'] = df['order_date'].dt.to_period('M')
    if 'mes_de_cohorte' not in df.columns:
        df['mes_de_cohorte'] = df.groupby('customer_id')['mes_de_pedido'].transform('min')
    
    # Crear matriz de cohortes
    cohort_data = df.groupby(['mes_de_cohorte', 'mes_de_pedido']).agg(
        numero_consumidores=('customer_id', 'nunique')
    ).reset_index(drop=False)
    
    # Calcular número de periodo
    cohort_data['numero_de_periodo'] = (cohort_data['mes_de_pedido'] - cohort_data['mes_de_cohorte']).apply(attrgetter('n'))
    
    # Pivotar para crear matriz
    cohort_matrix = cohort_data.pivot_table(
        index='mes_de_cohorte', 
        columns='numero_de_periodo', 
        values='numero_consumidores'
    )
    
    # Calcular tasa de retención
    cohort_size = cohort_matrix.iloc[:, 0]
    retention_matrix = cohort_matrix.divide(cohort_size, axis=0)
    
    return retention_matrix


# ==================== MÉTRICAS GENERALES ====================

def get_basic_metrics(df):
    """
    Métricas básicas generales (ventas totales, ticket promedio, etc.)
    """
    ventas_total = (df['price'] * df['quantity'] * (1 - df['discount'])).sum()
    ticket_promedio = df['total_amount'].mean()
    total_clientes = df['customer_id'].nunique()
    total_ordenes = df['order_id'].nunique()
    
    # Clientes recurrentes
    customers_multiple = df.groupby('customer_id').filter(lambda x: len(x) > 1)['customer_id'].nunique()
    pct_recurrentes = (customers_multiple / total_clientes) * 100 if total_clientes > 0 else 0
    
    # Fechas
    fecha_inicio = df['order_date'].min()
    fecha_fin = df['order_date'].max()
    
    return {
        'ventas_totales': ventas_total,
        'ticket_promedio': ticket_promedio,
        'total_clientes': total_clientes,
        'total_ordenes': total_ordenes,
        'clientes_recurrentes': customers_multiple,
        'pct_recurrentes': pct_recurrentes,
        'fecha_inicio': fecha_inicio,
        'fecha_fin': fecha_fin
    }