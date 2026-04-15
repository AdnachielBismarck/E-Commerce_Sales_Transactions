"""
Módulo para análisis de segmentación de clientes
Basado en el análisis original del notebook (RFM y K-Means)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def prepare_rfm_data(df):
    """
    Prepara los datos RFM (Recencia, Frecuencia, Monto)
    Basado en el código original del notebook
    """
    ultima_fecha_compra = df['order_date'].max()
    
    # De tu notebook: Resumen_consumidores_compras
    rfm = df.groupby('customer_id').agg(
        recencia=('order_date', lambda date: (ultima_fecha_compra - date.max()).days),
        frecuencia_compra=('order_id', 'nunique'),
        gasto_cliente=('total_amount', 'sum')
    ).reset_index()
    
    return rfm

def apply_kmeans(rfm, n_clusters=4, random_state=45):
    """
    Aplica K-Means clustering a los datos RFM
    Basado en el código original del notebook
    """
    # Escalar datos (de tu notebook)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['recencia', 'frecuencia_compra', 'gasto_cliente']])
    
    # Aplicar K-Means (de tu notebook)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Resumen de clusters (similar a tu análisis)
    cluster_summary = rfm.groupby('cluster').agg(
        recencia_promedio=('recencia', 'mean'),
        frecuencia_promedio=('frecuencia_compra', 'mean'),
        gasto_promedio=('gasto_cliente', 'mean'),
        n_clientes=('customer_id', 'count')
    ).round(2)
    
    # Ordenar por gasto
    cluster_summary = cluster_summary.sort_values('gasto_promedio', ascending=False)
    
    return rfm, cluster_summary

def get_cluster_profiles(cluster_summary):
    """Genera perfiles descriptivos para cada cluster"""
    profiles = {}
    
    for cluster in cluster_summary.index:
        gasto = cluster_summary.loc[cluster, 'gasto_promedio']
        frecuencia = cluster_summary.loc[cluster, 'frecuencia_promedio']
        recencia = cluster_summary.loc[cluster, 'recencia_promedio']
        
        if gasto > cluster_summary['gasto_promedio'].mean():
            valor = "💰 Alto valor"
        else:
            valor = "💸 Bajo valor"
            
        if frecuencia > cluster_summary['frecuencia_promedio'].mean():
            lealtad = "⭐ Alta lealtad"
        else:
            lealtad = "🔄 Baja lealtad"
            
        if recencia < cluster_summary['recencia_promedio'].mean():
            actividad = "🔥 Activo"
        else:
            actividad = "😴 Inactivo"
            
        profiles[cluster] = f"{valor} | {lealtad} | {actividad}"
    
    return profiles