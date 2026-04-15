"""
Módulo para cálculos analíticos y métricas de negocio
Basado en el análisis original del notebook
"""
import pandas as pd

def get_financial_metrics(df):
    """Calcula métricas financieras clave (de tu notebook)"""
    
    # De tu notebook: Resumen_categoría
    resumen_categoria = df.groupby('category').agg(
        numero_clientes=('customer_id', lambda x: x.nunique()),
        cantidad_productos_vendidos=('quantity', 'sum'),
        ventas_totales=('total_amount', 'sum'),
        margen_promedio_venta=('profit_margin', 'mean')
    ).round(2)
    
    # De tu notebook: margen_de_ganancia_por_descuento
    margen_por_descuento = df.groupby('rango_de_descuento', observed=False)['profit_margin'].mean()
    
    # De tu notebook: Datos_Ventas_Temporales
    datos_ventas_temporales = df.groupby(df['order_date'].dt.to_period('M')).agg(
        ventas_totales=('total_amount', 'sum'),
        margen_ganancias=('profit_margin', 'sum'),
        gasto_envio=('shipping_cost', 'sum'),
    )
    datos_ventas_temporales.index = datos_ventas_temporales.index.astype(str)
    
    # De tu notebook: proporciones_financieras
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

def get_logistics_metrics(df):
    """Calcula métricas logísticas (de tu notebook)"""
    
    # De tu notebook: Resumen_region
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

def get_commercial_metrics(df):
    """Calcula métricas comerciales (de tu notebook)"""
    
    # De tu notebook: Resumen_productos
    resumen_productos = df.groupby('product_id').agg(
        numero_clientes=('customer_id', lambda x: x.nunique()),
        cantidad_productos_vendidos=('quantity', 'sum'),
        ventas_totales=('total_amount', 'sum')
    ).round(2)
    
    # Top 10 productos
    top_10_products = resumen_productos.sort_values('ventas_totales', ascending=False).head(10)
    
    # Métodos de pago por región (de tu notebook)
    resumen_region_metodo_pago = df.groupby(['region', 'payment_method']).size().unstack(fill_value=0)
    
    # Rentabilidad por método de pago (de tu notebook)
    rentabilidad_metodo_pago = df.groupby('payment_method')['profit_margin'].mean().sort_values(ascending=False)
    
    return {
        'resumen_productos': resumen_productos,
        'top_10_productos': top_10_products,
        'metodo_pago_region': resumen_region_metodo_pago,
        'rentabilidad_metodo_pago': rentabilidad_metodo_pago
    }