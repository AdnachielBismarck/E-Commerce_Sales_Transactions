import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from operator import attrgetter

# ==================== CONFIGURACIÓN INICIAL ====================
st.set_page_config(page_title="Dashboard E-commerce", page_icon="📊", layout="wide")

# Configuración de gráficos (de tu notebook)
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['figure.dpi'] = 250

# Paletas de colores (de tu notebook)
custom_palette_1 = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']
custom_palette_2 = ['#f94144', '#f3722c', '#f8961e', '#f9844a', '#f9c74f', '#90be6d', '#43aa8b', '#4d908e', '#577590', '#277da1']
custom_palette_5 = ['#ff7b00', '#ff8800', '#ff9500', '#ffa200', '#ffaa00', '#ffb700', '#ffc300', '#ffd000', '#ffdd00', '#ffea00']

# Ocultar elementos por defecto de Streamlit
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ==================== CARGA Y PROCESAMIENTO DE DATOS ====================
@st.cache_data
def load_and_process_data():
    """Carga el CSV y realiza TODO el procesamiento como en tu notebook"""
    
    # Cargar datos
    data = pd.read_csv('data/ecommerce_sales_34500.csv')
    
    # Procesamiento de fechas
    data['order_date'] = pd.to_datetime(data['order_date'])
    
    # Crear grupos de edad
    data['age_group'] = pd.cut(data['customer_age'], 
                                bins=[0, 18, 25, 35, 50, 100],
                                labels=['<18', '18-25', '25-35', '35-50', '>50'])
    
    # Crear rangos de descuento
    data['rango_de_descuento'] = pd.cut(data['discount'],
                                        bins=[-0.01, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.31],
                                        labels=['0%', '1-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%'],
                                        right=False)
    
    # Calcular venta real después del descuento
    data['venta_real'] = data['price'] * data['quantity'] * (1 - data['discount'])
    
    # ========== AGREGACIONES DE TU NOTEBOOK ==========
    
    # 1. Ventas por mes
    datos_ventas_temporales = data.groupby(data['order_date'].dt.to_period('M')).agg(
        ventas_totales=('total_amount', 'sum'),
        margen_ganancias=('profit_margin', 'sum'),
        gasto_envio=('shipping_cost', 'sum'),
    )
    datos_ventas_temporales.index = datos_ventas_temporales.index.astype(str)
    
    # 2. Resumen por región (completo como en tu notebook)
    resumen_region = data.groupby('region').agg(
        numero_clientes=('customer_id', lambda x: x.nunique()),
        dias_entrega_promedio=('delivery_time_days', 'mean'),
        dias_entrega_promedio_articulos_devueltos=('delivery_time_days', lambda x: x[data.loc[x.index, 'returned'] == 'Yes'].mean()),
        dias_entrega_promedio_articulos_no_devueltos=('delivery_time_days', lambda x: x[data.loc[x.index, 'returned'] == 'No'].mean()),
        cantidad_productos_vendidos=('quantity', 'sum'),
        ventas_totales=('total_amount', 'sum'),
        cantidad_productos_devueltos=('quantity', lambda x: x[data.loc[x.index, 'returned'] == 'Yes'].sum()),
        ventas_devolutas=('total_amount', lambda x: x[data.loc[x.index, 'returned'] == 'Yes'].sum()),
        cantidad_productos_vendidos_netos=('quantity', lambda x: x[data.loc[x.index, 'returned'] == 'No'].sum()),
        ventas_netas=('total_amount', lambda x: x[data.loc[x.index, 'returned'] == 'No'].sum()),
        costo_total_envio=('shipping_cost', 'sum'),
        costo_promedio_envio=('shipping_cost', 'mean'),
        margen_promedio_venta=('profit_margin', 'mean')
    ).round(2)
    
    # 3. Resumen por categoría
    resumen_categoria = data.groupby('category').agg(
        numero_clientes=('customer_id', lambda x: x.nunique()),
        cantidad_productos_vendidos=('quantity', 'sum'),
        ventas_totales=('total_amount', 'sum'),
        cantidad_productos_devueltos=('quantity', lambda x: x[data.loc[x.index, 'returned'] == 'Yes'].sum()),
        ventas_devolutas=('total_amount', lambda x: x[data.loc[x.index, 'returned'] == 'Yes'].sum()),
        cantidad_productos_vendidos_netos=('quantity', lambda x: x[data.loc[x.index, 'returned'] == 'No'].sum()),
        ventas_netas=('total_amount', lambda x: x[data.loc[x.index, 'returned'] == 'No'].sum()),
        margen_promedio_venta=('profit_margin', 'mean')
    ).round(2)
    
    # 4. Margen por rango de descuento
    margen_por_descuento = data.groupby('rango_de_descuento', observed=False)['profit_margin'].mean().reset_index()
    
    # 5. Métodos de pago por región
    resumen_region_metodo_pago = data.groupby(['region', 'payment_method']).size().unstack(fill_value=0)
    resumen_region_metodo_pago_sorted = resumen_region_metodo_pago.sum(axis=1).sort_values(ascending=False)
    resumen_region_metodo_pago_sorted = resumen_region_metodo_pago.reindex(resumen_region_metodo_pago_sorted.index)
    
    # 6. Rentabilidad por método de pago
    rentabilidad_metodo_pago = data.groupby('payment_method')['profit_margin'].mean().sort_values(ascending=False)
    
    # 7. Top 10 productos
    resumen_productos = data.groupby('product_id').agg(
        numero_clientes=('customer_id', lambda x: x.nunique()),
        cantidad_productos_vendidos=('quantity', 'sum'),
        ventas_totales=('total_amount', 'sum')
    ).round(2)
    top_10_products = resumen_productos.sort_values('ventas_totales', ascending=False).head(10)
    
    # 8. Ventas totales y ticket promedio
    ventas_total = (data['price'] * data['quantity'] * (1 - data['discount'])).sum()
    ticket_promedio = data['total_amount'].mean()
    
    # 9. Proporciones financieras
    costo_total_envio = data['shipping_cost'].sum()
    ventas_perdidas_devoluciones = data[data['returned'] == 'Yes']['total_amount'].sum()
    margen_total = data['profit_margin'].sum()
    gastos_totales = costo_total_envio + ventas_perdidas_devoluciones
    proporciones_financieras = pd.Series({'Gastos (Envío + Devoluciones)': gastos_totales, 'Margen de Beneficio Total': margen_total})
    
    # 10. Ventas por mes promedio
    data['month'] = pd.to_datetime(data['order_date']).dt.month
    average_monthly_sales = data.groupby('month')['total_amount'].mean()
    
    # 11. Clientes recurrentes vs nuevos
    total_unique_customers = data['customer_id'].nunique()
    customers_multiple_purchases = data.groupby('customer_id').filter(lambda x: len(x) > 1)['customer_id'].nunique()
    new_customers = total_unique_customers - customers_multiple_purchases
    
    # 12. Clientes por género y edad
    clientes_por_genero_edad = data.groupby(['age_group', 'customer_gender'], observed=False)['customer_id'].nunique().unstack()
    
    # 13. Modelo K-Means (RFM)
    ultima_fecha_compra = data['order_date'].max()
    resumen_consumidores_compras = data.groupby('customer_id').agg(
        recencia=('order_date', lambda date: (ultima_fecha_compra - date.max()).days),
        frecuencia_compra=('order_id', 'nunique'),
        gasto_cliente=('total_amount', 'sum')
    ).reset_index()
    
    # Escalar y aplicar K-Means
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(resumen_consumidores_compras[['recencia', 'frecuencia_compra', 'gasto_cliente']])
    kmeans = KMeans(n_clusters=4, random_state=45, n_init=10)
    resumen_consumidores_compras['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Resumen de clusters
    cluster_summary = resumen_consumidores_compras.groupby('Cluster').agg(
        recencia_promedio=('recencia', 'mean'),
        frecuencia_promedio=('frecuencia_compra', 'mean'),
        gasto_promedio=('gasto_cliente', 'mean'),
        n_clientes=('customer_id', 'count')
    ).round(2)
    
    # 14. Matriz de retención (cohortes)
    data['mes_de_pedido'] = data['order_date'].dt.to_period('M')
    data['mes_de_cohorte'] = data.groupby('customer_id')['mes_de_pedido'].transform('min')
    
    cohort_data = data.groupby(['mes_de_cohorte', 'mes_de_pedido']).agg(
        numero_consumidores=('customer_id', 'nunique')
    ).reset_index(drop=False)
    
    cohort_data['numero_de_periodo'] = (cohort_data.mes_de_pedido - cohort_data.mes_de_cohorte).apply(attrgetter('n'))
    cohort_matrix = cohort_data.pivot_table(index='mes_de_cohorte', columns='numero_de_periodo', values='numero_consumidores')
    cohort_size = cohort_matrix.iloc[:, 0]
    retention_matrix = cohort_matrix.divide(cohort_size, axis=0)
    
    return {
        'data': data,
        'ventas_total': ventas_total,
        'ticket_promedio': ticket_promedio,
        'resumen_categoria': resumen_categoria,
        'resumen_region': resumen_region,
        'margen_por_descuento': margen_por_descuento,
        'ventas_mensuales': datos_ventas_temporales,
        'average_monthly_sales': average_monthly_sales,
        'proporciones_financieras': proporciones_financieras,
        'gastos_totales': gastos_totales,
        'margen_total': margen_total,
        'resumen_region_metodo_pago': resumen_region_metodo_pago,
        'resumen_region_metodo_pago_sorted': resumen_region_metodo_pago_sorted,
        'rentabilidad_metodo_pago': rentabilidad_metodo_pago,
        'top_10_products': top_10_products,
        'total_unique_customers': total_unique_customers,
        'customers_multiple_purchases': customers_multiple_purchases,
        'new_customers': new_customers,
        'clientes_por_genero_edad': clientes_por_genero_edad,
        'rfm_data': resumen_consumidores_compras,
        'cluster_summary': cluster_summary,
        'retention_matrix': retention_matrix
    }

# Cargar datos
resultados = load_and_process_data()

# ==================== BARRA LATERAL ====================
st.sidebar.title("📊 Navegación")
seccion = st.sidebar.radio(
    "Selecciona una sección:",
    ["🏠 Resumen Ejecutivo", 
     "💰 Análisis Financiero", 
     "🚚 Análisis Logístico",
     "📈 Análisis Comercial",
     "👥 Segmentación de Clientes"]
)

# ==================== SECCIÓN: RESUMEN EJECUTIVO ====================
if seccion == "🏠 Resumen Ejecutivo":
    st.title("📊 Dashboard de Análisis de E-commerce")
    st.markdown("### Autor: Adnachiel Bismarck Avendaño Chavez")
    st.markdown("---")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💵 Ventas Totales", f"${resultados['ventas_total']:,.2f}")
    with col2:
        st.metric("🎫 Ticket Promedio", f"${resultados['ticket_promedio']:.2f}")
    with col3:
        st.metric("👥 Clientes", f"{resultados['total_unique_customers']:,}")
    with col4:
        st.metric("🔄 Clientes Recurrentes", f"{resultados['customers_multiple_purchases']:,}")
    
    st.markdown("---")
    
    # Resumen ejecutivo (de tu notebook)
    st.header("📋 Resumen Ejecutivo")
    st.markdown(f"""
    **Área Financiera:** El negocio muestra una sólida salud financiera con un total de ventas de **${resultados['ventas_total']:,.2f}** y un ticket promedio de **${resultados['ticket_promedio']:.2f}**. 
    Las categorías de **Electrónica, Hogar y Deportes** son las más rentables, mientras que los descuentos impactan negativamente los márgenes.
    
    **Área Logística:** Los tiempos de entrega varían regionalmente, con el **Este** teniendo el promedio más alto. 
    No se observa una relación fuerte entre el tiempo de entrega y las devoluciones.
    
    **Área Comercial:** Las ventas muestran fluctuaciones estacionales, con picos hacia fin de año. 
    El Top 10 de productos contribuye con el **{(resultados['top_10_products']['ventas_totales'].sum() / resultados['ventas_total']) * 100:.2f}%** de las ventas totales.
    
    **Segmentación de Clientes:** El clustering RFM identificó 4 segmentos, siendo el **Cluster {resultados['cluster_summary'].iloc[0].name if len(resultados['cluster_summary']) > 0 else '0'}** el más valioso por su alta frecuencia, bajo recencia y alto gasto.
    """)

# ==================== SECCIÓN: ANÁLISIS FINANCIERO ====================
elif seccion == "💰 Análisis Financiero":
    st.title("💰 Análisis Financiero - Rentabilidad y Descuentos")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💵 Ventas Totales", f"${resultados['ventas_total']:,.2f}")
    with col2:
        st.metric("🎫 Ticket Promedio", f"${resultados['ticket_promedio']:.2f}")
    
    st.subheader("📊 Margen de Ganancias por Categoría")
    fig, ax = plt.subplots(figsize=(12, 6))
    resumen_categoria_sorted = resultados['resumen_categoria'].sort_values('margen_promedio_venta', ascending=False)
    resumen_categoria_sorted[['margen_promedio_venta']].plot(kind='bar', color=custom_palette_5[4:6], ax=ax)
    ax.set_title('Margen de ganancias promedio por categoría', fontsize=14, fontweight='bold')
    ax.set_xlabel('Categoría')
    ax.set_ylabel('Margen promedio (%)')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='x')
    st.pyplot(fig)
    
    st.subheader("📊 Margen de Ganancias por Región")
    fig, ax = plt.subplots(figsize=(12, 6))
    resumen_region_sorted = resultados['resumen_region'].sort_values('margen_promedio_venta', ascending=False)
    resumen_region_sorted[['margen_promedio_venta']].plot(kind='bar', stacked=True, ax=ax, color=custom_palette_5)
    ax.set_title('Margen de ganancias promedio por Región', fontsize=14, fontweight='bold')
    ax.set_xlabel('Región')
    ax.set_ylabel('Margen promedio (%)')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='x')
    st.pyplot(fig)
    
    st.subheader("📈 Evolución de Ventas Mensuales")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(resultados['ventas_mensuales'].index, resultados['ventas_mensuales']['ventas_totales'].values, 
            color=custom_palette_5[0], marker='o', linewidth=2)
    ax.set_title('Ventas Totales por Mes', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mes y Año')
    ax.set_ylabel('Ventas Totales')
    plt.xticks(rotation=45)
    ax.grid(True)
    st.pyplot(fig)
    
    st.subheader("📉 Impacto de los Descuentos en el Margen")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='rango_de_descuento', y='profit_margin', data=resultados['margen_por_descuento'], 
                palette=custom_palette_5[:7], hue='rango_de_descuento', legend=False)
    ax.set_title('Margen de Beneficio Promedio por Rango de Descuento', fontsize=14, fontweight='bold')
    ax.set_xlabel('Rango de Descuento')
    ax.set_ylabel('Margen de Beneficio Promedio')
    ax.grid(axis='y')
    st.pyplot(fig)
    
    st.subheader("📊 Proporción de Gastos vs Margen de Beneficio")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(resultados['proporciones_financieras'], labels=resultados['proporciones_financieras'].index, 
            autopct='%1.1f%%', startangle=140, colors=custom_palette_5)
    ax.set_title('Proporción de Gastos (Envío + Devoluciones) y Margen de Beneficio')
    ax.axis('equal')
    st.pyplot(fig)
    
    st.success("✅ **Conclusión:** Los descuentos tienen un impacto negativo directo en el margen de beneficio. Se recomienda gestionar estratégicamente las promociones.")

# ==================== SECCIÓN: ANÁLISIS LOGÍSTICO ====================
elif seccion == "🚚 Análisis Logístico":
    st.title("🚚 Análisis Logístico - Entregas y Devoluciones")
    st.markdown("---")
    
    st.subheader("📦 Tiempo Promedio de Entrega por Región")
    fig, ax = plt.subplots(figsize=(12, 6))
    resultados['resumen_region']['dias_entrega_promedio'].plot(kind='bar', color=custom_palette_1, ax=ax)
    ax.set_title('Tiempo Promedio de Entrega por Región', fontsize=14, fontweight='bold')
    ax.set_xlabel('Región')
    ax.set_ylabel('Días')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='x')
    st.pyplot(fig)
    
    st.subheader("📊 Comparación: Tiempo de Entrega - Devueltos vs No Devueltos")
    fig, ax = plt.subplots(figsize=(12, 6))
    comparacion_tiempos = pd.DataFrame({
        'Devueltos': resultados['resumen_region']['dias_entrega_promedio_articulos_devueltos'],
        'No Devueltos': resultados['resumen_region']['dias_entrega_promedio_articulos_no_devueltos']
    })
    comparacion_tiempos.plot(kind='bar', ax=ax, color=[custom_palette_2[0], custom_palette_1[2]])
    ax.set_title('Tiempo de Entrega: Artículos Devueltos vs No Devueltos', fontsize=14, fontweight='bold')
    ax.set_xlabel('Región')
    ax.set_ylabel('Días')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='x')
    st.pyplot(fig)
    
    st.subheader("💰 Costo de Envío por Región")
    fig, ax = plt.subplots(figsize=(12, 6))
    resultados['resumen_region']['costo_total_envio'].plot(kind='bar', color=custom_palette_2, ax=ax)
    ax.set_title('Costo Total de Envío por Región', fontsize=14, fontweight='bold')
    ax.set_xlabel('Región')
    ax.set_ylabel('Costo Total ($)')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='x')
    st.pyplot(fig)
    
    st.subheader("📊 Ventas vs Devoluciones por Región")
    fig, ax = plt.subplots(figsize=(12, 6))
    ventas_vs_devoluciones = pd.DataFrame({
        'Ventas Totales': resultados['resumen_region']['ventas_totales'],
        'Ventas Devolutas': resultados['resumen_region']['ventas_devolutas']
    })
    ventas_vs_devoluciones.plot(kind='bar', ax=ax, color=[custom_palette_5[0], custom_palette_2[0]])
    ax.set_title('Ventas Totales vs Ventas Devolutas por Región', fontsize=14, fontweight='bold')
    ax.set_xlabel('Región')
    ax.set_ylabel('Monto ($)')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='x')
    st.pyplot(fig)
    
    st.subheader("📋 Resumen Logístico por Región")
    st.dataframe(resultados['resumen_region'][[
        'dias_entrega_promedio', 
        'cantidad_productos_devueltos', 
        'costo_total_envio',
        'margen_promedio_venta'
    ]], use_container_width=True)
    
    st.info("📌 **Hallazgo:** La región Sur concentra el mayor volumen de ventas y también de devoluciones. Se sugiere revisar la calidad del servicio logístico en esta zona.")

# ==================== SECCIÓN: ANÁLISIS COMERCIAL ====================
elif seccion == "📈 Análisis Comercial":
    st.title("📈 Análisis Comercial - Ventas y Estacionalidad")
    st.markdown("---")
    
    st.subheader("🏷️ Ventas por Categoría")
    fig, ax = plt.subplots(figsize=(12, 6))
    resultados['resumen_categoria']['ventas_totales'].plot(kind='bar', color=custom_palette_1, ax=ax)
    ax.set_title('Ventas Totales por Categoría', fontsize=14, fontweight='bold')
    ax.set_xlabel('Categoría')
    ax.set_ylabel('Ventas ($)')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='x')
    st.pyplot(fig)
    
    st.subheader("📅 Estacionalidad de Ventas")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(resultados['average_monthly_sales'].index, resultados['average_monthly_sales'].values, 
            marker='o', color=custom_palette_5[0], linewidth=2, markersize=8)
    ax.set_title('Ventas Promedio por Mes', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mes')
    ax.set_ylabel('Ventas Promedio ($)')
    ax.set_xticks(range(1, 13))
    ax.grid(True)
    st.pyplot(fig)
    
    st.subheader("🏆 Top 10 Productos con Mayores Ventas")
    fig, ax = plt.subplots(figsize=(12, 6))
    resultados['top_10_products']['ventas_totales'].plot(kind='bar', color=custom_palette_2, ax=ax)
    ax.set_title('Top 10 Productos por Ventas', fontsize=14, fontweight='bold')
    ax.set_xlabel('Producto')
    ax.set_ylabel('Ventas ($)')
    plt.xticks(rotation=45)
    ax.grid(axis='x')
    st.pyplot(fig)
    
    st.info(f"📌 **Hallazgo:** El Top 10 de productos contribuye con **${resultados['top_10_products']['ventas_totales'].sum():,.2f}**, representando el **{(resultados['top_10_products']['ventas_totales'].sum() / resultados['ventas_total']) * 100:.2f}%** de las ventas totales.")
    
    st.subheader("💳 Métodos de Pago por Región")
    fig, ax = plt.subplots(figsize=(12, 6))
    resultados['resumen_region_metodo_pago_sorted'].plot(kind='bar', stacked=True, ax=ax, color=custom_palette_1[:len(resultados['resumen_region_metodo_pago_sorted'].columns)])
    ax.set_title('Distribución de Métodos de Pago por Región', fontsize=14, fontweight='bold')
    ax.set_xlabel('Región')
    ax.set_ylabel('Número de Transacciones')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Método de Pago', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
    
    st.subheader("💰 Rentabilidad Promedio por Método de Pago")
    fig, ax = plt.subplots(figsize=(10, 6))
    resultados['rentabilidad_metodo_pago'].plot(kind='bar', color=custom_palette_5, ax=ax)
    ax.set_title('Margen de Beneficio por Método de Pago', fontsize=14, fontweight='bold')
    ax.set_xlabel('Método de Pago')
    ax.set_ylabel('Margen Promedio (%)')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='x')
    st.pyplot(fig)

# ==================== SECCIÓN: SEGMENTACIÓN DE CLIENTES ====================
elif seccion == "👥 Segmentación de Clientes":
    st.title("👥 Segmentación de Clientes - Análisis RFM y Clustering")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Total Clientes", f"{resultados['total_unique_customers']:,}")
    with col2:
        st.metric("🔄 Clientes Recurrentes", f"{resultados['customers_multiple_purchases']:,}")
    with col3:
        pct_recurrentes = (resultados['customers_multiple_purchases'] / resultados['total_unique_customers']) * 100
        st.metric("📊 Tasa de Recurrencia", f"{pct_recurrentes:.1f}%")
    
    st.markdown("---")
    
    st.subheader("📊 Distribución de Clientes por Cluster")
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_counts = resultados['rfm_data']['Cluster'].value_counts().sort_index()
    cluster_counts.plot(kind='bar', color=custom_palette_1[:len(cluster_counts)], ax=ax)
    ax.set_title('Número de Clientes por Segmento (Cluster)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Cantidad de Clientes')
    for i, v in enumerate(cluster_counts.values):
        ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
    st.pyplot(fig)
    
    st.subheader("📋 Características de Cada Segmento")
    display_df = resultados['cluster_summary'].copy()
    display_df['recencia_promedio'] = display_df['recencia_promedio'].map(lambda x: f"{x:.0f} días")
    display_df['frecuencia_promedio'] = display_df['frecuencia_promedio'].map(lambda x: f"{x:.1f} compras")
    display_df['gasto_promedio'] = display_df['gasto_promedio'].map(lambda x: f"${x:,.2f}")
    st.dataframe(display_df, use_container_width=True)
    
    st.subheader("🔍 Visualización de Segmentos: Recencia vs Gasto")
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(
        resultados['rfm_data']['recencia'], 
        resultados['rfm_data']['gasto_cliente'],
        c=resultados['rfm_data']['Cluster'], 
        cmap='viridis', 
        alpha=0.6,
        s=50
    )
    ax.set_xlabel("Recencia (días desde última compra)")
    ax.set_ylabel("Gasto Total ($)")
    ax.set_title("Segmentación de Clientes: Recencia vs Gasto", fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.subheader("📊 Perfil de Clusters - Comparativa de Métricas")
    fig, ax = plt.subplots(figsize=(12, 6))
    metricas_cluster = resultados['cluster_summary'][['recencia_promedio', 'frecuencia_promedio', 'gasto_promedio']]
    metricas_norm = (metricas_cluster - metricas_cluster.min()) / (metricas_cluster.max() - metricas_cluster.min())
    metricas_norm.plot(kind='bar', ax=ax, color=custom_palette_2[:3])
    ax.set_title('Métricas Normalizadas por Cluster', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Valor Normalizado')
    ax.legend(['Recencia (menor es mejor)', 'Frecuencia', 'Gasto'])
    st.pyplot(fig)
    
    st.subheader("👥 Distribución de Clientes por Género y Edad")
    fig, ax = plt.subplots(figsize=(12, 6))
    resultados['clientes_por_genero_edad'].plot(kind='bar', ax=ax, color=custom_palette_1[:2])
    ax.set_title('Clientes por Grupo de Edad y Género', fontsize=14, fontweight='bold')
    ax.set_xlabel('Grupo de Edad')
    ax.set_ylabel('Número de Clientes')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Género')
    ax.grid(axis='x')
    st.pyplot(fig)
    
    st.subheader("📈 Matriz de Retención de Cohorte")
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(resultados['retention_matrix'], annot=True, fmt='.0%', cmap='Blues', ax=ax, cbar_kws={'label': 'Tasa de Retención'})
    ax.set_title('Matriz de Retención de Clientes por Cohorte', fontsize=14, fontweight='bold')
    ax.set_xlabel('Número de Periodo')
    ax.set_ylabel('Mes de Cohorte')
    st.pyplot(fig)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("📅 **Proyecto desarrollado por Adnachiel Bismarck Avendaño Chavez** | Octubre 2025")