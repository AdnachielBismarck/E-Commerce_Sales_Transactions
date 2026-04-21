"""
app.py - Dashboard E-commerce (Version Formal)
Autor: Adnachiel Bismarck Avendaño Chavez

Ejecucion: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data
from config import setup_plot_style, CUSTOM_PALETTE_1, CUSTOM_PALETTE_2, CUSTOM_PALETTE_3, CUSTOM_PALETTE_4, CUSTOM_PALETTE_5
from filters import render_sidebar_filters, apply_filters, get_filter_summary
from metrics import (
    get_basic_metrics,
    get_financial_metrics,
    get_logistics_metrics,
    get_commercial_metrics,
    prepare_rfm_with_age,
    apply_kmeans_with_age,
    get_cltv_by_cluster,
    get_cluster_profiles,
    get_retention_matrix
)
from prediction import get_sales_prediction_linear, get_prediction_summary, get_available_regions_for_prediction
from plots import (
    plot_bar,
    plot_line,
    plot_pie,
    plot_stacked_bar,
    plot_heatmap,
    plot_boxplot,
    plot_prediction,
    plot_monthly_seasonality,
    plot_cluster_scatter_matrix
)

# Configuracion inicial
st.set_page_config(
    page_title="Dashboard E-commerce",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Barra colapsada al inicio
)
setup_plot_style()

# Ocultar elementos por defecto de Streamlit (NO ocultar el header)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Carga de datos
@st.cache_data
def load_and_cache_data():
    return load_data('data/ecommerce_sales_34500.csv')

data_raw = load_and_cache_data()

# Filtros globales
st.sidebar.title("Dashboard E-commerce")
st.sidebar.markdown("### Autor: Adnachiel Bismarck Avendaño Chavez")

filtros = render_sidebar_filters(data_raw)
data = apply_filters(
    data_raw,
    years=filtros['years'],
    months=filtros['months'],
    categories=filtros['categories'],
    regions=filtros['regions']
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Resumen de filtros: {get_filter_summary(filtros)}")
st.sidebar.caption(f"Registros seleccionados: {len(data):,}")

# Navegacion
seccion = st.sidebar.radio(
    "Secciones del informe",
    ["Resumen Ejecutivo", 
     "Analisis Financiero", 
     "Analisis Logistico",
     "Analisis Comercial",
     "Segmentacion de Clientes"]
)

# Metricas base
if len(data) > 0:
    basic_metrics = get_basic_metrics(data)
    financial_metrics = get_financial_metrics(data)
    logistics_metrics = get_logistics_metrics(data)
    commercial_metrics = get_commercial_metrics(data)


def add_figure_description(text):
    """Agrega una descripcion estilo pie de figura"""
    st.caption(f"Figura: {text}")


# ==================== SECCION: RESUMEN EJECUTIVO ====================
if seccion == "Resumen Ejecutivo":
    st.title("Analisis Integral de E-commerce")
    st.markdown("#### Resumen Ejecutivo")
    st.markdown("---")
    
    if len(data) == 0:
        st.warning("No hay datos con los filtros seleccionados. Pruebe con un rango mas amplio.")
    else:
        # Indicadores clave
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Ventas Totales", f"${basic_metrics['ventas_totales']:,.2f}")
        with col2:
            st.metric("Ticket Promedio", f"${basic_metrics['ticket_promedio']:.2f}")
        with col3:
            st.metric("Clientes Unicos", f"{basic_metrics['total_clientes']:,}")
        with col4:
            st.metric("Clientes Recurrentes", f"{basic_metrics['clientes_recurrentes']:,} ({basic_metrics['pct_recurrentes']:.1f}%)")
        
        st.markdown("---")
        
        # Hallazgos del resumen ejecutivo
        st.markdown("#### Hallazgos Principales")
        st.markdown("""
        El analisis integral del e-commerce revela los siguientes hallazgos clave:
        
        - **Area Financiera:** El negocio genera ventas solidas con un ticket promedio adecuado. 
          Las categorias mas rentables son Electronica, Hogar y Deportes.
        
        - **Area Logistica:** Existen variaciones regionales significativas en los tiempos de entrega. 
          La region Este presenta el tiempo promedio mas alto, representando una oportunidad de mejora.
        
        - **Area Comercial:** Las ventas muestran estacionalidad con picos en Abril y Diciembre. 
          La tarjeta de credito es el metodo de pago predominante en todas las regiones.
        
        - **Segmentacion de Clientes:** Se identificaron 4 segmentos de clientes. El segmento de alto valor 
          representa a los clientes mas rentables y debe ser priorizado en estrategias de retencion.
        """)
        
        st.markdown("---")
        
        # Prediccion
        st.subheader("Proyeccion de Ventas")
        
        regiones_validas = get_available_regions_for_prediction(data)
        opciones_prediccion = ["Ventas Totales"] + regiones_validas
        region_prediccion = st.selectbox("Seleccione region para prediccion:", opciones_prediccion)
        
        if region_prediccion == "Ventas Totales":
            pred_result = get_sales_prediction_linear(data, months_ahead=3)
            titulo_pred = "Prediccion de Ventas Totales - Proximos 3 Meses"
        else:
            pred_result = get_sales_prediction_linear(data, months_ahead=3, region=region_prediccion)
            titulo_pred = f"Prediccion de Ventas - Region {region_prediccion}"
        
        if pred_result.get('error'):
            st.warning(pred_result['error'])
        else:
            fig = plot_prediction(
                pred_result['historico'],
                pred_result['prediccion'],
                titulo_pred,
                "Ventas (USD)",
                color_hist=CUSTOM_PALETTE_1[0],
                color_pred=CUSTOM_PALETTE_1[4]
            )
            st.pyplot(fig)
            add_figure_description("Evolucion historica de ventas (linea continua) y proyeccion para los proximos 3 meses (linea discontinua). "
                                  "La prediccion se realiza mediante regresion lineal sobre datos mensuales.")
            st.info(get_prediction_summary(pred_result, region_prediccion if region_prediccion != "Ventas Totales" else None))


# ==================== SECCION: ANALISIS FINANCIERO ====================
elif seccion == "Analisis Financiero":
    st.title("Analisis Financiero")
    st.markdown("#### Rentabilidad y Descuentos")
    st.markdown("---")
    
    if len(data) == 0:
        st.warning("No hay datos con los filtros seleccionados.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Ventas Totales", f"${basic_metrics['ventas_totales']:,.2f}")
        with col2:
            st.metric("Ticket Promedio", f"${basic_metrics['ticket_promedio']:.2f}")
        
        st.markdown("---")
        
        # Margen por categoria (paleta 5)
        st.subheader("Margen de Ganancia por Categoria")
        margen_cat = financial_metrics['resumen_categoria']['margen_promedio_venta'].sort_values(ascending=False)
        fig = plot_bar(margen_cat, 
                       "Margen de Ganancia Promedio por Categoria",
                       "Categoria",
                       "Margen (porcentaje)",
                       CUSTOM_PALETTE_5)
        st.pyplot(fig)
        add_figure_description("Comparativa del margen de ganancia promedio entre categorias. "
                              "Electronica, Hogar y Deportes presentan los margenes mas altos.")
        
        # Evolucion de ventas (paleta 5)
        st.subheader("Evolucion de Ventas Mensuales")
        fig = plot_line(
            financial_metrics['ventas_por_mes']['ventas_totales'],
            "Ventas Totales por Mes",
            "Ventas (USD)",
            color=CUSTOM_PALETTE_5[0]
        )
        st.pyplot(fig)
        add_figure_description("Serie temporal de ventas mensuales. Se observan fluctuaciones con picos en ciertos periodos.")
        
        # Impacto de descuentos (paleta 5)
        st.subheader("Impacto de los Descuentos en el Margen")
        fig = plot_bar(
            financial_metrics['margen_por_descuento'],
            "Margen de Beneficio por Rango de Descuento",
            "Rango de Descuento",
            "Margen (porcentaje)",
            CUSTOM_PALETTE_5
        )
        st.pyplot(fig)
        add_figure_description("Relacion inversa entre el porcentaje de descuento aplicado y el margen de beneficio. "
                              "A mayor descuento, menor es el margen obtenido.")
        
        # Proporcion gastos vs margen (paleta 5)
        st.subheader("Estructura de Costos")
        proporciones = pd.Series({
            'Gastos (Envio + Devoluciones)': financial_metrics['gastos_totales'],
            'Margen de Beneficio': financial_metrics['margen_total']
        })
        fig = plot_pie(proporciones, "Proporcion de Gastos Operativos vs Margen", CUSTOM_PALETTE_5)
        st.pyplot(fig)
        add_figure_description("Distribucion entre gastos operativos (envios y devoluciones) y margen de beneficio total. "
                              "El margen representa la mayor parte de los ingresos netos.")
        
        # Conclusion financiera
        st.markdown("---")
        st.markdown("#### Conclusion del Analisis Financiero")
        st.success("Los descuentos tienen un impacto negativo directo en el margen de beneficio. "
                  "Se recomienda gestionar estrategicamente las promociones y enfocar esfuerzos en "
                  "las categorias mas rentables: Electronica, Hogar y Deportes.")


# ==================== SECCION: ANALISIS LOGISTICO ====================
elif seccion == "Analisis Logistico":
    st.title("Analisis Logistico")
    st.markdown("#### Entregas y Devoluciones")
    st.markdown("---")
    
    if len(data) == 0:
        st.warning("No hay datos con los filtros seleccionados.")
    else:
        resumen_region = logistics_metrics['resumen_region']
        
        # Tiempo de entrega (paleta 2)
        st.subheader("Tiempo de Entrega por Region")
        fig = plot_bar(
            resumen_region['dias_entrega_promedio'],
            "Tiempo Promedio de Entrega por Region",
            "Region",
            "Dias",
            CUSTOM_PALETTE_2
        )
        st.pyplot(fig)
        add_figure_description("Variacion del tiempo de entrega promedio entre regiones. "
                              "La region Este presenta el tiempo mas elevado, lo que representa una oportunidad de mejora.")
        
        # Comparativa devueltos vs no devueltos (paleta 2)
        st.subheader("Comparativa: Articulos Devueltos vs No Devueltos")
        comparacion = pd.DataFrame({
            'Devueltos': resumen_region['dias_entrega_promedio_articulos_devueltos'],
            'No Devueltos': resumen_region['dias_entrega_promedio_articulos_no_devueltos']
        })
        fig = plot_stacked_bar(comparacion, 
                              "Tiempo de Entrega: Devueltos vs No Devueltos",
                              "Region",
                              "Dias",
                              [CUSTOM_PALETTE_2[0], CUSTOM_PALETTE_2[5]])
        st.pyplot(fig)
        add_figure_description("Comparacion de tiempos de entrega entre articulos devueltos y no devueltos. "
                              "Las diferencias son minimas, sugiriendo que el tiempo de entrega no es el principal factor de devolucion.")
        
        # Costo de envio (paleta 2)
        st.subheader("Costo de Envio por Region")
        fig = plot_bar(
            resumen_region['costo_total_envio'],
            "Costo Total de Envio por Region",
            "Region",
            "Costo (USD)",
            CUSTOM_PALETTE_2
        )
        st.pyplot(fig)
        add_figure_description("Distribucion de los costos de envio acumulados por region. "
                              "Las regiones con mayor volumen de ventas presentan mayores costos de envio.")
        
        # Ventas vs devoluciones (paleta 2 combinada con 5 para contraste)
        st.subheader("Ventas vs Devoluciones por Region")
        ventas_vs_dev = pd.DataFrame({
            'Ventas Totales': resumen_region['ventas_totales'],
            'Ventas Devolutas': resumen_region['ventas_devolutas']
        })
        fig = plot_stacked_bar(ventas_vs_dev,
                              "Ventas Totales vs Ventas Devolutas por Region",
                              "Region",
                              "Monto (USD)",
                              [CUSTOM_PALETTE_2[0], CUSTOM_PALETTE_2[8]])
        st.pyplot(fig)
        add_figure_description("Relacion entre el volumen de ventas y el monto devuelto por region. "
                              "La region Sur concentra el mayor volumen tanto de ventas como de devoluciones.")
        
        st.subheader("Resumen Logistico por Region")
        st.dataframe(resumen_region[['dias_entrega_promedio', 'cantidad_productos_devueltos', 'costo_total_envio']], 
                    use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### Conclusion del Analisis Logistico")
        st.info("La region Sur concentra el mayor volumen de ventas y tambien de devoluciones. "
                "Se sugiere revisar la calidad del servicio logistico en esta zona. "
                "La region Este requiere atencion prioritaria para reducir sus tiempos de entrega.")


# ==================== SECCION: ANALISIS COMERCIAL ====================
elif seccion == "Analisis Comercial":
    st.title("Analisis Comercial")
    st.markdown("#### Ventas y Estacionalidad")
    st.markdown("---")
    
    if len(data) == 0:
        st.warning("No hay datos con los filtros seleccionados.")
    else:
        # Ventas por categoria (paleta 3)
        st.subheader("Ventas por Categoria")
        ventas_cat = financial_metrics['resumen_categoria']['ventas_totales'].sort_values(ascending=False)
        fig = plot_bar(ventas_cat, 
                      "Ventas Totales por Categoria",
                      "Categoria",
                      "Ventas (USD)",
                      CUSTOM_PALETTE_3)
        st.pyplot(fig)
        add_figure_description("Distribucion de ventas entre categorias. Electronica, Hogar y Deportes "
                              "son las categorias que generan mayores ingresos.")
        
        # Top 10 productos (paleta 3)
        st.subheader("Top 10 Productos")
        fig = plot_bar(
            commercial_metrics['top_10_productos']['ventas_totales'],
            "Productos con Mayor Contribucion a Ventas",
            "Identificador del Producto",
            "Ventas (USD)",
            CUSTOM_PALETTE_3,
            rotate_xticks=45
        )
        st.pyplot(fig)
        add_figure_description("Listado de los 10 productos que mas ventas generan. "
                              "Estos productos representan una contribucion significativa al total de ventas.")
        
        # Estacionalidad (paleta 3)
        st.subheader("Estacionalidad de Ventas")
        fig = plot_monthly_seasonality(commercial_metrics['avg_monthly_sales'],
                                       title="Comportamiento Estacional de Ventas por Mes")
        st.pyplot(fig)
        add_figure_description("Promedio historico de ventas por mes del año. "
                              "Se identifican picos estacionales en Abril y Diciembre, "
                              "mientras que Enero y Agosto presentan menor actividad.")
        
        # Metodos de pago por region (paleta 3)
        st.subheader("Metodos de Pago por Region")
        fig = plot_stacked_bar(
            commercial_metrics['metodo_pago_region'],
            "Distribucion de Metodos de Pago por Region",
            "Region",
            "Numero de Transacciones",
            CUSTOM_PALETTE_3
        )
        st.pyplot(fig)
        add_figure_description("Preferencias de metodo de pago segmentadas por region. "
                              "La tarjeta de credito es el metodo predominante en todas las regiones.")
        
        # Rentabilidad por metodo de pago (paleta 3)
        st.subheader("Rentabilidad por Metodo de Pago")
        fig = plot_bar(
            commercial_metrics['rentabilidad_metodo_pago'],
            "Margen de Beneficio por Metodo de Pago",
            "Metodo de Pago",
            "Margen (porcentaje)",
            CUSTOM_PALETTE_3
        )
        st.pyplot(fig)
        add_figure_description("Comparativa de rentabilidad entre diferentes metodos de pago. "
                              "Las diferencias son minimas, indicando que el metodo de pago no afecta significativamente el margen.")
        
        st.markdown("---")
        st.markdown("#### Conclusion del Analisis Comercial")
        st.info("Las ventas muestran estacionalidad clara con picos en Abril y Diciembre. "
                "Se recomienda planificar inventarios y promociones segun estos patrones. "
                "La tarjeta de credito es el metodo de pago preferido por los clientes.")


# ==================== SECCION: SEGMENTACION DE CLIENTES ====================
elif seccion == "Segmentacion de Clientes":
    st.title("Segmentacion de Clientes")
    st.markdown("#### Analisis RFM y Clustering")
    st.markdown("---")
    
    if len(data) == 0:
        st.warning("No hay datos con los filtros seleccionados.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Clientes", f"{basic_metrics['total_clientes']:,}")
        with col2:
            st.metric("Clientes Recurrentes", f"{basic_metrics['clientes_recurrentes']:,}")
        with col3:
            st.metric("Tasa de Recurrencia", f"{basic_metrics['pct_recurrentes']:.1f}%")
        
        st.markdown("---")
        
        # Preparar datos RFM
        rfm = prepare_rfm_with_age(data)
        rfm_clustered, cluster_summary = apply_kmeans_with_age(rfm)
        rfm_with_cltv, cltv_summary = get_cltv_by_cluster(rfm_clustered)
        cluster_profiles = get_cluster_profiles(cluster_summary)
        
        # Distribucion de clusters (paleta 4)
        st.subheader("Distribucion de Clientes por Segmento")
        fig = plot_bar(
            cluster_summary['n_clientes'],
            "Numero de Clientes por Segmento Identificado",
            "Segmento (Cluster)",
            "Cantidad de Clientes",
            CUSTOM_PALETTE_4
        )
        st.pyplot(fig)
        add_figure_description("Distribucion de la base de clientes entre los 4 segmentos identificados mediante clustering. "
                              "Los segmentos 0 y 1 concentran la mayor parte de los clientes.")
        
        # Caracteristicas de clusters
        st.subheader("Caracteristicas de Cada Segmento")
        display_df = cluster_summary.copy()
        display_df['recencia_promedio'] = display_df['recencia_promedio'].map(lambda x: f"{x:.0f} dias")
        display_df['frecuencia_promedio'] = display_df['frecuencia_promedio'].map(lambda x: f"{x:.1f} compras")
        display_df['gasto_promedio'] = display_df['gasto_promedio'].map(lambda x: f"${x:,.2f}")
        display_df['edad_promedio'] = display_df['edad_promedio'].map(lambda x: f"{x:.0f} años")
        st.dataframe(display_df, use_container_width=True)
        add_figure_description("Tabla resumen de las metricas promedio por segmento de cliente.")
        
        # Perfiles descriptivos
        st.subheader("Perfiles de Segmentos")
        for cluster, perfil in cluster_profiles.items():
            st.write(f"**Segmento {cluster}:** {perfil}")
        
        # Matriz de dispersion (paleta 4 se usa internamente en la funcion)
        st.subheader("Visualizacion de Segmentos")
        fig = plot_cluster_scatter_matrix(rfm_clustered)
        st.pyplot(fig)
        add_figure_description("Visualizacion tridimensional de los segmentos: relacion entre recencia, "
                              "frecuencia de compra, edad y gasto del cliente.")
        
        # CLTV por cluster (paleta 4)
        st.subheader("Valor de Vida del Cliente por Segmento")
        fig = plot_boxplot(
            rfm_with_cltv,
            'cluster_ranked',
            'cltv_historico',
            "Distribucion del CLTV Historico por Segmento",
            "Segmento",
            "CLTV (USD)",
            CUSTOM_PALETTE_4
        )
        st.pyplot(fig)
        add_figure_description("Distribucion del Valor de Vida del Cliente historico para cada segmento. "
                              "El Segmento 3 (Alto Valor) presenta el CLTV mas elevado y una mayor dispersion.")
        
        # Matriz de retencion
        st.subheader("Matriz de Retencion por Cohorte")
        retention_matrix = get_retention_matrix(data)
        fig = plot_heatmap(retention_matrix, 
                          "Tasa de Retencion de Clientes por Cohorte",
                          fmt='.0%', 
                          cmap='Reds')
        st.pyplot(fig)
        add_figure_description("Matriz de retencion que muestra como evoluciona la fidelidad de los clientes "
                              "a lo largo del tiempo. La retencion disminuye progresivamente desde el 100 por ciento inicial.")
        
        st.markdown("---")
        st.markdown("#### Conclusion del Analisis de Segmentacion")
        st.success("Se identificaron 4 segmentos de clientes. El Segmento 3 (Alto Valor) es el mas rentable "
                  "con un gasto promedio superior a ${:,.2f}. Se recomienda priorizar estrategias de retencion "
                  "y fidelizacion para este segmento.".format(cluster_summary['gasto_promedio'].max()))


# ==================== FOOTER ====================
st.markdown("---")
st.markdown("**Proyecto desarrollado por Adnachiel Bismarck Avendaño Chavez** | Octubre 2025")
st.markdown("*Este dashboard presenta un analisis integral del rendimiento del e-commerce, "
           "abarcando aspectos financieros, logisticos, comerciales y de comportamiento de clientes.*")