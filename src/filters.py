"""
filters.py - Lógica de filtros interactivos (año, mes, categoría, región)
Autor: Adnachiel Bismarck Avendaño Chavez
Ubicación: src/filters.py
"""

import pandas as pd
import streamlit as st


def get_available_years(df):
    """Obtiene los años disponibles en los datos"""
    return sorted(df['order_date'].dt.year.unique())


def get_available_months(df):
    """Obtiene los meses disponibles (1-12)"""
    return sorted(df['order_date'].dt.month.unique())


def get_available_categories(df):
    """Obtiene las categorías disponibles"""
    return sorted(df['category'].unique())


def get_available_regions(df):
    """Obtiene las regiones disponibles"""
    return sorted(df['region'].unique())


def apply_filters(df, years=None, months=None, categories=None, regions=None):
    """
    Aplica filtros al DataFrame
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame original
    years : list or None
        Lista de años a filtrar
    months : list or None
        Lista de meses a filtrar
    categories : list or None
        Lista de categorías a filtrar (puede contener 'Todas')
    regions : list or None
        Lista de regiones a filtrar (puede contener 'Todas')
    
    Retorna:
    --------
    pd.DataFrame filtrado
    """
    filtered_df = df.copy()
    
    # Filtrar por años
    if years and len(years) > 0:
        filtered_df = filtered_df[filtered_df['order_date'].dt.year.isin(years)]
    
    # Filtrar por meses
    if months and len(months) > 0:
        filtered_df = filtered_df[filtered_df['order_date'].dt.month.isin(months)]
    
    # Filtrar por categorías (excluir 'Todas')
    if categories and len(categories) > 0 and 'Todas' not in categories:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    
    # Filtrar por regiones (excluir 'Todas')
    if regions and len(regions) > 0 and 'Todas' not in regions:
        filtered_df = filtered_df[filtered_df['region'].isin(regions)]
    
    return filtered_df


def render_sidebar_filters(df):
    """
    Renderiza los filtros en la barra lateral de Streamlit
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame original (sin filtrar)
    
    Retorna:
    --------
    dict con los valores seleccionados
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filtros Globales")
    
    # Obtener opciones disponibles
    available_years = get_available_years(df)
    available_categories = ['Todas'] + get_available_categories(df)
    available_regions = ['Todas'] + get_available_regions(df)
    
    # Nombres de meses en español
    month_names = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                   'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    
    # Filtro de años (múltiple selección)
    selected_years = st.sidebar.multiselect(
        "📅 Año(s)",
        options=available_years,
        default=available_years,
        help="Selecciona uno o más años"
    )
    
    # Filtro de meses (múltiple selección)
    available_months_num = get_available_months(df)
    selected_months_num = st.sidebar.multiselect(
        "📆 Mes(es)",
        options=available_months_num,
        format_func=lambda x: month_names[x-1],
        default=available_months_num,
        help="Selecciona uno o más meses"
    )
    
    # Filtro de categoría
    selected_categories = st.sidebar.multiselect(
        "🏷️ Categoría(s)",
        options=available_categories,
        default=['Todas'],
        help="Selecciona una o más categorías"
    )
    
    # Filtro de región
    selected_regions = st.sidebar.multiselect(
        "🗺️ Región(es)",
        options=available_regions,
        default=['Todas'],
        help="Selecciona una o más regiones"
    )
    
    # Mostrar resumen de filtros activos
    active_filters = []
    if selected_years != available_years:
        active_filters.append(f"{len(selected_years)} año(s)")
    if selected_months_num != available_months_num:
        active_filters.append(f"{len(selected_months_num)} mes(es)")
    if selected_categories != ['Todas']:
        active_filters.append(f"{len(selected_categories)} categoría(s)")
    if selected_regions != ['Todas']:
        active_filters.append(f"{len(selected_regions)} región(es)")
    
    if active_filters:
        st.sidebar.info(f"✅ Filtros activos: {', '.join(active_filters)}")
    
    return {
        'years': selected_years,
        'months': selected_months_num,
        'categories': selected_categories,
        'regions': selected_regions
    }


def get_filter_summary(filters_dict):
    """
    Genera un texto resumen de los filtros aplicados
    """
    month_names = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                   'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    
    parts = []
    if filters_dict['years']:
        years_str = ', '.join(str(y) for y in filters_dict['years'])
        parts.append(f"Años: {years_str}")
    
    if filters_dict['months']:
        months_str = ', '.join(month_names[m-1] for m in filters_dict['months'])
        parts.append(f"Meses: {months_str}")
    
    if filters_dict['categories'] and 'Todas' not in filters_dict['categories']:
        parts.append(f"Categorías: {', '.join(filters_dict['categories'])}")
    
    if filters_dict['regions'] and 'Todas' not in filters_dict['regions']:
        parts.append(f"Regiones: {', '.join(filters_dict['regions'])}")
    
    return " | ".join(parts) if parts else "Sin filtros (datos completos)"