import streamlit as st
import pandas as pd
import numpy as np
import requests
import urllib.parse
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid, GridOptionsBuilder
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="SeriesEcon Dashboard 📊",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
API_BASE = "https://apis.datos.gob.ar/series/api/"
META_URL = f"{API_BASE}dump/series-tiempo-metadatos.csv"
ID_CPI = "145.3_INGNACNAL_DICI_M_15"
SAFE_CHARS = ",:._-"

PROVINCIAS = [
    "Buenos Aires", "Catamarca", "Chaco", "Chubut", "Córdoba", "Corrientes",
    "Entre Ríos", "Formosa", "Jujuy", "La Pampa", "La Rioja", "Mendoza", "Misiones",
    "Neuquén", "Río Negro", "Salta", "San Juan", "San Luis", "Santa Cruz", "Santa Fe",
    "Santiago Del Estero", "Tierra Del Fuego", "Tucumán", "Ciudad Autónoma De Buenos Aires"
]

# Paleta de colores mejorada
COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
    '#DDA0DD', '#FFB347', '#87CEEB', '#98D8C8', '#F7DC6F'
]

##############################################################################
# Funciones de utilidad y cache
##############################################################################

@st.cache_data(show_spinner=False)
def load_metadata():
    """Carga y procesa metadatos de series económicas"""
    try:
        meta = pd.read_csv(META_URL)
        col_tit = next(c for c in meta.columns if re.search(r"titulo", c, flags=re.I))
        meta = meta.rename(columns={col_tit: "titulo"})

        # Simplificar nombres ruidosos
        meta["titulo_simple"] = (meta["titulo"]
            .str.replace(r"[_\-]+", " ", regex=True)
            .str.replace(r"pshhogurb ninxs", "tasa desocupacion", flags=re.I, regex=True)
            .str.replace(r"\b\d+\b", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.title().str.strip())

        freq_map = {"R/P1D":"Diaria","R/P1M":"Mensual","R/P3M":"Trimestral",
                    "R/P6M":"Semestral","R/P1Y":"Anual"}
        meta["frecuencia"] = meta["indice_tiempo_frecuencia"].map(freq_map)\
                                    .fillna(meta["indice_tiempo_frecuencia"])
        meta["ultimo_dato"] = pd.to_datetime(meta["serie_indice_final"], errors="coerce")
        meta = meta[meta["ultimo_dato"] >= "2022-01-01"]

        def ambito(r):
            txt = f"{r['titulo']} {r.get('dataset_titulo','')}".lower()
            for p in PROVINCIAS:
                if p.lower() in txt:
                    return p
            return "Nacional"

        meta["ambito"] = meta.apply(ambito, axis=1)
        
        # Agregar categorización por temas económicos
        meta["categoria_economica"] = meta.apply(categorizar_serie, axis=1)
        
        return meta
    except Exception as e:
        st.error(f"Error al cargar metadatos: {e}")
        return pd.DataFrame()

def categorizar_serie(row):
    """Categoriza series por tema económico"""
    titulo = row['titulo'].lower()
    dataset = str(row.get('dataset_titulo', '')).lower()
    texto = f"{titulo} {dataset}"
    
    if any(word in texto for word in ['ipc', 'inflacion', 'precios']):
        return "Inflación y Precios"
    elif any(word in texto for word in ['pib', 'producto', 'bruto']):
        return "Producto Bruto"
    elif any(word in texto for word in ['empleo', 'desocupacion', 'trabajo']):
        return "Mercado Laboral"
    elif any(word in texto for word in ['comercio', 'exportacion', 'importacion']):
        return "Comercio Exterior"
    elif any(word in texto for word in ['fiscal', 'gasto', 'ingresos', 'deficit']):
        return "Sector Fiscal"
    elif any(word in texto for word in ['monetario', 'credito', 'depositos']):
        return "Sector Monetario"
    else:
        return "Otros"

@st.cache_data(show_spinner="Descargando series…")
def fetch_single_series(sid, start="1900-01-01"):
    """Descarga una serie individual"""
    try:
        frames, offset, prev_first = [], 0, None
        while True:
            qp = {"ids": sid, "start_date": start, "format": "json",
                  "limit": 5000, "offset": offset}
            url = API_BASE + "series?" + urllib.parse.urlencode(qp, safe=SAFE_CHARS)
            response = requests.get(url, timeout=40)
            response.raise_for_status()
            js = response.json()
            data = js.get("data", [])
            if not data or (prev_first and data[0][0] == prev_first):
                break
            prev_first = data[0][0] if data else None
            blk = pd.DataFrame(data, columns=["Fecha", sid])
            blk["Fecha"] = pd.to_datetime(blk["Fecha"])
            frames.append(blk.set_index("Fecha"))
            if len(data) < 5000:
                break
            offset += 5000
        return pd.concat(frames).sort_index() if frames else pd.DataFrame()
    except Exception as e:
        st.warning(f"Error descargando serie {sid}: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_series(ids, start="1900-01-01"):
    """Descarga múltiples series"""
    dfs = [fetch_single_series(s, start) for s in ids]
    valid_dfs = [df for df in dfs if not df.empty]
    return pd.concat(valid_dfs, axis=1).sort_index() if valid_dfs else pd.DataFrame()

def calculate_statistics(df):
    """Calcula estadísticas descriptivas avanzadas"""
    stats_dict = {}
    for col in df.columns:
        series = df[col].dropna()
        if len(series) > 0:
            stats_dict[col] = {
                'Media': series.mean(),
                'Mediana': series.median(),
                'Desv. Estándar': series.std(),
                'Mínimo': series.min(),
                'Máximo': series.max(),
                'CV (%)': (series.std() / series.mean() * 100) if series.mean() != 0 else 0,
                'Asimetría': stats.skew(series),
                'Curtosis': stats.kurtosis(series),
                'Var. Anual (%)': calculate_annual_change(series)
            }
    return pd.DataFrame(stats_dict).T

def calculate_annual_change(series):
    """Calcula variación anual promedio"""
    try:
        if len(series) < 12:
            return np.nan
        annual_changes = []
        for i in range(12, len(series)):
            change = ((series.iloc[i] / series.iloc[i-12]) - 1) * 100
            annual_changes.append(change)
        return np.mean(annual_changes) if annual_changes else np.nan
    except:
        return np.nan

def perform_correlation_analysis(df):
    """Realiza análisis de correlación"""
    return df.corr()

def decompose_series(series, model='additive', period=12):
    """Descompone una serie temporal"""
    try:
        series_clean = series.dropna()
        if len(series_clean) < 24:
            return None
        return seasonal_decompose(series_clean, model=model, period=period)
    except:
        return None

def forecast_series(series, periods=12, method='exponential'):
    """Realiza pronósticos de series temporales"""
    try:
        series_clean = series.dropna()
        if len(series_clean) < 24:
            return None, None
        
        if method == 'exponential':
            model = ExponentialSmoothing(series_clean, trend='add', seasonal='add', seasonal_periods=12)
            fitted_model = model.fit()
            forecast = fitted_model.forecast(periods)
            confidence_int = fitted_model.get_prediction(start=len(series_clean), 
                                                       end=len(series_clean)+periods-1).conf_int()
        else:
            model = sm.tsa.ARIMA(series_clean, order=(1,1,1))
            fitted_model = model.fit()
            forecast_result = fitted_model.get_forecast(periods)
            forecast = forecast_result.predicted_mean
            confidence_int = forecast_result.conf_int()
            
        return forecast, confidence_int
    except:
        return None, None

##############################################################################
# Interfaz principal
##############################################################################

# Header principal
st.markdown("""
<div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
            border-radius: 10px; margin-bottom: 2rem;'>
    <h1 style='color: white; margin: 0;'>📊 SeriesEcon Dashboard</h1>
    <p style='color: white; margin: 0; opacity: 0.9;'>
        Análisis avanzado de series económicas de Argentina
    </p>
</div>
""", unsafe_allow_html=True)

# Carga de metadatos
meta = load_metadata()

if meta.empty:
    st.error("No se pudieron cargar los metadatos. Verifica tu conexión a internet.")
    st.stop()

# Navegación principal
selected_page = option_menu(
    menu_title=None,
    options=["🏠 Explorar Series", "📈 Análisis Avanzado", "🔍 Correlaciones", "📊 Dashboard Ejecutivo"],
    icons=["house", "graph-up", "search", "speedometer2"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "#667eea", "font-size": "18px"}, 
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#667eea"},
    }
)

##############################################################################
# Sidebar de filtros
##############################################################################

with st.sidebar:
    st.markdown("### 🎛️ Filtros de Series")
    
    # Filtros básicos
    amb_sel = st.selectbox("🌍 Ámbito Geográfico", ["Todos","Nacional"] + PROVINCIAS)
    meta_sel = meta if amb_sel == "Todos" else meta[meta["ambito"] == amb_sel]
    
    cat_econ = st.selectbox("📈 Categoría Económica", 
                           ["Todas"] + sorted(meta_sel["categoria_economica"].unique()))
    meta_sel = meta_sel if cat_econ == "Todas" else meta_sel[meta_sel["categoria_economica"] == cat_econ]
    
    tema = st.selectbox("📂 Tema", ["Todos"] + sorted(meta_sel["dataset_tema"].dropna().unique()))
    meta_sel = meta_sel if tema == "Todos" else meta_sel[meta_sel["dataset_tema"] == tema]

    org = st.selectbox("🏛️ Organismo", ["Todos"] + sorted(meta_sel["dataset_fuente"].dropna().unique()))
    meta_sel = meta_sel if org == "Todos" else meta_sel[meta_sel["dataset_fuente"] == org]

    freqs = sorted(meta_sel["frecuencia"].unique())
    sel_freq = st.multiselect("⏰ Frecuencia", freqs, default=freqs)
    meta_sel = meta_sel[meta_sel["frecuencia"].isin(sel_freq)]

    # Filtro de fecha
    st.markdown("### 📅 Rango de Fechas")
    fecha_desde = st.date_input("Desde", value=datetime(2020, 1, 1))
    fecha_hasta = st.date_input("Hasta", value=datetime.now())
    
    # Búsqueda avanzada
    st.markdown("### 🔍 Búsqueda")
    query = st.text_input("Buscar en títulos")
    if query:
        mask = meta_sel["titulo"].str.contains(query, case=False, na=False) | \
               meta_sel.get("serie_descripcion", "").str.contains(query, case=False, na=False)
        meta_sel = meta_sel[mask]

    # Información de series disponibles
    series_dict = dict(zip(meta_sel["titulo_simple"], meta_sel["serie_id"]))
    st.info(f"📊 Series encontradas: **{len(series_dict)}**")
    
    # Selector de series
    sel_titles = st.multiselect("📋 Seleccionar series", 
                               list(series_dict), 
                               help="Selecciona hasta 10 series para análisis")
    sel_ids = [series_dict[t] for t in sel_titles]
    
    # Configuraciones adicionales
    st.markdown("### ⚙️ Configuración")
    escala = st.radio("💰 Escala de valores", ["Nominal", "Reales (IPC dic‑2016=100)"])
    show_missing = st.checkbox("📝 Mostrar datos faltantes", value=False)

# Validación de series seleccionadas
if not sel_ids and selected_page != "📊 Dashboard Ejecutivo":
    st.info("👆 Selecciona al menos una serie en el panel lateral para comenzar el análisis.")
    st.stop()

# Descarga y procesamiento de datos
if sel_ids:
    aux_ids = [ID_CPI] if escala.startswith("Reales") else []
    
    with st.spinner("🔄 Descargando datos..."):
        df = fetch_series(sel_ids + aux_ids, start=fecha_desde.strftime("%Y-%m-%d"))
    
    if df.empty:
        st.error("❌ No se pudieron obtener datos. Verifica tu conexión y las series seleccionadas.")
        st.stop()
    
    # Filtrar por rango de fechas
    df = df[(df.index >= pd.to_datetime(fecha_desde)) & (df.index <= pd.to_datetime(fecha_hasta))]
    
    # Ajuste por inflación si es necesario
    if escala.startswith("Reales") and ID_CPI in df.columns:
        cpi = df[ID_CPI].resample("M").last().ffill()/100
        df = df.drop(columns=[ID_CPI]).resample("M").last()
        df = df.divide(cpi, axis=0)
    
    # Renombrar columnas para mejor presentación
    df_disp = df.rename(columns={v: k for k, v in series_dict.items() if v in df.columns})

##############################################################################
# Páginas del Dashboard
##############################################################################

if selected_page == "🏠 Explorar Series":
    st.markdown("## 📊 Exploración de Series Temporales")
    
    if 'df_disp' in locals() and not df_disp.empty:
        # Métricas resumen
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Series Seleccionadas", len(df_disp.columns))
        with col2:
            st.metric("Periodo", f"{df_disp.index.min().strftime('%Y-%m')} - {df_disp.index.max().strftime('%Y-%m')}")
        with col3:
            st.metric("Observaciones", len(df_disp))
        with col4:
            st.metric("Datos Completos", f"{(1-df_disp.isnull().sum().sum()/(len(df_disp)*len(df_disp.columns))):.1%}")
        
        # Opciones de visualización
        col1, col2 = st.columns([3, 1])
        with col2:
            view_type = st.selectbox("Tipo de Vista", ["Gráfico Interactivo", "Tabla de Datos", "Estadísticas"])
            chart_type = st.selectbox("Tipo de Gráfico", ["Líneas", "Área"])
            normalize = st.checkbox("Normalizar (Base 100)")
        
        with col1:
            if view_type == "Gráfico Interactivo":
                fig = go.Figure()
                
                df_plot = df_disp.copy()
                if normalize:
                    df_plot = df_plot.div(df_plot.iloc[0]) * 100
                
                for i, col in enumerate(df_plot.columns):
                    if chart_type == "Líneas":
                        fig.add_trace(go.Scatter(
                            x=df_plot.index, y=df_plot[col], name=col,
                            line=dict(color=COLORS[i % len(COLORS)], width=2)
                        ))
                    elif chart_type == "Área":
                        fig.add_trace(go.Scatter(
                            x=df_plot.index, y=df_plot[col], name=col,
                            fill='tonexty' if i > 0 else 'tozeroy',
                            line=dict(color=COLORS[i % len(COLORS)])
                        ))
                
                fig.update_layout(
                    title=f"Series Temporales - {escala}",
                    xaxis_title="Fecha",
                    yaxis_title="Valor" + (" (Base 100)" if normalize else ""),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif view_type == "Tabla de Datos":
                st.dataframe(df_disp.round(2), use_container_width=True)
                
            elif view_type == "Estadísticas":
                stats_df = calculate_statistics(df_disp)
                st.dataframe(stats_df.round(2), use_container_width=True)
        
        # Botones de descarga
        st.markdown("### 📥 Descargas")
        col1, col2 = st.columns(2)
        with col1:
            csv = df_disp.to_csv().encode('utf-8')
            st.download_button("📊 Descargar CSV", csv, "series_data.csv", "text/csv")
        with col2:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df_disp.to_excel(writer, sheet_name='Datos')
            st.download_button("📊 Descargar Excel", excel_buffer.getvalue(), "series_data.xlsx", "application/vnd.ms-excel")

elif selected_page == "📈 Análisis Avanzado":
    st.markdown("## 🔬 Análisis Estadístico Avanzado")
    
    if 'df_disp' in locals() and not df_disp.empty:
        selected_series = st.selectbox("Selecciona una serie para análisis detallado", df_disp.columns)
        
        if selected_series:
            serie_data = df_disp[selected_series].dropna()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 📊 Estadísticas: {selected_series}")
                stats = calculate_statistics(df_disp[[selected_series]])
                st.dataframe(stats.round(3))
                
                st.markdown("### 🔄 Descomposición Estacional")
                decomp = decompose_series(serie_data)
                if decomp is not None:
                    fig_decomp = make_subplots(rows=4, cols=1, 
                                             subplot_titles=['Original', 'Tendencia', 'Estacional', 'Residuos'])
                    
                    fig_decomp.add_trace(go.Scatter(x=decomp.observed.index, y=decomp.observed.values, 
                                                  name='Original'), row=1, col=1)
                    fig_decomp.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend.values, 
                                                  name='Tendencia'), row=2, col=1)
                    fig_decomp.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal.values, 
                                                  name='Estacional'), row=3, col=1)
                    fig_decomp.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid.values, 
                                                  name='Residuos'), row=4, col=1)
                    
                    fig_decomp.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig_decomp, use_container_width=True)
            
            with col2:
                st.markdown("### 🔮 Pronósticos")
                forecast_periods = st.slider("Periodos a pronosticar", 1, 24, 12)
                forecast_method = st.selectbox("Método", ["exponential", "arima"])
                
                forecast, conf_int = forecast_series(serie_data, forecast_periods, forecast_method)
                
                if forecast is not None:
                    fig_forecast = go.Figure()
                    
                    fig_forecast.add_trace(go.Scatter(
                        x=serie_data.index, y=serie_data.values,
                        name='Histórico', line=dict(color='blue')
                    ))
                    
                    forecast_index = pd.date_range(start=serie_data.index[-1] + pd.DateOffset(months=1), 
                                                 periods=forecast_periods, freq='MS')
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_index, y=forecast.values,
                        name='Pronóstico', line=dict(color='red', dash='dash')
                    ))
                    
                    fig_forecast.update_layout(
                        title=f"Pronóstico - {selected_series}",
                        xaxis_title="Fecha", yaxis_title="Valor"
                    )
                    st.plotly_chart(fig_forecast, use_container_width=True)

elif selected_page == "🔍 Correlaciones":
    st.markdown("## 🔗 Análisis de Correlaciones")
    
    if 'df_disp' in locals() and not df_disp.empty and len(df_disp.columns) > 1:
        corr_matrix = perform_correlation_analysis(df_disp)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                               title="Matriz de Correlación",
                               color_continuous_scale='RdBu_r')
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Correlaciones Destacadas")
            
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if not pd.isna(corr_val):
                        corr_pairs.append({
                            'Serie 1': corr_matrix.columns[i],
                            'Serie 2': corr_matrix.columns[j],
                            'Correlación': corr_val
                        })
            
            corr_df = pd.DataFrame(corr_pairs)
            if not corr_df.empty:
                corr_df = corr_df.reindex(corr_df['Correlación'].abs().sort_values(ascending=False).index)
                st.dataframe(corr_df.head(10).round(3), use_container_width=True)
    else:
        st.info("Selecciona al menos 2 series para realizar análisis de correlaciones.")

elif selected_page == "📊 Dashboard Ejecutivo":
    st.markdown("## 📊 Dashboard Ejecutivo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Series Disponibles", f"{len(meta):,}")
    with col2:
        st.metric("Organismos", len(meta['dataset_fuente'].unique()))
    with col3:
        st.metric("Provincias Cubiertas", len(meta['ambito'].unique()) - 1)
    with col4:
        st.metric("Última Actualización", meta['ultimo_dato'].max().strftime('%Y-%m-%d'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Series por Categoría Económica")
        cat_counts = meta['categoria_economica'].value_counts()
        fig_pie = px.pie(values=cat_counts.values, names=cat_counts.index, 
                        title="Distribución de Series por Categoría")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### 🌍 Series por Ámbito Geográfico")
        amb_counts = meta['ambito'].value_counts().head(10)
        fig_bar = px.bar(x=amb_counts.values, y=amb_counts.index, orientation='h',
                        title="Top 10 Ámbitos por Número de Series")
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("### 🏛️ Organismos con Más Series")
    org_counts = meta['dataset_fuente'].value_counts().head(10).reset_index()
    org_counts.columns = ['Organismo', 'Número de Series']
    st.dataframe(org_counts, use_container_width=True)

# Metadatos en expandible
if 'sel_ids' in locals() and sel_ids:
    with st.expander("📋 Metadatos de Series Seleccionadas"):
        if 'sel_titles' in locals() and sel_titles:
            meta_det = (meta_sel.set_index("titulo_simple").loc[sel_titles,
                       ["ambito", "categoria_economica", "frecuencia", "serie_indice_inicio", "serie_indice_final"]]
                       .rename(columns={"serie_indice_inicio": "Primer Dato", "serie_indice_final": "Último Dato"}))
            st.dataframe(meta_det, use_container_width=True)
