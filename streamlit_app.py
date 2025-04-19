import streamlit as st
import pandas as pd
import io

############################################
# 1. Cargar datos
############################################
@st.cache_data
def load_data(meta_path: str, valores_path: str):
    meta = pd.read_csv(meta_path)
    valores = pd.read_csv(valores_path, parse_dates=["Fecha"], index_col="Fecha")
    # detectar columna titulo
    col_tit = next(c for c in meta.columns if "titulo" in c.lower())
    meta = meta.rename(columns={col_tit: "titulo"})
    return meta, valores

# Ruta de archivos locales (ajusta si los moviste)
META_FILE = "series-tiempo-metadatos.csv"
VAL_FILE  = "datosgobar_econ_mensual.csv"   # DataFrame mensual creado en pasos previos

meta, df = load_data(META_FILE, VAL_FILE)

############################################
# 2. Sidebar: filtros
############################################
st.sidebar.header("Filtros de series")

# --- Tema / palabras clave ---
query = st.sidebar.text_input("Buscar en títulos / descripción", "")

# --- Periodicidad ---
freq_options = sorted(meta["indice_tiempo_frecuencia"].unique())
sel_freq = st.sidebar.multiselect("Seleccionar frecuencia", freq_options, default=freq_options)

# --- Dataset / organismo opcional ---
if "dataset_tema" in meta.columns:
    tema_opts = sorted(meta["dataset_tema"].dropna().unique())
    sel_tema = st.sidebar.multiselect("Filtrar por tema (dataset_tema)", tema_opts)
else:
    sel_tema = []

# Aplicar filtros a metadatos
mask = meta["indice_tiempo_frecuencia"].isin(sel_freq)
if sel_tema:
    mask &= meta["dataset_tema"].isin(sel_tema)
if query:
    mask &= meta["titulo"].str.contains(query, case=False, na=False) | \
            meta.get("serie_descripcion", "").str.contains(query, case=False, na=False)

meta_filt = meta[mask]

############################################
# 3. Selección de series
############################################
series_dict = dict(zip(meta_filt["titulo"], meta_filt["serie_id"]))

st.sidebar.write(f"Series disponibles: **{len(series_dict)}**")
sel_titles = st.sidebar.multiselect("Elegir series", list(series_dict.keys()))
sel_ids = [series_dict[t] for t in sel_titles]

############################################
# 4. Presentación
############################################
present_opt = st.radio("Cómo presentar los datos", ["Tabla", "Gráfico", "Descargar CSV"])

if not sel_ids:
    st.info("Selecciona al menos una serie en el panel lateral.")
    st.stop()

sub_df = df[sel_titles].copy()

if present_opt == "Tabla":
    st.dataframe(sub_df)

elif present_opt == "Gráfico":
    st.line_chart(sub_df)

elif present_opt == "Descargar CSV":
    buffer = io.StringIO()
    sub_df.to_csv(buffer)
    st.download_button("Descargar CSV", buffer.getvalue(), file_name="series_seleccionadas.csv", mime="text/csv")

############################################
# 5. Info adicional de las series seleccionadas
############################################
meta_sel = meta_filt.set_index("titulo").loc[sel_titles]
meta_sel = meta_sel[["indice_tiempo_frecuencia", "serie_indice_inicio", "serie_indice_final", "serie_descripcion"]]
meta_sel = meta_sel.rename(columns={
    "indice_tiempo_frecuencia": "frecuencia",
    "serie_indice_inicio": "primer_dato",
    "serie_indice_final": "ultimo_dato",
    "serie_descripcion": "descripcion"
})

with st.expander("Detalle de series seleccionadas"):
    st.dataframe(meta_sel)
