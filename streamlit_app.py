import streamlit as st
import pandas as pd
import requests, urllib.parse, re, textwrap

API_BASE = "https://apis.datos.gob.ar/series/api/"
META_URL = API_BASE + "dump/series-tiempo-metadatos.csv"

###########################################################################
# 1. UTILIDADES CACHÉ
###########################################################################
@st.cache_data(show_spinner=False)
def load_metadata():
    """Descarga una sola vez el dump de metadatos y detecta las columnas clave."""
    meta = pd.read_csv(META_URL)
    # columna título (puede ser serie_titulo, title_es, etc.)
    col_tit = next(c for c in meta.columns if re.search(r"titulo", c, flags=re.I))
    meta = meta.rename(columns={col_tit: "titulo"})
    return meta

@st.cache_data(show_spinner="Descargando series…")
def fetch_series(ids: list[str], start="1900-01-01") -> pd.DataFrame:
    """Intenta pedir todas las series en un lote; si falla prueba individualmente."""
    if not ids:
        return pd.DataFrame()

    qp = {"ids": ",".join(ids), "start_date": start, "format": "json", "limit": 50000}
    url = API_BASE + "series?" + urllib.parse.urlencode(qp, safe=",:_")
    resp = requests.get(url, timeout=60).json()

    def json_to_df(js, ids_order):
        data = js.get("data", [])
        if not data:
            return pd.DataFrame()
        cols = ["Fecha"] + ids_order
        df = pd.DataFrame(data, columns=cols)
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        return df.set_index("Fecha")

    df = json_to_df(resp, ids)
    if not df.empty:
        return df.sort_index()

    # ---- fallback por ID ----
    frames = {}
    for sid in ids:
        url_single = API_BASE + "series?" + urllib.parse.urlencode({
            "ids": sid, "start_date": start, "format": "json", "limit": 50000}, safe=",:_")
        js_single = requests.get(url_single, timeout=60).json()
        tmp = json_to_df(js_single, [sid])
        if not tmp.empty:
            frames[sid] = tmp
    if not frames:
        return pd.DataFrame()
    df_full = pd.concat(frames.values(), axis=1)
    df_full = df_full.reindex(ids, axis=1)  # mantiene orden de selección
    return df_full.sort_index()

###########################################################################
# 2. CARGA METADATOS
###########################################################################
meta = load_metadata()

# Mapear códigos de frecuencia a nombres legibles
FREQ_MAP = {
    "R/P1D": "Diaria",
    "R/P1M": "Mensual",
    "R/P3M": "Trimestral",
    "R/P1Y": "Anual",
}
meta["frecuencia_readable"] = meta["indice_tiempo_frecuencia"].map(FREQ_MAP).fillna(meta["indice_tiempo_frecuencia"])

###########################################################################
# 3. SIDEBAR – NAVEGACIÓN JERÁRQUICA
###########################################################################
st.sidebar.header("Filtros de series económicas")

# --- Nivel primario: tema del dataset ---
if "dataset_tema" in meta.columns:
    temas = sorted(meta["dataset_tema"].dropna().unique())
else:
    temas = []
sel_tema = st.sidebar.selectbox("Tema principal", ["Todos"] + temas)

meta_lvl1 = meta if sel_tema == "Todos" else meta[meta["dataset_tema"] == sel_tema]

# --- Nivel secundario: organismo responsable ---
if "dataset_fuente" in meta.columns:
    orgs = sorted(meta_lvl1["dataset_fuente"].dropna().unique())
else:
    orgs = []
sel_org = st.sidebar.selectbox("Organismo", ["Todos"] + orgs)
meta_lvl2 = meta_lvl1 if sel_org == "Todos" else meta_lvl1[meta_lvl1["dataset_fuente"] == sel_org]

# --- Nivel terciario: búsqueda texto libre ---
query = st.sidebar.text_input("Buscar palabra clave", "")
if query:
    mask_txt = meta_lvl2["titulo"].str.contains(query, case=False, na=False) | \
               meta_lvl2.get("serie_descripcion", "").str.contains(query, case=False, na=False)
    meta_filtered = meta_lvl2[mask_txt]
else:
    meta_filtered = meta_lvl2

# --- Filtro periodicidad ---
sel_freq = st.sidebar.multiselect("Periodicidad", sorted(meta_filtered["frecuencia_readable"].unique()),
                                  default=sorted(meta_filtered["frecuencia_readable"].unique()))
meta_filtered = meta_filtered[meta_filtered["frecuencia_readable"].isin(sel_freq)]

# Diccionario título -> id
series_dict = dict(zip(meta_filtered["titulo"], meta_filtered["serie_id"]))

st.sidebar.write(f"**{len(series_dict)}** series encontradas")
sel_titles = st.sidebar.multiselect("Elige series", list(series_dict))
sel_ids = [series_dict[t] for t in sel_titles]

if not sel_ids:
    st.info("Selecciona al menos una serie para continuar.")
    st.stop()

###########################################################################
# 4. PRESENTACIÓN
###########################################################################
present_opt = st.radio("Presentar como", ["Tabla", "Gráfico", "Descargar CSV"])

with st.spinner("Descargando datos desde la API"):
    df = fetch_series(sel_ids)
    df = df.rename(columns={v: k for k, v in series_dict.items() if v in df.columns})

if df.empty:
    st.error("No llegaron datos para las series seleccionadas.")
    st.stop()

if present_opt == "Tabla":
    st.dataframe(df)
elif present_opt == "Gráfico":
    st.line_chart(df)
else:
    st.download_button("Descargar CSV", df.to_csv().encode(), "series_seleccionadas.csv", mime="text/csv")

###########################################################################
# 5. METADATOS DETALLADOS
###########################################################################
meta_sel = meta_filtered.set_index("titulo").loc[sel_titles][[
    "frecuencia_readable", "serie_indice_inicio", "serie_indice_final",
    "serie_descripcion" if "serie_descripcion" in meta.columns else "titulo"]]
meta_sel = meta_sel.rename(columns={
    "frecuencia_readable": "frecuencia",
    "serie_indice_inicio": "primer_dato",
    "serie_indice_final": "ultimo_dato",
    "serie_descripcion": "descripcion"
})
with st.expander("Detalles de series seleccionadas"):
    st.dataframe(meta_sel)
