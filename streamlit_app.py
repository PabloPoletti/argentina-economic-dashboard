import streamlit as st
import pandas as pd
import requests, urllib.parse, re, textwrap

aPI_BASE = "https://apis.datos.gob.ar/series/api/"
META_URL = aPI_BASE + "dump/series-tiempo-metadatos.csv"

@st.cache_data(show_spinner=False)
def load_metadata():
    meta = pd.read_csv(META_URL)
    # Detect column that contains "titulo"
    col_tit = next(c for c in meta.columns if re.search(r"titulo", c, flags=re.I))
    meta = meta.rename(columns={col_tit: "titulo"})
    return meta

@st.cache_data(show_spinner="Descargando series…")
def fetch_series(ids: list[str], start="1980-01-01") -> pd.DataFrame:
    """Request values for the given list of ids in a single call."""
    qp = {
        "ids": ",".join(ids),
        "start_date": start,
        "format": "json",
        "limit": 20000,
    }
    url = aPI_BASE + "series?" + urllib.parse.urlencode(qp, safe=",:")
    js = requests.get(url, timeout=60).json()
    data = js.get("data", [])
    if not data:
        return pd.DataFrame(index=pd.to_datetime([]))
    df = pd.DataFrame(data, columns=["Fecha"] + ids)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df.set_index("Fecha").sort_index()

# 1. Load metadata once
meta = load_metadata()

# 2. Sidebar filters --------------------------------------------------
st.sidebar.header("Filtros")
query = st.sidebar.text_input("Buscar palabra clave", "inflación")
freq_options = sorted(meta["indice_tiempo_frecuencia"].unique())
sel_freq = st.sidebar.multiselect("Periodicidad", freq_options, default=freq_options)

mask = meta["indice_tiempo_frecuencia"].isin(sel_freq)
if query:
    mask &= meta["titulo"].str.contains(query, case=False, na=False) |\
            meta.get("serie_descripcion", "").str.contains(query, case=False, na=False)

meta_filt = meta[mask]
series_dict = dict(zip(meta_filt["titulo"], meta_filt["serie_id"]))

st.sidebar.write(f"Series encontradas: **{len(series_dict)}**")
sel_titles = st.sidebar.multiselect("Elige series", list(series_dict))
sel_ids = [series_dict[t] for t in sel_titles]

if not sel_ids:
    st.info("Selecciona al menos una serie para continuar.")
    st.stop()

# 3. Presentation choice
present_opt = st.radio("Presentar como", ["Tabla", "Gráfico", "Descargar CSV"])

# 4. Fetch data on-demand -------------------------------------------
with st.spinner("Descargando datos desde la API"):
    df = fetch_series(sel_ids)
    df = df.rename(columns={v: k for k, v in series_dict.items() if v in df.columns})

if df.empty:
    st.error("No llegaron datos para las series seleccionadas.")
    st.stop()

# 5. Show output -----------------------------------------------------
if present_opt == "Tabla":
    st.dataframe(df)
elif present_opt == "Gráfico":
    st.line_chart(df)
else:  # CSV
    st.download_button("Descargar CSV", df.to_csv().encode(), "series_seleccionadas.csv", mime="text/csv")

# 6. Expandable metadata --------------------------------------------
meta_sel = meta_filt.set_index("titulo").loc[sel_titles][[
    "indice_tiempo_frecuencia", "serie_indice_inicio", "serie_indice_final", "serie_descripcion" if "serie_descripcion" in meta else "titulo"]]
meta_sel = meta_sel.rename(columns={
    "indice_tiempo_frecuencia": "frecuencia",
    "serie_indice_inicio": "primer_dato",
    "serie_indice_final": "ultimo_dato",
    "serie_descripcion": "descripcion"
})
with st.expander("Detalles de metadatos"):
    st.dataframe(meta_sel)
