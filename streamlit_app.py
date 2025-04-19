##############################################################################
# streamlit_app.py  –  abril 2025
##############################################################################
import streamlit as st
import pandas as pd, requests, urllib.parse, re, datetime as dt

API_BASE = "https://apis.datos.gob.ar/series/api/"
META_URL = f"{API_BASE}dump/series-tiempo-metadatos.csv"

PROVINCIAS = ["Buenos Aires", "Catamarca", "Chaco", "Chubut", "Córdoba",
              "Corrientes", "Entre Ríos", "Formosa", "Jujuy", "La Pampa",
              "La Rioja", "Mendoza", "Misiones", "Neuquén", "Río Negro",
              "Salta", "San Juan", "San Luis", "Santa Cruz", "Santa Fe",
              "Santiago Del Estero", "Tierra Del Fuego", "Tucumán",
              "Ciudad Autónoma De Buenos Aires"]

##############################################################################
# 1 ▸ utilidades cache
##############################################################################
@st.cache_data(show_spinner=False)
def load_metadata():
    meta = pd.read_csv(META_URL)

    # columna del título (serie_titulo o similar) → 'titulo'
    col_tit = next(c for c in meta.columns if re.search(r"titulo", c, flags=re.I))
    meta = meta.rename(columns={col_tit: "titulo"})

    # título legible
    meta["titulo_simple"] = (meta["titulo"]
                             .str.replace(r"[_\\-]+", " ", regex=True)
                             .str.replace(r"\\b\\d+\\b", "", regex=True)
                             .str.title()
                             .str.strip())

    # frecuencia legible
    freq_map = {"R/P1D": "Diaria", "R/P1M": "Mensual",
                "R/P3M": "Trimestral", "R/P6M": "Semestral",
                "R/P1Y": "Anual"}
    meta["frecuencia"] = meta["indice_tiempo_frecuencia"].map(freq_map)\
                            .fillna(meta["indice_tiempo_frecuencia"])

    # último dato como datetime
    meta["ultimo_dato"] = pd.to_datetime(meta["serie_indice_final"], errors="coerce")
    meta = meta[meta["ultimo_dato"] >= "2023-01-01"]  # solo series “vigentes”

    # ámbito: nacional vs provincia
    def detectar_ambito(tit, dataset):
        txt = f"{tit} {dataset}".lower()
        for prov in PROVINCIAS:
            if prov.lower() in txt:
                return prov
        if "nacion" in txt or "nacional" in txt:
            return "Nacional"
        return "Nacional"  # por defecto

    meta["ambito"] = meta.apply(lambda r: detectar_ambito(r["titulo"], 
                                                          r.get("dataset_titulo", "")), axis=1)
    return meta

@st.cache_data(show_spinner="Descargando series…")
def fetch_series(ids, start="1900-01-01"):
    SAFE = ",:._-"
    bloques = []
    for sid in ids:
        offset, prev_first = 0, None
        while True:
            qp = {"ids": sid, "start_date": start,
                  "format": "json", "limit": 5000, "offset": offset}
            url = f"{API_BASE}series?" + urllib.parse.urlencode(qp, safe=SAFE)
            js  = requests.get(url, timeout=60).json()
            data = js.get("data", [])
            if not data or data[0][0] == prev_first:
                break
            prev_first = data[0][0]
            blk = (pd.DataFrame(data, columns=["Fecha", sid])
                     .assign(Fecha=lambda d: pd.to_datetime(d["Fecha"]))
                     .set_index("Fecha"))
            bloques.append(blk)
            if len(data) < 5000:
                break
            offset += 5000
    if not bloques:
        return pd.DataFrame()
    return pd.concat(bloques, axis=1).sort_index()

##############################################################################
# 2 ▸ carga metadatos
##############################################################################
meta = load_metadata()

##############################################################################
# 3 ▸ sidebar filtros
##############################################################################
st.sidebar.header("Filtros")

# Ámbito nacional / provincial
ambitos = ["Todos", "Nacional"] + PROVINCIAS
sel_amb = st.sidebar.selectbox("Ámbito", ambitos)
meta_a  = meta if sel_amb == "Todos" else meta[meta["ambito"] == sel_amb]

# Tema
temas = ["Todos"] + sorted(meta_a["dataset_tema"].dropna().unique())
sel_tema = st.sidebar.selectbox("Tema", temas)
meta_t = meta_a if sel_tema == "Todos" else meta_a[meta_a["dataset_tema"] == sel_tema]

# Organismo
orgs = ["Todos"] + sorted(meta_t["dataset_fuente"].dropna().unique())
sel_org = st.sidebar.selectbox("Organismo", orgs)
meta_o  = meta_t if sel_org == "Todos" else meta_t[meta_t["dataset_fuente"] == sel_org]

# Frecuencia
freqs = sorted(meta_o["frecuencia"].unique())
sel_freq = st.sidebar.multiselect("Frecuencia", freqs, default=freqs)
meta_f = meta_o[meta_o["frecuencia"].isin(sel_freq)]

# Búsqueda texto
query = st.sidebar.text_input("Buscar texto")
if query:
    mask = meta_f["titulo"].str.contains(query, case=False, na=False) | \
           meta_f.get("serie_descripcion", "").str.contains(query, case=False, na=False)
    meta_f = meta_f[mask]

# Diccionario “nombre legible” → ID
series_dict = dict(zip(meta_f["titulo_simple"], meta_f["serie_id"]))
st.sidebar.write(f"Series encontradas: **{len(series_dict)}**")
sel_titles = st.sidebar.multiselect("Elige series", list(series_dict))
sel_ids = [series_dict[t] for t in sel_titles]

if not sel_ids:
    st.info("Selecciona al menos una serie válida con datos 2023‑2025.")
    st.stop()

##############################################################################
# 4 ▸ presentación
##############################################################################
view = st.radio("Ver como", ["Tabla", "Gráfico", "Descargar CSV"])

with st.spinner("Descargando datos…"):
    df = fetch_series(sel_ids)
    df = df.rename(columns={v: k for k, v in series_dict.items()})

if df.empty:
    st.error("La API no devolvió valores para esas series.")
    st.stop()

if view == "Tabla":
    st.dataframe(df)
elif view == "Gráfico":
    st.line_chart(df)
else:
    st.download_button("CSV", df.to_csv().encode(),
                       "series_elegidas.csv", "text/csv")

##############################################################################
# 5 ▸ metadatos detalle
##############################################################################
meta_det = (meta_f.set_index("titulo_simple")
                   .loc[sel_titles,
                        ["ambito", "frecuencia", "serie_indice_inicio",
                         "serie_indice_final", "serie_descripcion"]]
                   .rename(columns={"serie_indice_inicio": "primer_dato",
                                    "serie_indice_final":  "último_dato",
                                    "serie_descripcion":   "descripción"}))
with st.expander("Detalles de metadatos"):
    st.dataframe(meta_det)
