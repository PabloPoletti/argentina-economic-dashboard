import streamlit as st
import pandas as pd, requests, urllib.parse, re, altair as alt

API_BASE = "https://apis.datos.gob.ar/series/api/"
META_URL = f"{API_BASE}dump/series-tiempo-metadatos.csv"

# ID auxiliar para deflactar (IPC base dic‑2016=100)
ID_CPI = "145.3_INGNACNAL_DICI_M_15"
SAFE_CHARS = ",:._-"

PROVINCIAS = [
    "Buenos Aires", "Catamarca", "Chaco", "Chubut", "Córdoba", "Corrientes",
    "Entre Ríos", "Formosa", "Jujuy", "La Pampa", "La Rioja", "Mendoza", "Misiones",
    "Neuquén", "Río Negro", "Salta", "San Juan", "San Luis", "Santa Cruz", "Santa Fe",
    "Santiago Del Estero", "Tierra Del Fuego", "Tucumán", "Ciudad Autónoma De Buenos Aires"
]

##############################################################################
# 1 ▸ utilidades cache
##############################################################################
@st.cache_data(show_spinner=False)
def load_metadata():
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
    meta = meta[meta["ultimo_dato"] >= "2023-01-01"]

    def ambito(r):
        txt = f"{r['titulo']} {r.get('dataset_titulo','')}".lower()
        for p in PROVINCIAS:
            if p.lower() in txt:
                return p
        return "Nacional"

    meta["ambito"] = meta.apply(ambito, axis=1)
    return meta

@st.cache_data(show_spinner="Descargando series…")
def fetch_single_series(sid, start="1900-01-01"):
    frames, offset, prev_first = [], 0, None
    while True:
        qp = {"ids": sid, "start_date": start, "format": "json",
              "limit": 5000, "offset": offset}
        url = API_BASE + "series?" + urllib.parse.urlencode(qp, safe=SAFE_CHARS)
        js  = requests.get(url, timeout=40).json()
        data = js.get("data", [])
        if not data or data[0][0] == prev_first:
            break
        prev_first = data[0][0]
        blk = pd.DataFrame(data, columns=["Fecha", sid])
        blk["Fecha"] = pd.to_datetime(blk["Fecha"])
        frames.append(blk.set_index("Fecha"))
        if len(data) < 5000:
            break
        offset += 5000
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_series(ids, start="1900-01-01"):
    dfs = [fetch_single_series(s, start) for s in ids]
    return pd.concat(dfs, axis=1).sort_index() if dfs else pd.DataFrame()

##############################################################################
# 2 ▸ carga metadatos
##############################################################################
meta = load_metadata()

##############################################################################
# 3 ▸ sidebar filtros
##############################################################################
st.sidebar.header("Filtros")
amb_sel = st.sidebar.selectbox("Ámbito", ["Todos","Nacional"] + PROVINCIAS)
meta_sel = meta if amb_sel == "Todos" else meta[meta["ambito"] == amb_sel]

tema = st.sidebar.selectbox("Tema", ["Todos"] + sorted(meta_sel["dataset_tema"].dropna().unique()))
meta_sel = meta_sel if tema == "Todos" else meta_sel[meta_sel["dataset_tema"] == tema]

org = st.sidebar.selectbox("Organismo", ["Todos"] + sorted(meta_sel["dataset_fuente"].dropna().unique()))
meta_sel = meta_sel if org == "Todos" else meta_sel[meta_sel["dataset_fuente"] == org]

freqs = sorted(meta_sel["frecuencia"].unique())
sel_freq = st.sidebar.multiselect("Frecuencia", freqs, default=freqs)
meta_sel = meta_sel[meta_sel["frecuencia"].isin(sel_freq)]

query = st.sidebar.text_input("Buscar texto")
if query:
    mask = meta_sel["titulo"].str.contains(query, case=False, na=False) | \
           meta_sel.get("serie_descripcion", "").str.contains(query, case=False, na=False)
    meta_sel = meta_sel[mask]

series_dict = dict(zip(meta_sel["titulo_simple"], meta_sel["serie_id"]))
st.sidebar.write(f"Series encontradas: **{len(series_dict)}**")
sel_titles = st.sidebar.multiselect("Elige series", list(series_dict))
sel_ids = [series_dict[t] for t in sel_titles]

if not sel_ids:
    st.info("Selecciona al menos una serie vigente.")
    st.stop()

##############################################################################
# 4 ▸ escala nominal / real
##############################################################################
escala = st.sidebar.radio("Escala", ["Nominal", "Reales (IPC dic‑2016=100)"])
aux_ids = [ID_CPI] if escala.startswith("Reales") else []

df = fetch_series(sel_ids + aux_ids)
if df.empty:
    st.error("La API no devolvió datos.")
    st.stop()

if escala.startswith("Reales") and ID_CPI in df.columns:
    cpi = df[ID_CPI].resample("M").last().ffill()/100
    df = df.drop(columns=[ID_CPI]).resample("M").last()
    df = df.divide(cpi, axis=0)

##############################################################################
# 5 ▸ presentación
##############################################################################
view = st.radio("Ver como", ["Tabla", "Gráfico", "Descargar CSV"])

df_disp = df.rename(columns={v: k for k, v in series_dict.items() if v in df.columns})

if view == "Tabla":
    st.dataframe(df_disp)
elif view == "Descargar CSV":
    st.download_button("CSV", df_disp.to_csv().encode(), "series_elegidas.csv", mime="text/csv")
else:
    base = df_disp.reset_index().melt("Fecha", var_name="Serie", value_name="Valor")
    base = base.assign(Fecha_str=base["Fecha"].dt.strftime("%Y-%m-%d"))
    chart = (
        alt.Chart(base).mark_line().encode(
            x=alt.X("Fecha:T", title="Fecha"),
            y=alt.Y("Valor:Q", title="Valor"),
            color="Serie:N",
            tooltip=["Serie:N", "Fecha_str:N", "Valor:Q"]
        ).interactive()
    )
    st.altair_chart(chart, use_container_width=True)
    if escala != "Nominal":
        st.caption("*Serie deflactada según IPC base dic‑2016=100*")

##############################################################################
# 6 ▸ metadatos
##############################################################################
meta_det = (meta_sel.set_index("titulo_simple").loc[sel_titles,
            ["ambito", "frecuencia", "serie_indice_inicio", "serie_indice_final"]]
           .rename(columns={"serie_indice_inicio": "primer dato", "serie_indice_final": "último dato"}))
with st.expander("Metadatos"):
    st.dataframe(meta_det)
