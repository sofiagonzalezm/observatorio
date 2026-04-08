import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from urllib.parse import quote


st.set_page_config(
    page_title="Análisis Forense Municipal",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# ESTILOS
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 14px 16px;
        border-radius: 14px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# CONFIG
# =========================
FILE_PATH = "df_analisis_forense.csv"
SUPPLIER_COL = "compiledRelease/awards/0/suppliers/0/name"
AWARD_COL = "compiledRelease/awards/0/value/amount"

def determinar_tipo(x):
    if pd.isna(x): return ""
    
    nombre = str(x).upper()
    # Definimos qué palabras "anulan" que una coma signifique persona natural
    es_empresa = "SOCIEDAD ANONIMA" in nombre or " S.A." in nombre or " S. A." in nombre
    tiene_coma = "," in nombre
    
    # Solo es natural si tiene coma Y NO es una empresa conocida
    tipo = "natural" if (tiene_coma and not es_empresa) else "juridico"
    
    return (
        f"https://firmaconcerteza.com/dashboard/search?query={quote(str(x))}"
        f"&type={tipo}&page=1&pageSize=10"
    )

# =========================
# CARGA Y LIMPIEZA
# =========================
@st.cache_data

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # quitar columna auxiliar si existe
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # normalizar texto
    text_cols = [
        "unidad", "especie", "etapa_actual", "institucion", "proyecto", "sector",
        "situacion_actual", "municipio", "departamento", "municipio_fuente_pdf",
        "alcalde_ganador", "tipo_organizacion_ganadora", "siglas_ganadora",
        "organizacion_ganadora", SUPPLIER_COL
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # columnas numéricas
    num_cols = [
        "año", "monto_solicitado", "monto_inicial", "monto_vigente", "monto_ejecutado",
        "meta_fisica", "meta_ejecutada", "periodo_alcalde", AWARD_COL, "votos_ganador"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # proveedor limpio
    df["proveedor"] = df[SUPPLIER_COL]
    df["monto_adjudicado"] = df[AWARD_COL]

    # ratios útiles
    df["ratio_meta_ejecucion"] = np.where(
        df["meta_fisica"].fillna(0) > 0,
        df["meta_ejecutada"] / df["meta_fisica"],
        np.nan,
    )

    df["ratio_presupuesto_ejecucion"] = np.where(
        df["monto_solicitado"].fillna(0) > 0,
        df["monto_ejecutado"] / df["monto_solicitado"],
        np.nan,
    )

    # ciclo electoral Guatemala municipal
    election_years = [2015, 2019, 2023, 2027]

    def classify_election_period(year):
        if pd.isna(year):
            return "Sin año"
        year = int(year)
        if year in election_years:
            return "Año electoral"
        if year in [y - 1 for y in election_years]:
            return "Año previo"
        if year in [y + 1 for y in election_years]:
            return "Año posterior"
        return "Año regular"

    df["periodo_electoral"] = df["año"].apply(classify_election_period)

    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")

    departamentos = sorted(df["departamento"].dropna().unique().tolist())
    dept_sel = st.sidebar.multiselect("Departamento", departamentos)

    df_f = df.copy()
    if dept_sel:
        df_f = df_f[df_f["departamento"].isin(dept_sel)]

    municipios = sorted(df_f["municipio"].dropna().unique().tolist())
    muni_sel = st.sidebar.multiselect("Municipalidad", municipios)
    if muni_sel:
        df_f = df_f[df_f["municipio"].isin(muni_sel)]

    sectores = sorted(df_f["sector"].dropna().unique().tolist())
    sector_sel = st.sidebar.multiselect("Sector", sectores)
    if sector_sel:
        df_f = df_f[df_f["sector"].isin(sector_sel)]

    min_year = int(df["año"].min())
    max_year = int(df["año"].max())
    year_range = st.sidebar.slider("Rango de años", min_year, max_year, (min_year, max_year))
    df_f = df_f[df_f["año"].between(year_range[0], year_range[1])]

    st.sidebar.markdown("---")
    etapa_sel = st.sidebar.multiselect(
        "Etapa actual",
        sorted(df_f["etapa_actual"].dropna().unique().tolist())
    )
    if etapa_sel:
        df_f = df_f[df_f["etapa_actual"].isin(etapa_sel)]

    return df_f


# =========================
# HELPERS ANALÍTICOS
# =========================
def only_one_supplier_by_group(df, group_col):
    out = (
        df.groupby(group_col)
        .agg(
            proveedores_unicos=("proveedor", "nunique"),
            proveedor_principal=("proveedor", lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else np.nan),
            proyectos=("snip", "nunique"),
            monto_total=("monto_ejecutado", "sum")
        )
        .reset_index()
        .sort_values(["proveedores_unicos", "monto_total"], ascending=[True, False])
    )
    return out[out["proveedores_unicos"] == 1]


def supplier_concentration(df, group_col):
    tmp = (
        df.groupby([group_col, "proveedor"])
        .agg(
            monto_total=("monto_adjudicado", "sum"),
            proyectos=("snip", "nunique")
        )
        .reset_index()
    )

    total_group = (
        tmp.groupby(group_col)["monto_total"]
        .sum()
        .rename("monto_grupo")
        .reset_index()
    )

    tmp = tmp.merge(total_group, on=group_col, how="left")
    tmp["share_grupo"] = np.where(tmp["monto_grupo"] > 0, tmp["monto_total"] / tmp["monto_grupo"], np.nan)

    top = (
        tmp.sort_values([group_col, "share_grupo"], ascending=[True, False])
        .groupby(group_col)
        .head(1)
        .sort_values("share_grupo", ascending=False)
    )
    return top


def format_money(x):
    if pd.isna(x):
        return "-"
    return f"Q {x:,.0f}"


# =========================
# APP
# =========================
df = load_data(FILE_PATH)
df_filtered = apply_filters(df)

st.title("Análisis Forense de Proyectos Municipales")
st.caption("Dashboard exploratorio para detectar concentración de proveedores, patrones por alcalde, ejecución de metas y posibles anomalías alrededor de elecciones.")

if df_filtered.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

# =========================
# KPIS
# =========================
col1, col2, col3,  = st.columns(3)
col1.metric("Registros", f"{len(df_filtered):,}")
col2.metric("Municipios", f"{df_filtered['municipio'].nunique():,}")
col3.metric("Alcaldes", f"{df_filtered['alcalde_ganador'].nunique():,}")


col4, col5 = st.columns(2)
col4.metric("Proveedores", f"{df_filtered['proveedor'].nunique():,}")
col5.metric("% promedio de meta ejecutada", f"{df_filtered['ratio_meta_ejecucion'].mean() * 100:,.1f}%")

col6, col7 = st.columns(2)
col6.metric("Monto ejecutado total", format_money(df_filtered["monto_ejecutado"].sum()))
col7.metric("Monto adjudicado total", format_money(df_filtered["monto_adjudicado"].sum()))


# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Municipios y proveedores",
    "Alcaldes y proveedores",
    "Proyectos sospechosos",
    "Análisis por partido político",
    "Proveedores",
    "Costo por unidad física"
])

# -------------------------
# TAB 1
# -------------------------

with tab1:
    muni_one_insights = only_one_supplier_by_group(df_filtered, "municipio").copy()
    muni_one_insights = muni_one_insights[muni_one_insights["proyectos"] > 2].copy()

    muni_conc_insights = supplier_concentration(df_filtered, "municipio").copy()

    # 1) Municipio con mayor concentración
    top_conc_text = "N/D"
    if not muni_conc_insights.empty:
        top_conc = muni_conc_insights.iloc[0]
        top_conc_text = f"{top_conc['municipio']} ({top_conc['share_grupo']:.1%})"

    # 2) Cuántos municipios cumplen condición
    count_unique_supplier = muni_one_insights["municipio"].nunique() if not muni_one_insights.empty else 0

    # 3) Municipio con mayor monto ejecutado dentro del grupo
    top_monto_text = "N/D"
    if not muni_one_insights.empty:
        top_monto_row = muni_one_insights.sort_values("monto_total", ascending=False).iloc[0]
        top_monto_text = f"{top_monto_row['municipio']} (Q {top_monto_row['monto_total']:,.0f})"

    # 4) Municipio con peor ratio meta ejecutada dentro del grupo
    worst_ratio_text = "N/D"
    ratio_muni_all = (
        df_filtered.groupby("municipio")["ratio_meta_ejecucion"]
        .mean()
        .reset_index(name="ratio_promedio")
        .sort_values("ratio_promedio", ascending=True)
    )

    if not ratio_muni_all.empty:
        worst_ratio = ratio_muni_all.iloc[0]
        worst_ratio_text = f"{worst_ratio['municipio']} ({worst_ratio['ratio_promedio']:.2%})"

    # 5) Municipio con mayor ratio de proyectos sospechosos (TODOS los proyectos)
    top_sospechoso_text = "N/D"

    df_tmp = df_filtered.copy()
    df_tmp["diferencia"] = df_tmp["monto_adjudicado"] - df_tmp["monto_ejecutado"]

    df_tmp["sospechoso"] = np.where(
        (df_tmp["diferencia"].abs() < 1000) &
        (df_tmp["ratio_meta_ejecucion"] < 0.90),
        1,
        0
    )

    ratio_sos = (
        df_tmp.groupby("municipio")
        .agg(
            sospechosos=("sospechoso", "sum"),
            total_proyectos=("snip", "nunique")
        )
    )

    ratio_sos["ratio"] = ratio_sos["sospechosos"] / ratio_sos["total_proyectos"]

    # opcional → evitar ruido (ej: municipios con 1 proyecto)
    ratio_sos = ratio_sos[ratio_sos["total_proyectos"] >= 3]

    ratio_sos = ratio_sos.sort_values("ratio", ascending=False)

    if not ratio_sos.empty:
        top = ratio_sos.iloc[0]
        top_sospechoso_text = f"{ratio_sos.index[0]} ({top['ratio']:.2%})"

    st.markdown(f"- **Municipio con mayor concentración en un solo proveedor:** {top_conc_text}")
    st.markdown(f"- **Municipios con un único proveedor y más de 2 proyectos:** {count_unique_supplier:,.0f}")
    st.markdown(f"- **Municipio con mayor monto ejecutado dentro de ese grupo:** {top_monto_text}")
    st.markdown(f"- **Municipio con menor promedio de ratio meta ejecutada:** {worst_ratio_text}")
    st.markdown(f"- **Municipio con mayor proporción de proyectos sospechosos:** {top_sospechoso_text}")
    st.subheader("Municipios que trabajan con un solo proveedor")

    # municipios con un único proveedor válido
    muni_one = only_one_supplier_by_group(df_filtered, "municipio").copy()
    muni_one = muni_one[muni_one["proyectos"] > 2].copy()

    # métricas SOLO dentro del municipio
    metricas_muni = (
        df_filtered.groupby("municipio")
        .agg(
            monto_total_ejecutado=("monto_ejecutado", "sum"),
            promedio_monto_ejecutado=("monto_ejecutado", "mean"),
            promedio_ratio_meta_ejecutada=("ratio_meta_ejecucion", "mean"),
            departamento=("departamento", lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else ""),
        )
        .reset_index()
    )

    muni_one = muni_one.merge(metricas_muni, on="municipio", how="left")

    muni_display = muni_one[
        [
            "municipio",
            "departamento",
            "proveedor_principal",
            "proyectos",
            "monto_total_ejecutado",
            "promedio_monto_ejecutado",
            "promedio_ratio_meta_ejecutada",
        ]
    ].copy()
    
    muni_display["proveedor_principal"] = muni_display["proveedor_principal"].apply(determinar_tipo)
    muni_display["monto_total_ejecutado"] = pd.to_numeric(muni_display["monto_total_ejecutado"], errors="coerce")
    muni_display["promedio_monto_ejecutado"] = pd.to_numeric(muni_display["promedio_monto_ejecutado"], errors="coerce")
    muni_display["proyectos"] = pd.to_numeric(muni_display["proyectos"], errors="coerce")
    muni_display["promedio_ratio_meta_ejecutada"] = muni_display["promedio_ratio_meta_ejecutada"] * 100


    styled_muni = muni_display.style.format(
        {
            "proyectos": "{:,.0f}",
            "monto_total_ejecutado": "{:,.2f}",
            "promedio_monto_ejecutado": "{:,.2f}",
        }
    )


    st.caption("Solo se muestran municipios con un único proveedor válido y más de dos proyectos. Todas las métricas se calculan únicamente dentro de ese municipio.")
    st.dataframe(styled_muni, use_container_width=True, 
                 column_config={
            "proveedor_principal": st.column_config.LinkColumn(
            "Proveedor principal",
             display_text=r"query=(.*?)&"
        ),
            "promedio_ratio_meta_ejecutada": st.column_config.NumberColumn(
            "Promedio Ratio Meta ejecutada",
            format="%.2f%%" ), })

    st.subheader("Buscar municipio")
    municipios_disponibles = sorted(df_filtered["municipio"].dropna().unique().tolist())
    municipio_sel = st.selectbox("Selecciona un municipio", options=[""] + municipios_disponibles)

    if municipio_sel:
        detalle_municipio = df_filtered[df_filtered["municipio"] == municipio_sel].copy()

        detalle_cols = [
            col for col in [
                "proyecto",
                "proveedor",
                "alcalde_ganador",
                "monto_adjudicado",
                "monto_ejecutado",
                "meta_ejecutada",
                "ratio_meta_ejecucion",
            ] if col in detalle_municipio.columns
        ]

        detalle_municipio = detalle_municipio[detalle_cols].copy()

        detalle_municipio = detalle_municipio.rename(
            columns={
                "alcalde_ganador": "alcalde",
                "ratio_meta_ejecucion": "ratio_meta_ejecutada",
            }
        )

        # brecha adjudicado vs ejecutado SOLO cuando adjudicado existe
        detalle_municipio["brecha_adjudicado_ejecutado"] = np.where(
            detalle_municipio["monto_adjudicado"].notna(),
            detalle_municipio["monto_adjudicado"] - detalle_municipio["monto_ejecutado"],
            np.nan
        )

        # proyectos sospechosos originales
        detalle_municipio["sospechoso"] = np.where(
            (
                detalle_municipio["brecha_adjudicado_ejecutado"].abs() < 1000
            ) &
            (
                detalle_municipio["ratio_meta_ejecutada"] < 0.90
            ) &
            (
                detalle_municipio["monto_adjudicado"].notna()
            ),
            "Sí",
            "No"
        )

        # nuevo flag: meta_ejecutada = 0 pero con gasto
        detalle_municipio["sin_meta_ejecutada_con_gasto"] = np.where(
            (detalle_municipio["meta_ejecutada"] == 0) &
            (detalle_municipio["monto_ejecutado"] > 0),
            "Sí",
            "No"
        )

        pct_sospechosos = (
            (detalle_municipio["sospechoso"] == "Sí").mean()
            if not detalle_municipio.empty else np.nan
        )

        pct_sin_meta_ejecutada = (
            (detalle_municipio["sin_meta_ejecutada_con_gasto"] == "Sí").mean()
            if not detalle_municipio.empty else np.nan
        )

        total_proveedores = detalle_municipio["proveedor"].dropna().nunique() if "proveedor" in detalle_municipio.columns else 0
        total_proyectos = detalle_municipio["proyecto"].nunique() if "proyecto" in detalle_municipio.columns else 0
        monto_total_ejecutado = detalle_municipio["monto_ejecutado"].sum() if "monto_ejecutado" in detalle_municipio.columns else 0
        ratio_promedio = detalle_municipio["ratio_meta_ejecutada"].mean() if "ratio_meta_ejecutada" in detalle_municipio.columns else np.nan
        detalle_municipio["sobreejecucion_financiera"] = np.where((detalle_municipio["monto_adjudicado"].notna()) &(detalle_municipio["monto_adjudicado"] < detalle_municipio["monto_ejecutado"]),1,0)
        proyectos_subejecucion = detalle_municipio["sobreejecucion_financiera"].sum()

        k1, k2, k3 = st.columns(3)
        k1.metric("Número de proveedores", f"{total_proveedores:,.0f}")
        k2.metric("Número de proyectos", f"{total_proyectos:,.0f}")
        k3.metric("% de proyectos sospechosos", f"{pct_sospechosos:.2%}" if pd.notna(pct_sospechosos) else "")

        k4, k5, k6, k7 = st.columns(4)
        k4.metric("Total monto ejecutado (Q)", f"{monto_total_ejecutado:,.0f}")
        k5.metric("Promedio ratio meta ejecutada", f"{ratio_promedio:.2%}" if pd.notna(ratio_promedio) else "")
        k6.metric("% con meta física = 0 y gasto", f"{pct_sin_meta_ejecutada:.2%}" if pd.notna(pct_sin_meta_ejecutada) else "")
        k7.metric("Proyectos con sobre-ejecución", f"{proyectos_subejecucion:,.0f}")

        # ordenar: primero meta física 0 con gasto, luego sospechosos, luego mayor monto
        detalle_municipio["orden_meta0"] = np.where(detalle_municipio["sin_meta_ejecutada_con_gasto"] == "Sí", 0, 1)
        detalle_municipio["orden_sospechoso"] = np.where(detalle_municipio["sospechoso"] == "Sí", 0, 1)

        detalle_municipio = detalle_municipio.sort_values(
            by=["orden_meta0", "orden_sospechoso", "monto_ejecutado"],
            ascending=[True, True, False]
        ).drop(columns=["orden_meta0", "orden_sospechoso"])

        # columnas finales
        detalle_municipio = detalle_municipio[
            [
                "proyecto",
                "proveedor",
                "alcalde",
                "monto_adjudicado",
                "monto_ejecutado",
                "brecha_adjudicado_ejecutado",
                "meta_ejecutada",
                "ratio_meta_ejecutada",
                "sin_meta_ejecutada_con_gasto",
                "sospechoso",
            ]
        ].copy()

        detalle_display = detalle_municipio.copy()

        def highlight_flags(row):
            if row["sin_meta_ejecutada_con_gasto"] == "Sí":
                return ["background-color: #fde2e2"] * len(row)   # rojo más fuerte
            if row["sospechoso"] == "Sí":
                return ["background-color: #f8d7da"] * len(row)   # rojo claro
            return [""] * len(row)
        

        detalle_display["ratio_meta_ejecutada"] = detalle_display["ratio_meta_ejecutada"] * 100
        detalle_display["proveedor"] = detalle_display["proveedor"].apply(determinar_tipo)
        detalle_display["alcalde"] = detalle_display["alcalde"].apply(
        lambda x: (
            f"https://firmaconcerteza.com/dashboard/search?query={quote(str(x))}"
            f"&type={'natural'}&page=1&pageSize=10"
            if pd.notna(x) else ""
        ))
        detalle_display["monto_adjudicado"] = pd.to_numeric(detalle_display["monto_adjudicado"], errors="coerce")
        detalle_display["monto_ejecutado"] = pd.to_numeric(detalle_display["monto_ejecutado"], errors="coerce")
        detalle_display["brecha_adjudicado_ejecutado"] = pd.to_numeric(detalle_display["brecha_adjudicado_ejecutado"], errors="coerce")
        detalle_display["meta_ejecutada"] = pd.to_numeric(detalle_display["meta_ejecutada"], errors="coerce")

        styled_df = (
            detalle_display.style
            .format(
            {
                "monto_adjudicado": "{:,.2f}",
                "monto_ejecutado": "{:,.2f}",
                "brecha_adjudicado_ejecutado": "{:,.2f}",
                "meta_ejecutada": "{:,.0f}",

            }
        ).apply(highlight_flags, axis=1).hide(axis="columns", subset=["sospechoso", "sin_meta_ejecutada_con_gasto"])
        )
        
        st.dataframe(styled_df, use_container_width=True, 
                     column_config={
              "proveedor": st.column_config.LinkColumn(
            "Proveedor",
             display_text=r"query=(.*?)&"
        ),
         "alcalde": st.column_config.LinkColumn(
            "Alcalde",
             display_text=r"query=(.*?)&"
        ),
            "ratio_meta_ejecutada": st.column_config.NumberColumn(
            "Ratio Meta ejecutada",
            format="%.2f%%" ), })

        st.caption(
            "Los proyectos resaltados en rojo muestran señales de riesgo. "
            "Rojo más intenso: proyectos con meta física igual a cero pero con gasto ejecutado. "
            "Rojo claro: proyectos que ejecutaron todo o casi todo su monto adjudicado, "
            "pero no han completado la meta física."
        )
       

# -------------------------
# TAB 2
# -------------------------

with tab2:
    st.subheader("Hallazgos clave")

    mayor_one_insights = only_one_supplier_by_group(df_filtered, "alcalde_ganador").copy()
    mayor_one_insights = mayor_one_insights[mayor_one_insights["proyectos"] > 2].copy()

    mayor_conc_insights = supplier_concentration(df_filtered, "alcalde_ganador").copy()

    # lookup alcalde -> partido
    partido_lookup = (
        df_filtered.groupby("alcalde_ganador")["siglas_ganadora"]
        .agg(lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else "N/D")
        .to_dict()
    )

    # 1) Alcalde con mayor concentración
    top_conc_text = "N/D"
    if not mayor_conc_insights.empty:
        top_conc = mayor_conc_insights.iloc[0]
        partido = partido_lookup.get(top_conc["alcalde_ganador"], "N/D")
        top_conc_text = f"{top_conc['alcalde_ganador']} ({partido} — {top_conc['share_grupo']:.1%})"

    # 2) Cuántos alcaldes cumplen condición
    count_unique_supplier = mayor_one_insights["alcalde_ganador"].nunique() if not mayor_one_insights.empty else 0

    # 3) Alcalde con mayor monto ejecutado dentro del grupo
    top_monto_text = "N/D"
    if not mayor_one_insights.empty:
        top_monto_row = mayor_one_insights.sort_values("monto_total", ascending=False).iloc[0]
        partido = partido_lookup.get(top_monto_row["alcalde_ganador"], "N/D")
        top_monto_text = f"{top_monto_row['alcalde_ganador']} ({partido} — Q {top_monto_row['monto_total']:,.0f})"

    # 4) Alcalde con peor ratio meta ejecutada
    worst_ratio_text = "N/D"
    if not mayor_one_insights.empty:
        ratio_alcalde = (
            df_filtered[df_filtered["alcalde_ganador"].isin(mayor_one_insights["alcalde_ganador"])]
            .groupby("alcalde_ganador")["ratio_meta_ejecucion"]
            .mean()
            .reset_index(name="ratio_promedio")
            .sort_values("ratio_promedio", ascending=True)
        )
        if not ratio_alcalde.empty:
            worst_ratio = ratio_alcalde.iloc[0]
            partido = partido_lookup.get(worst_ratio["alcalde_ganador"], "N/D")
            worst_ratio_text = f"{worst_ratio['alcalde_ganador']} ({partido} — {worst_ratio['ratio_promedio']:.2%})"

    # 5) Alcalde con mayor proporción de proyectos sospechosos
    top_sospechoso_text = "N/D"

    df_tmp = df_filtered.copy()
    df_tmp["diferencia"] = df_tmp["monto_adjudicado"] - df_tmp["monto_ejecutado"]
    df_tmp["sospechoso"] = np.where(
        (df_tmp["diferencia"].abs() < 1000) &
        (df_tmp["ratio_meta_ejecucion"] < 0.90),
        1,
        0
    )

    ratio_sos = (
        df_tmp.groupby("alcalde_ganador")
        .agg(
            sospechosos=("sospechoso", "sum"),
            total_proyectos=("snip", "nunique")
        )
    )

    ratio_sos = ratio_sos[ratio_sos["total_proyectos"] >= 3]
    ratio_sos["ratio"] = ratio_sos["sospechosos"] / ratio_sos["total_proyectos"]
    ratio_sos = ratio_sos.sort_values("ratio", ascending=False)

    if not ratio_sos.empty:
        alcalde_top = ratio_sos.index[0]
        partido = partido_lookup.get(alcalde_top, "N/D")
        top_sospechoso_text = f"{alcalde_top} ({partido} — {ratio_sos.iloc[0]['ratio']:.2%})"

    st.markdown(f"- **Alcalde con mayor concentración en un solo proveedor:** {top_conc_text}")
    st.markdown(f"- **Alcaldes con un único proveedor y más de 2 proyectos:** {count_unique_supplier:,.0f}")
    st.markdown(f"- **Alcalde con mayor monto ejecutado dentro de ese grupo:** {top_monto_text}")
    st.markdown(f"- **Alcalde con menor promedio de ratio meta ejecutada dentro de ese grupo:** {worst_ratio_text}")
    st.markdown(f"- **Alcalde con mayor proporción de proyectos sospechosos:** {top_sospechoso_text}")

    st.subheader("Alcaldes que trabajan con un solo proveedor")

    mayor_one = only_one_supplier_by_group(df_filtered, "alcalde_ganador").copy()
    mayor_one = mayor_one[mayor_one["proyectos"] > 2].copy()

    # flags auxiliares para agregación
    df_flags = df_filtered.copy()
    df_flags["sin_meta_ejecutada_con_gasto"] = np.where(
        (df_flags["meta_ejecutada"] == 0) & (df_flags["monto_ejecutado"] > 0),
        1,
        0
    )
    df_flags["sobreejecucion_financiera"] = np.where(
        (df_flags["monto_adjudicado"].notna()) & (df_flags["monto_ejecutado"] > df_flags["monto_adjudicado"]),
        1,
        0
    )

    # métricas + ubicación
    metricas_alcalde = (
        df_flags.groupby("alcalde_ganador")
        .agg(
            periodo_alcalde=("periodo_alcalde", "first"),
            monto_total_ejecutado=("monto_ejecutado", "sum"),
            promedio_monto_ejecutado=("monto_ejecutado", "mean"),
            promedio_ratio_meta_ejecutada=("ratio_meta_ejecucion", "mean"),
            proyectos_meta0_gasto=("sin_meta_ejecutada_con_gasto", "sum"),
            proyectos_sobreejecucion=("sobreejecucion_financiera", "sum"),
            municipio=("municipio", lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else ""),
            departamento=("departamento", lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else ""),
        )
        .reset_index()
    )

    mayor_one = mayor_one.merge(metricas_alcalde, on="alcalde_ganador", how="left")

    mayor_one_display = mayor_one[
        [
            "alcalde_ganador",
            "periodo_alcalde",
            "municipio",
            "departamento",
            "proveedor_principal",
            "proyectos",
            "monto_total_ejecutado",
            "promedio_monto_ejecutado",
            "promedio_ratio_meta_ejecutada",
            "proyectos_meta0_gasto",
            "proyectos_sobreejecucion",
        ]
    ].copy()

    # formatear

   # if "promedio_ratio_meta_ejecutada" in mayor_one_display.columns:
     #   mayor_one_display["promedio_ratio_meta_ejecutada"] = mayor_one_display["promedio_ratio_meta_ejecutada"].map(
     #       lambda x: f"{x:.2%}" if pd.notna(x) else ""
     #   )

    st.caption("Solo se muestran alcaldes con un único proveedor válido y más de dos proyectos.")

    mayor_one_display["proveedor_principal"] = mayor_one_display["proveedor_principal"].apply(determinar_tipo)
    mayor_one_display["alcalde_ganador"] = mayor_one_display["alcalde_ganador"].apply(
        lambda x: (
            f"https://firmaconcerteza.com/dashboard/search?query={quote(str(x))}"
            f"&type={'natural'}&page=1&pageSize=10"
            if pd.notna(x) else ""
        ))
    
    mayor_one_display["proyectos"] = pd.to_numeric(mayor_one_display["proyectos"], errors="coerce")
    mayor_one_display["monto_total_ejecutado"] = pd.to_numeric(mayor_one_display["monto_total_ejecutado"], errors="coerce")
    mayor_one_display["promedio_monto_ejecutado"] = pd.to_numeric(mayor_one_display["promedio_monto_ejecutado"], errors="coerce")
    mayor_one_display["proyectos_meta0_gasto"] = pd.to_numeric(mayor_one_display["proyectos_meta0_gasto"], errors="coerce")
    mayor_one_display["proyectos_sobreejecucion"] = pd.to_numeric(mayor_one_display["proyectos_sobreejecucion"], errors="coerce")
    mayor_one_display["promedio_ratio_meta_ejecutada"] = mayor_one_display["promedio_ratio_meta_ejecutada"] * 100

    styled_mayor_one_display = (
            mayor_one_display.style
            .format(
            {
                "proyectos": "{:,.0f}",
                "monto_total_ejecutado": "{:,.2f}",
                "promedio_monto_ejecutado": "{:,.2f}",
                "proyectos_meta0_gasto": "{:,.0f}",
                "proyectos_sobreejecucion": "{:,.0f}",
                "periodo_alcalde": "{:.0f}",
            }
        ))
        

    st.dataframe(styled_mayor_one_display, use_container_width=True,  column_config={
         "proveedor_principal": st.column_config.LinkColumn(
            "Proveedor Principal",
             display_text=r"query=(.*?)&"
        ),
         "alcalde_ganador": st.column_config.LinkColumn(
            "Alcalde",
             display_text=r"query=(.*?)&"
        ),
         "promedio_ratio_meta_ejecutada": st.column_config.NumberColumn(
            "Promedio Ratio Meta ejecutada",
            format="%.2f%%" ),})

    st.subheader("Relación entre número de proyectos y monto ejecutado por alcalde")

    scatter_alcaldes = mayor_one.copy()

    fig = px.scatter(
        scatter_alcaldes,
        x="proyectos",
        y="monto_total_ejecutado",
        size="promedio_monto_ejecutado",
        hover_data={
            "alcalde_ganador": True,
            "municipio": True,
            "departamento": True,
            "proveedor_principal": True,
            "promedio_monto_ejecutado": ":,.0f",
            "promedio_ratio_meta_ejecutada": ":.2%",
            "proyectos_meta0_gasto": True,
            "proyectos_sobreejecucion": True,
        },
        title="Alcaldes con un solo proveedor: proyectos vs monto ejecutado"
    )

    fig.update_layout(
        xaxis_title="Número de proyectos",
        yaxis_title="Monto total ejecutado",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Buscar alcalde")
    alcaldes_disponibles = sorted(df_filtered["alcalde_ganador"].dropna().unique().tolist())
    alcalde_sel = st.selectbox("Selecciona un alcalde", options=[""] + alcaldes_disponibles)

    if alcalde_sel:
        detalle_alcalde = df_filtered[df_filtered["alcalde_ganador"] == alcalde_sel].copy()

        detalle_cols = [
            col for col in [
                "proyecto",
                "proveedor",
                "año",
                "monto_adjudicado",
                "monto_ejecutado",
                "meta_ejecutada",
                "ratio_meta_ejecucion",
            ] if col in detalle_alcalde.columns
        ]

        detalle_alcalde = detalle_alcalde[detalle_cols].copy()

        detalle_alcalde = detalle_alcalde.rename(
            columns={"ratio_meta_ejecucion": "ratio_meta_ejecutada"}
        )

        # brecha adjudicado vs ejecutado solo cuando adjudicado existe
        detalle_alcalde["brecha_adjudicado_ejecutado"] = np.where(
            detalle_alcalde["monto_adjudicado"].notna(),
            detalle_alcalde["monto_adjudicado"] - detalle_alcalde["monto_ejecutado"],
            np.nan
        )

        # flags forenses
        detalle_alcalde["sospechoso"] = np.where(
            (detalle_alcalde["brecha_adjudicado_ejecutado"].abs() < 1000) &
            (detalle_alcalde["ratio_meta_ejecutada"] < 0.90) &
            (detalle_alcalde["monto_adjudicado"].notna()),
            "Sí",
            "No"
        )

        detalle_alcalde["sin_meta_ejecutada_con_gasto"] = np.where(
            (detalle_alcalde["meta_ejecutada"] == 0) &
            (detalle_alcalde["monto_ejecutado"] > 0),
            "Sí",
            "No"
        )

        detalle_alcalde["sobreejecucion_financiera"] = np.where(
            (detalle_alcalde["monto_adjudicado"].notna()) &
            (detalle_alcalde["monto_ejecutado"] > detalle_alcalde["monto_adjudicado"]),
            1,
            0
        )

        pct_sospechosos = (
            (detalle_alcalde["sospechoso"] == "Sí").mean()
            if not detalle_alcalde.empty else np.nan
        )

        pct_sin_meta_ejecutada = (
            (detalle_alcalde["sin_meta_ejecutada_con_gasto"] == "Sí").mean()
            if not detalle_alcalde.empty else np.nan
        )

        total_proyectos = detalle_alcalde["proyecto"].nunique() if "proyecto" in detalle_alcalde.columns else 0
        monto_total_ejecutado = detalle_alcalde["monto_ejecutado"].sum() if "monto_ejecutado" in detalle_alcalde.columns else 0
        ratio_promedio = detalle_alcalde["ratio_meta_ejecutada"].mean() if "ratio_meta_ejecutada" in detalle_alcalde.columns else np.nan
        municipio_val = df_filtered.loc[df_filtered["alcalde_ganador"] == alcalde_sel, "municipio"].dropna().mode().iloc[0] if not df_filtered.loc[df_filtered["alcalde_ganador"] == alcalde_sel, "municipio"].dropna().empty else ""
        departamento_val = df_filtered.loc[df_filtered["alcalde_ganador"] == alcalde_sel, "departamento"].dropna().mode().iloc[0] if not df_filtered.loc[df_filtered["alcalde_ganador"] == alcalde_sel, "departamento"].dropna().empty else ""
        año_val = df_filtered.loc[df_filtered["alcalde_ganador"] == alcalde_sel, "periodo_alcalde"].dropna().mode().iloc[0] if not df_filtered.loc[df_filtered["alcalde_ganador"] == alcalde_sel, "periodo_alcalde"].dropna().empty else ""
        proyectos_meta0_gasto = (detalle_alcalde["sin_meta_ejecutada_con_gasto"] == "Sí").sum()
        proyectos_sobreejecucion = detalle_alcalde["sobreejecucion_financiera"].sum()

        k1, k2, k3 = st.columns(3)
        k1.metric("Municipio", municipio_val)
        k2.metric("Departamento", departamento_val)
        k3.metric("Año Electo", año_val)
        
        
        k4, k5, k6 = st.columns(3)
        k4.metric("Total proyectos", f"{total_proyectos:,.0f}")
        k5.metric("Monto total ejecutado (Q)", f"{monto_total_ejecutado:,.0f}")
        k6.metric("Promedio ratio meta ejecutada", f"{ratio_promedio:.2%}" if pd.notna(ratio_promedio) else "")
        

        k7, k8, k9 = st.columns(3)
        k7.metric("% de proyectos sospechosos", f"{pct_sospechosos:.2%}" if pd.notna(pct_sospechosos) else "")
        k8.metric("% con meta física = 0 y gasto", f"{pct_sin_meta_ejecutada:.2%}" if pd.notna(pct_sin_meta_ejecutada) else "")
        k9.metric("Proyectos con sobreejecución", f"{proyectos_sobreejecucion:,.0f}")

        # ordenar
        detalle_alcalde["orden_meta0"] = np.where(detalle_alcalde["sin_meta_ejecutada_con_gasto"] == "Sí", 0, 1)
        detalle_alcalde["orden_sospechoso"] = np.where(detalle_alcalde["sospechoso"] == "Sí", 0, 1)

        detalle_alcalde = detalle_alcalde.sort_values(
            by=["orden_meta0", "orden_sospechoso", "monto_ejecutado"],
            ascending=[True, True, False]
        ).drop(columns=["orden_meta0", "orden_sospechoso"])

        detalle_alcalde = detalle_alcalde[
            [
                "proyecto",
                "proveedor",
                "año",
                "monto_adjudicado",
                "monto_ejecutado",
                "brecha_adjudicado_ejecutado",
                "meta_ejecutada",
                "ratio_meta_ejecutada",
                "sin_meta_ejecutada_con_gasto",
                "sospechoso",
            ]
        ].copy()

        detalle_display = detalle_alcalde.copy()

        def highlight_flags(row):
            if row["sin_meta_ejecutada_con_gasto"] == "Sí":
                return ["background-color: #fde2e2"] * len(row)
            if row["sospechoso"] == "Sí":
                return ["background-color: #f8d7da"] * len(row)
            return [""] * len(row)


        detalle_display["proveedor"] = detalle_display["proveedor"].apply(determinar_tipo)
        detalle_display["monto_adjudicado"] = pd.to_numeric(detalle_display["monto_adjudicado"], errors="coerce")
        detalle_display["monto_ejecutado"] = pd.to_numeric(detalle_display["monto_ejecutado"], errors="coerce")
        detalle_display["brecha_adjudicado_ejecutado"] = pd.to_numeric(detalle_display["brecha_adjudicado_ejecutado"], errors="coerce")
        detalle_display["meta_ejecutada"] = pd.to_numeric(detalle_display["meta_ejecutada"], errors="coerce")
        detalle_display["ratio_meta_ejecutada"] = detalle_display["ratio_meta_ejecutada"] * 100
        
        styled_df = (
            detalle_display.style
            .apply(highlight_flags, axis=1)
            .hide(axis="columns", subset=["sospechoso", "sin_meta_ejecutada_con_gasto"])
            .format(
                {
                    "monto_adjudicado": "{:,.2f}",
                    "monto_ejecutado": "{:,.2f}",
                    "brecha_adjudicado_ejecutado": "{:,.2f}",
                    "meta_ejecutada": "{:,.2f}",
                }
        ))

        st.dataframe(styled_df, use_container_width=True, 
                     column_config={
         "proveedor": st.column_config.LinkColumn(
            "Proveedor",
             display_text=r"query=(.*?)&"
        ), "ratio_meta_ejecutada": st.column_config.NumberColumn(
            "Ratio Meta ejecutada",
            format="%.2f%%" ),})

        st.caption(
            "Los proyectos resaltados en rojo muestran señales de riesgo. "
            "Rojo más intenso: proyectos con meta física igual a cero pero con gasto ejecutado. "
            "Rojo claro: proyectos que ejecutaron todo o casi todo su monto adjudicado, "
            "pero no han completado la meta física."
        )

# -------------------------
# TAB 3
# -------------------------
with open("municipalidades.json", "r", encoding="utf-8") as f:
    geojson_munis = json.load(f)


with tab3:
    st.subheader("Hallazgos clave")

    df_sos = df_filtered.copy()

    # flags
    df_sos["diferencia"] = df_sos["monto_adjudicado"] - df_sos["monto_ejecutado"]

    df_sos["sospechoso"] = np.where(
        (df_sos["monto_adjudicado"].notna()) &
        (df_sos["diferencia"].abs() < 1000) &
        (df_sos["ratio_meta_ejecucion"] < 0.90),
        1,
        0
    )

    df_sos["sin_meta_ejecutada_con_gasto"] = np.where(
        (df_sos["meta_ejecutada"] == 0) &
        (df_sos["monto_ejecutado"] > 0),
        1,
        0
    )

    df_sos["sobreejecucion_financiera"] = np.where(
        (df_sos["monto_adjudicado"].notna()) &
        (df_sos["monto_ejecutado"] > df_sos["monto_adjudicado"]),
        1,
        0
    )
    # proveedores sospechosos por municipio
    prov_sos_muni = (
        df_sos[df_sos["sospechoso"] == 1]
        .groupby(["municipio", "proveedor"])["snip"]
        .nunique()
        .reset_index(name="proyectos_sospechosos")
    )

    # contar proveedores sospechosos únicos por municipio
    mapa_prov_sos = (
        prov_sos_muni.groupby("municipio")["proveedor"]
        .nunique()
        .reset_index(name="num_proveedores_sospechosos")
    )

    mapa_prov_sos["municipio"] = mapa_prov_sos["municipio"].astype(str).str.strip().str.upper()

    # lookup alcalde -> partido
    partido_lookup = (
        df_sos.groupby("alcalde_ganador")["siglas_ganadora"]
        .agg(lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else "N/D")
        .to_dict()
    )

    # ---------- helpers ----------
    def top_count_and_ratio(df_in, group_col, flag_col, min_projects=1):
        base = (
            df_in.groupby(group_col)
            .agg(
                casos=(flag_col, "sum"),
                total_proyectos=("snip", "nunique")
            )
            .reset_index()
        )
        base = base[base["total_proyectos"] >= min_projects].copy()
        base["ratio"] = base["casos"] / base["total_proyectos"]
        base = base.sort_values(["casos", "ratio"], ascending=[False, False])
        return base

    # ---------- top sospechosos ----------
    top_alcalde_sos_text = "N/D"
    top_municipio_sos_text = "N/D"
    top_proveedor_sos_text = "N/D"

    top_alcalde_sos = top_count_and_ratio(df_sos, "alcalde_ganador", "sospechoso", min_projects=1)
    if not top_alcalde_sos.empty and top_alcalde_sos.iloc[0]["casos"] > 0:
        row = top_alcalde_sos.iloc[0]
        partido = partido_lookup.get(row["alcalde_ganador"], "N/D")
        top_alcalde_sos_text = (
            f"{row['alcalde_ganador']} ({partido}, "
            f"{row['casos']:,.0f} / {row['ratio']:.2%})"
        )

    top_municipio_sos = top_count_and_ratio(df_sos, "municipio", "sospechoso", min_projects=1)
    if not top_municipio_sos.empty and top_municipio_sos.iloc[0]["casos"] > 0:
        row = top_municipio_sos.iloc[0]
        top_municipio_sos_text = f"{row['municipio']} ({row['casos']:,.0f} / {row['ratio']:.2%})"

    top_proveedor_sos = top_count_and_ratio(df_sos, "proveedor", "sospechoso", min_projects=1)
    if not top_proveedor_sos.empty and top_proveedor_sos.iloc[0]["casos"] > 0:
        row = top_proveedor_sos.iloc[0]
        top_proveedor_sos_text = f"{row['proveedor']} ({row['casos']:,.0f} / {row['ratio']:.2%})"

    # ---------- top meta_ejecutada = 0 con gasto ----------
    top_alcalde_meta0_text = "N/D"
    top_municipio_meta0_text = "N/D"
    top_proveedor_meta0_text = "N/D"

    top_alcalde_meta0 = top_count_and_ratio(df_sos, "alcalde_ganador", "sin_meta_ejecutada_con_gasto", min_projects=1)
    if not top_alcalde_meta0.empty and top_alcalde_meta0.iloc[0]["casos"] > 0:
        row = top_alcalde_meta0.iloc[0]
        partido = partido_lookup.get(row["alcalde_ganador"], "N/D")
        top_alcalde_meta0_text = (
            f"{row['alcalde_ganador']} ({partido}, "
            f"{row['casos']:,.0f} / {row['ratio']:.2%})"
        )

    top_municipio_meta0 = top_count_and_ratio(df_sos, "municipio", "sin_meta_ejecutada_con_gasto", min_projects=1)
    if not top_municipio_meta0.empty and top_municipio_meta0.iloc[0]["casos"] > 0:
        row = top_municipio_meta0.iloc[0]
        top_municipio_meta0_text = f"{row['municipio']} ({row['casos']:,.0f} / {row['ratio']:.2%})"

    top_proveedor_meta0 = top_count_and_ratio(df_sos, "proveedor", "sin_meta_ejecutada_con_gasto", min_projects=1)
    if not top_proveedor_meta0.empty and top_proveedor_meta0.iloc[0]["casos"] > 0:
        row = top_proveedor_meta0.iloc[0]
        top_proveedor_meta0_text = f"{row['proveedor']} ({row['casos']:,.0f} / {row['ratio']:.2%})"

    st.markdown(f"- **Alcalde con más proyectos sospechosos:** {top_alcalde_sos_text}")
    st.markdown(f"- **Municipio con más proyectos sospechosos:** {top_municipio_sos_text}")
    st.markdown(f"- **Proveedor con más proyectos sospechosos:** {top_proveedor_sos_text}")
    st.markdown(f"- **Alcalde con más proyectos donde hay ejecución de dinero pero meta ejecutada = 0:** {top_alcalde_meta0_text}")
    st.markdown(f"- **Municipio con más proyectos donde hay ejecución de dinero pero meta ejecutada = 0:** {top_municipio_meta0_text}")
    st.markdown(f"- **Proveedor con más proyectos donde hay ejecución de dinero pero meta ejecutada = 0:** {top_proveedor_meta0_text}")

    st.subheader("Mapas de riesgo")

    modo_mapa = st.radio(
        "Mostrar mapas como:",
        ["Número", "Porcentaje"],
        horizontal=True
    )
    df_sospechosos = df_sos[df_sos["sospechoso"] == 1].copy()
    
    # -------------------------
    # MAPA 1: PROYECTOS SOSPECHOSOS
    # -------------------------
    st.subheader("Mapa de proyectos sospechosos")

    if not df_sospechosos.empty:
        total_muni = (
            df_sos.groupby("municipio")["snip"]
            .nunique()
            .reset_index(name="total_proyectos")
        )

        mapa_sos = (
            df_sospechosos.groupby("municipio")["snip"]
            .nunique()
            .reset_index(name="num_sospechosos")
        )

        mapa_sos = mapa_sos.merge(total_muni, on="municipio", how="left")
        mapa_sos["pct_sospechosos"] = mapa_sos["num_sospechosos"] / mapa_sos["total_proyectos"]
        mapa_sos["municipio"] = mapa_sos["municipio"].astype(str).str.strip().str.upper()

        color_col = "num_sospechosos" if modo_mapa == "Número" else "pct_sospechosos"
        colorbar_title = "Proyectos sospechosos" if modo_mapa == "Número" else "% proyectos sospechosos"

        fig = px.choropleth_mapbox(
            mapa_sos,
            geojson=geojson_munis,
            locations="municipio",
            featureidkey="properties.name",
            color=color_col,
            color_continuous_scale="Reds",
            mapbox_style="carto-positron",
            zoom=6,
            center={"lat": 15.5, "lon": -90.3},
            opacity=0.65,
            hover_name="municipio",
            hover_data={
                "num_sospechosos": True,
                "pct_sospechosos": ":.2%",
                "total_proyectos": True,
            },
            height=550
        )

        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title=colorbar_title)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No existen proyectos sospechosos para mostrar en el mapa.")

    
    # -------------------------
    # MAPA 2: NÚMERO DE PROVEEDORES SOSPECHOSOS POR MUNICIPIO
    # -------------------------
    st.subheader("Mapa de proveedores sospechosos por municipio")

    if not mapa_prov_sos.empty:
        total_prov_muni = (
            df_sos.groupby("municipio")["proveedor"]
            .nunique()
            .reset_index(name="total_proveedores")
        )

        mapa_prov_sos = mapa_prov_sos.merge(total_prov_muni, on="municipio", how="left")
        mapa_prov_sos["pct_proveedores_sospechosos"] = (
            mapa_prov_sos["num_proveedores_sospechosos"] / mapa_prov_sos["total_proveedores"]
        )
        mapa_prov_sos["municipio"] = mapa_prov_sos["municipio"].astype(str).str.strip().str.upper()

        color_col = "num_proveedores_sospechosos" if modo_mapa == "Número" else "pct_proveedores_sospechosos"
        colorbar_title = "Proveedores sospechosos" if modo_mapa == "Número" else "% proveedores sospechosos"

        fig = px.choropleth_mapbox(
            mapa_prov_sos,
            geojson=geojson_munis,
            locations="municipio",
            featureidkey="properties.name",
            color=color_col,
            color_continuous_scale="Reds",
            mapbox_style="carto-positron",
            zoom=6,
            center={"lat": 15.5, "lon": -90.3},
            opacity=0.65,
            hover_name="municipio",
            hover_data={
                "num_proveedores_sospechosos": True,
                "pct_proveedores_sospechosos": ":.2%",
                "total_proveedores": True,
            },
            height=550
        )

        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title=colorbar_title)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No existen proveedores sospechosos para mostrar en el mapa.")

    # -------------------------
    # MAPA 3: meta ejecutada = 0 pero con gasto
    # -------------------------
    st.subheader("Mapa de proyectos con ejecución de dinero pero meta ejecutada = 0")

    df_meta0 = df_sos[df_sos["sin_meta_ejecutada_con_gasto"] == 1].copy()

    if not df_meta0.empty:
        total_muni = (
            df_sos.groupby("municipio")["snip"]
            .nunique()
            .reset_index(name="total_proyectos")
        )

        mapa_meta0 = (
            df_meta0.groupby("municipio")["snip"]
            .nunique()
            .reset_index(name="num_meta0_con_gasto")
        )

        mapa_meta0 = mapa_meta0.merge(total_muni, on="municipio", how="left")
        mapa_meta0["pct_meta0_con_gasto"] = mapa_meta0["num_meta0_con_gasto"] / mapa_meta0["total_proyectos"]
        mapa_meta0["municipio"] = mapa_meta0["municipio"].astype(str).str.strip().str.upper()

        color_col = "num_meta0_con_gasto" if modo_mapa == "Número" else "pct_meta0_con_gasto"
        colorbar_title = "Meta=0 con gasto" if modo_mapa == "Número" else "% meta=0 con gasto"

        fig = px.choropleth_mapbox(
            mapa_meta0,
            geojson=geojson_munis,
            locations="municipio",
            featureidkey="properties.name",
            color=color_col,
            color_continuous_scale="Reds",
            mapbox_style="carto-positron",
            zoom=6,
            center={"lat": 15.5, "lon": -90.3},
            opacity=0.65,
            hover_name="municipio",
            hover_data={
                "num_meta0_con_gasto": True,
                "pct_meta0_con_gasto": ":.2%",
                "total_proyectos": True,
            },
            height=550
        )

        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title=colorbar_title)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No existen proyectos con meta ejecutada = 0 y gasto para mostrar en el mapa.")
    # -------------------------
    # TABLA
    # -------------------------
    # -------------------------
    # KPIs DE RIESGO
    # -------------------------
    st.subheader("Indicadores de proyectos de riesgo")

    total_proyectos_tab = df_sos["snip"].nunique() if "snip" in df_sos.columns else len(df_sos)

    pct_sospechosos = (
        df_sos[df_sos["sospechoso"] == 1]["snip"].nunique() / total_proyectos_tab
        if total_proyectos_tab > 0 else np.nan
    )

    pct_meta0_gasto = (
        df_sos[df_sos["sin_meta_ejecutada_con_gasto"] == 1]["snip"].nunique() / total_proyectos_tab
        if total_proyectos_tab > 0 else np.nan
    )

    pct_sobreejecucion = (
        df_sos[df_sos["sobreejecucion_financiera"] == 1]["snip"].nunique() / total_proyectos_tab
        if total_proyectos_tab > 0 else np.nan
    )
    proyectos_con_tres_flags = (
    df_sos[
        (df_sos["sospechoso"] == 1) &
        (df_sos["sin_meta_ejecutada_con_gasto"] == 1) &
        (df_sos["sobreejecucion_financiera"] == 1)
    ]["snip"].nunique()
    if total_proyectos_tab > 0 else 0
)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "% proyectos sospechosos",
        f"{pct_sospechosos:.2%}" if pd.notna(pct_sospechosos) else ""
    )
    k2.metric(
        "% con meta ejecutada = 0 y gasto",
        f"{pct_meta0_gasto:.2%}" if pd.notna(pct_meta0_gasto) else ""
    )
    k3.metric(
        "% con sobreejecución financiera",
        f"{pct_sobreejecucion:.2%}" if pd.notna(pct_sobreejecucion) else ""
    )
    k4.metric(
        "Proyectos con las 3 señales",
        f"{proyectos_con_tres_flags:,.0f}"
    )

    st.subheader("Detalle de proyectos de riesgo")

    # solo proyectos que cumplan alguna de las 3 condiciones
    df_riesgo = df_sos[
        (df_sos["sin_meta_ejecutada_con_gasto"] == 1) |
        (df_sos["sospechoso"] == 1) |
        (df_sos["sobreejecucion_financiera"] == 1)
    ].copy()

    cols = [
        col for col in [
            "proyecto",
            "proveedor",
            "municipio",
            "departamento",
            "alcalde_ganador",
            "año",
            "monto_adjudicado",
            "monto_ejecutado",
            "brecha_adjudicado_ejecutado",
            "meta_ejecutada",
            "ratio_meta_ejecucion",
            "sin_meta_ejecutada_con_gasto",
            "sobreejecucion_financiera",
            "sospechoso",
        ] if col in (
            list(df_riesgo.columns) + ["brecha_adjudicado_ejecutado"]
        )
    ]

    # crear brecha para la tabla
    df_riesgo["brecha_adjudicado_ejecutado"] = np.where(
        df_riesgo["monto_adjudicado"].notna(),
        df_riesgo["monto_adjudicado"] - df_riesgo["monto_ejecutado"],
        np.nan
    )

    tabla = df_riesgo[
        [
            "proyecto",
            "proveedor",
            "municipio",
            "departamento",
            "alcalde_ganador",
            "año",
            "monto_adjudicado",
            "monto_ejecutado",
            "brecha_adjudicado_ejecutado",
            "meta_ejecutada",
            "ratio_meta_ejecucion",
            "sin_meta_ejecutada_con_gasto",
            "sobreejecucion_financiera",
            "sospechoso",
        ]
    ].copy()

    tabla = tabla.rename(columns={
        "alcalde_ganador": "alcalde",
        "ratio_meta_ejecucion": "ratio_meta_ejecutada",
    })

    # ordenar: primero meta=0 con gasto, luego sospechosos, luego sobreejecución, luego mayor monto
    tabla["orden_meta0"] = np.where(tabla["sin_meta_ejecutada_con_gasto"] == 1, 0, 1)
    tabla["orden_sos"] = np.where(tabla["sospechoso"] == 1, 0, 1)
    tabla["orden_sobre"] = np.where(tabla["sobreejecucion_financiera"] == 1, 0, 1)

    tabla = tabla.sort_values(
        by=["orden_meta0", "orden_sos", "orden_sobre", "monto_ejecutado"],
        ascending=[True, True, True, False]
    ).drop(columns=["orden_meta0", "orden_sos", "orden_sobre"])

    tabla["sin_meta_ejecutada_con_gasto"] = np.where(tabla["sin_meta_ejecutada_con_gasto"] == 1, "Sí", "No")
    tabla["sobreejecucion_financiera"] = np.where(tabla["sobreejecucion_financiera"] == 1, "Sí", "No")
    tabla["sospechoso"] = np.where(tabla["sospechoso"] == 1, "Sí", "No")

    tabla_display = tabla.copy()

    for col in ["monto_adjudicado", "monto_ejecutado", "brecha_adjudicado_ejecutado", "meta_ejecutada"]:
        if col in tabla_display.columns:
            tabla_display[col] = tabla_display[col].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")

    if "ratio_meta_ejecutada" in tabla_display.columns:
        tabla_display["ratio_meta_ejecutada"] = tabla_display["ratio_meta_ejecutada"].map(
            lambda x: f"{x:.2%}" if pd.notna(x) else ""
        )

    def highlight_riesgo(row):
        if row["sin_meta_ejecutada_con_gasto"] == "Sí":
            return ["background-color: #fde2e2"] * len(row)
        if row["sospechoso"] == "Sí":
            return ["background-color: #f8d7da"] * len(row)
        if row["sobreejecucion_financiera"] == "Sí":
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    styled_df = (
        tabla_display.style
        .apply(highlight_riesgo, axis=1)
    )

    st.dataframe(styled_df, use_container_width=True)

    st.caption(
        "La tabla incluye únicamente proyectos que presentan al menos una señal de riesgo: "
        "meta ejecutada igual a cero con gasto ejecutado, proyecto sospechoso o sobreejecución financiera. "
        "Rojo intenso: meta ejecutada = 0 con gasto. Rojo claro: proyecto sospechoso. Amarillo: sobreejecución financiera."
    )

# -------------------------
# TAB 4
# -------------------------

with tab4:
    st.subheader("Hallazgos clave")

    df_partidos_insights = df_filtered.copy()


    df_partidos_insights["diferencia"] = (
        df_partidos_insights["monto_adjudicado"] - df_partidos_insights["monto_ejecutado"]
    )

    df_partidos_insights["sospechoso"] = np.where(
        (df_partidos_insights["diferencia"].abs() < 1000) &
        (df_partidos_insights["ratio_meta_ejecucion"] < 0.90),
        1,
        0
    )

    # base agregada
    agg_partido = (
        df_partidos_insights.groupby("siglas_ganadora")
        .agg(
            monto_ejecutado=("monto_ejecutado", "sum"),
            proyectos=("snip", "nunique"),
            alcaldes=("alcalde_ganador", "nunique"),
            ratio_meta=("ratio_meta_ejecucion", "mean"),
            sospechosos=("sospechoso", "sum"),
        )
    )

    agg_partido["ratio_sospechosos"] = agg_partido["sospechosos"] / agg_partido["proyectos"]

    # 1) Mayor monto ejecutado
    top_monto = agg_partido.sort_values("monto_ejecutado", ascending=False).iloc[0]
    top_monto_text = f"{top_monto.name} (Q {top_monto['monto_ejecutado']:,.0f} — {top_monto['ratio_meta']:.2%})"

    # 2) Partido con más alcaldes (solo periodo 2023, únicos)
    top_alcaldes_text = "N/D"

    df_2023 = df_partidos_insights[
        df_partidos_insights["periodo_alcalde"] == 2023
    ].copy()

    alcaldes_2023 = (
        df_2023[["siglas_ganadora", "alcalde_ganador"]]
        .dropna()
        .drop_duplicates(subset=["siglas_ganadora", "alcalde_ganador"])
    )

    total_alcaldes_2023 = alcaldes_2023["alcalde_ganador"].nunique()

    alcaldes_2023_partido = (
        alcaldes_2023.groupby("siglas_ganadora")
        .size()
        .reset_index(name="alcaldes")
        .sort_values("alcaldes", ascending=False)
    )

    if not alcaldes_2023_partido.empty and total_alcaldes_2023 > 0:
        top = alcaldes_2023_partido.iloc[0]
        pct_alcaldes = top["alcaldes"] / total_alcaldes_2023

        top_alcaldes_text = (
            f"{top['siglas_ganadora']} "
            f"({top['alcaldes']:,.0f} alcaldes — {pct_alcaldes:.2%})"
        )

    # 3) Más proyectos
    top_proyectos = agg_partido.sort_values("proyectos", ascending=False).iloc[0]
    top_proyectos_text = f"{top_proyectos.name} (Q {top_proyectos['monto_ejecutado']:,.0f} — {top_proyectos['ratio_meta']:.2%})"

    # 4) Peor ejecución ejecutada
    worst_ratio = agg_partido.sort_values("ratio_meta", ascending=True).iloc[0]
    worst_ratio_text = f"{worst_ratio.name} (Q {worst_ratio['monto_ejecutado']:,.0f} — {worst_ratio['ratio_meta']:.2%})"

    # 5) Mayor proporción de sospechosos
    ratio_sos = agg_partido[agg_partido["proyectos"] >= 3].sort_values("ratio_sospechosos", ascending=False)

    top_sos_text = "N/D"
    if not ratio_sos.empty:
        top_sos = ratio_sos.iloc[0]
        top_sos_text = f"{top_sos.name} (Q {top_sos['monto_ejecutado']:,.0f} — {top_sos['ratio_sospechosos']:.2%})"

    # mostrar
    st.markdown(f"- **Partido con mayor monto ejecutado:** {top_monto_text}")
    st.markdown(f"- **Partido con mayor número de alcaldes (2023):** {top_alcaldes_text}")
    st.markdown(f"- **Partido con mayor número de proyectos:** {top_proyectos_text}")
    st.markdown(f"- **Partido con peor ejecución física:** {worst_ratio_text}")
    st.markdown(f"- **Partido con mayor proporción de proyectos sospechosos:** {top_sos_text}")

    df_partidos = df_filtered.copy()

    # -------------------------
    # FLAG SOSPECHOSO (reutilizamos lógica)
    # -------------------------
    df_partidos["diferencia"] = df_partidos["monto_adjudicado"] - df_partidos["monto_ejecutado"]

    df_partidos["sospechoso"] = np.where(
        (df_partidos["diferencia"].abs() < 1000) &
        (df_partidos["ratio_meta_ejecucion"] < 0.90),
        "Sí",
        "No"
    )

    df_partidos["sospechoso_flag"] = np.where(
        (df_partidos["monto_adjudicado"].notna()) &
        (df_partidos["diferencia"].abs() < 1000) &
        (df_partidos["ratio_meta_ejecucion"] < 0.90),
        1,
        0
    )

    df_partidos["meta_ejecutada_0_con_gasto"] = np.where(
        (df_partidos["meta_ejecutada"] == 0) &
        (df_partidos["monto_ejecutado"] > 0),
        1,
        0
    )

    df_partidos["sobreejecucion_financiera"] = np.where(
        (df_partidos["monto_adjudicado"].notna()) &
        (df_partidos["monto_ejecutado"] > df_partidos["monto_adjudicado"]),
        1,
        0
    )

    # -------------------------
    # TABLA AGREGADA POR PARTIDO
    # -------------------------
    st.subheader("Resumen por partido")

    partidos_stats = (
    df_partidos.groupby("siglas_ganadora")
    .agg(
        proyectos=("snip", "nunique"),
        alcaldes=("alcalde_ganador", "nunique"),
        proveedores=("proveedor", "nunique"),
        monto_adjudicado=("monto_adjudicado", "sum"),
        monto_ejecutado=("monto_ejecutado", "sum"),
        ratio_promedio=("ratio_meta_ejecucion", "mean"),
        proyectos_sospechosos=("sospechoso_flag", "sum"),
        proyectos_meta0_gasto=("meta_ejecutada_0_con_gasto", "sum"),
        proyectos_sobreejecucion=("sobreejecucion_financiera", "sum"),
    )
    .reset_index()
    .sort_values("monto_ejecutado", ascending=False)
)

    partidos_stats["pct_sospechosos"] = partidos_stats["proyectos_sospechosos"] / partidos_stats["proyectos"]
    partidos_stats["pct_meta0_gasto"] = partidos_stats["proyectos_meta0_gasto"] / partidos_stats["proyectos"]
    partidos_stats["pct_sobreejecucion"] = partidos_stats["proyectos_sobreejecucion"] / partidos_stats["proyectos"]

    partidos_display = partidos_stats.copy()

    partidos_display["monto_adjudicado"] = pd.to_numeric(partidos_display["monto_adjudicado"], errors="coerce")
    partidos_display["monto_ejecutado"] = pd.to_numeric(partidos_display["monto_ejecutado"], errors="coerce")
    partidos_display["ratio_promedio"] = partidos_display["ratio_promedio"] * 100
    partidos_display["pct_sospechosos"] = partidos_display["pct_sospechosos"] * 100
    partidos_display["pct_meta0_gasto"] = partidos_display["pct_meta0_gasto"] * 100
    partidos_display["pct_sobreejecucion"] = partidos_display["pct_sobreejecucion"] * 100

    styled_partidos_display = (
            partidos_display.style
            .format(
            {
                "monto_adjudicado": "{:,.2f}",
                "monto_ejecutado": "{:,.2f}",
            }
        ))
        

    st.dataframe(styled_partidos_display, use_container_width=True, 
                 column_config={
            "proyectos": st.column_config.NumberColumn("Proyectos",format="%.0f"), 
            "alcaldes": st.column_config.NumberColumn("Alcaldes",format="%.0f"), 
            "proveedores": st.column_config.NumberColumn("Proveedores",format="%.0f"), 
            "ratio_promedio": st.column_config.NumberColumn("Ratio Promedio",format="%.2f%%"), 
            "proyectos_sospechosos": st.column_config.NumberColumn("Proyectos Sospechosos",format="%.0f"), 
            "proyectos_meta0_gasto": st.column_config.NumberColumn("Proyectos con gastos pero sin avance",format="%.0f"), 
            "proyectos_sobreejecucion": st.column_config.NumberColumn("Proyectos con Sobreejecucion",format="%.0f"), 
            "pct_sospechosos": st.column_config.NumberColumn("% Proyectos Sospechosose",format="%.2f%%"), 
            "pct_meta0_gasto": st.column_config.NumberColumn("% Proyectos con gastos pero sin avance",format="%.2f%%"), 
            "pct_sobreejecucion": st.column_config.NumberColumn("% Proyectos con Sobreejecucion",format="%.2f%%"), 

            })

    # -------------------------
    # SCATTER
    # -------------------------
    st.subheader("Proyectos vs monto ejecutado por partido")

    fig = px.scatter(
        partidos_stats,
        x="proyectos",
        y="monto_ejecutado",
        size="monto_adjudicado",
        hover_data={
            "siglas_ganadora": True,
            "alcaldes": True,
            "ratio_promedio": ":.2%",
        },
        title="Actividad vs ejecución por partido"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # BUSCADOR
    # -------------------------
    st.subheader("Buscar partido")

    partidos = sorted(df_partidos["siglas_ganadora"].dropna().unique().tolist())
    partido_sel = st.selectbox("Selecciona un partido", [""] + partidos)

    if partido_sel:
        df_p = df_partidos[df_partidos["siglas_ganadora"] == partido_sel].copy()

        # flags
        df_p["diferencia"] = df_p["monto_adjudicado"] - df_p["monto_ejecutado"]

        df_p["sospechoso"] = np.where(
            (df_p["monto_adjudicado"].notna()) &
            (df_p["diferencia"].abs() < 1000) &
            (df_p["ratio_meta_ejecucion"] < 0.90),
            "Sí",
            "No"
        )

        df_p["meta_ejecutada_0_con_gasto"] = np.where(
            (df_p["meta_ejecutada"] == 0) &
            (df_p["monto_ejecutado"] > 0),
            "Sí",
            "No"
        )

        df_p["sobreejecucion_financiera"] = np.where(
            (df_p["monto_adjudicado"].notna()) &
            (df_p["monto_ejecutado"] > df_p["monto_adjudicado"]),
            "Sí",
            "No"
        )

        total_proveedores = df_p["proveedor"].nunique()
        monto_total = df_p["monto_ejecutado"].sum()
        ratio_prom = df_p["ratio_meta_ejecucion"].mean()
        total_proyectos = df_p["snip"].nunique()

        sospechosos = df_p[df_p["sospechoso"] == "Sí"]["snip"].nunique()
        meta0_gasto = df_p[df_p["meta_ejecutada_0_con_gasto"] == "Sí"]["snip"].nunique()
        sobreejecucion = df_p[df_p["sobreejecucion_financiera"] == "Sí"]["snip"].nunique()

        # KPIs fila 1
        k1, k2, k3 = st.columns(3)
        k1.metric("Número de proveedores", f"{total_proveedores:,.0f}")
        k2.metric("Monto total ejecutado (Q)", f"{monto_total:,.0f}")
        k3.metric("Promedio ratio meta ejecutada", f"{ratio_prom:.2%}" if pd.notna(ratio_prom) else "")

        # KPIs fila 2
        k4, k5, k6, k7 = st.columns(4)
        k4.metric("Número de proyectos", f"{total_proyectos:,.0f}")
        k5.metric("Proyectos sospechosos", f"{sospechosos:,.0f}")
        k6.metric("Meta ejecutada = 0 con gasto", f"{meta0_gasto:,.0f}")
        k7.metric("Proyectos con sobreejecución", f"{sobreejecucion:,.0f}")

        # -------------------------
        # TABLA DETALLE
        # -------------------------
        detalle = df_p[
            [
                "proyecto",
                "municipio",
                "departamento",
                "alcalde_ganador",
                "proveedor",
                "monto_adjudicado",
                "monto_ejecutado",
                "ratio_meta_ejecucion",
                "meta_ejecutada_0_con_gasto",
                "sobreejecucion_financiera",
                "sospechoso",
            ]
        ].copy()

        detalle = detalle.rename(columns={
            "ratio_meta_ejecucion": "ratio_meta_ejecutada",
            "alcalde_ganador": "alcalde"
        })

        # ordenar: primero meta ejecutada=0 con gasto, luego sospechosos, luego sobreejecución
        detalle["orden_meta0"] = np.where(detalle["meta_ejecutada_0_con_gasto"] == "Sí", 0, 1)
        detalle["orden_sos"] = np.where(detalle["sospechoso"] == "Sí", 0, 1)
        detalle["orden_sobre"] = np.where(detalle["sobreejecucion_financiera"] == "Sí", 0, 1)

        detalle = detalle.sort_values(
            by=["orden_meta0", "orden_sos", "orden_sobre", "monto_ejecutado"],
            ascending=[True, True, True, False]
        ).drop(columns=["orden_meta0", "orden_sos", "orden_sobre"])

        # formato

        def highlight(row):
            if row["meta_ejecutada_0_con_gasto"] == "Sí":
                return ["background-color: #fde2e2"] * len(row)
            if row["sospechoso"] == "Sí":
                return ["background-color: #f8d7da"] * len(row)
            if row["sobreejecucion_financiera"] == "Sí":
                return ["background-color: #fff3cd"] * len(row)
            return [""] * len(row)
        
        detalle["proveedor"] = detalle["proveedor"].apply(determinar_tipo)

        detalle["alcalde"] = detalle["alcalde"].apply(
        lambda x: (
            f"https://firmaconcerteza.com/dashboard/search?query={quote(str(x))}"
            f"&type={'natural'}&page=1&pageSize=10"
            if pd.notna(x) else ""
        ))
        detalle["monto_adjudicado"] = pd.to_numeric(detalle["monto_adjudicado"], errors="coerce")
        detalle["monto_ejecutado"] = pd.to_numeric(detalle["monto_ejecutado"], errors="coerce")
        detalle["ratio_meta_ejecutada"] = detalle["ratio_meta_ejecutada"] * 100

        styled = (
            detalle.style
            .apply(highlight, axis=1)
            .hide(axis="columns", subset=["sospechoso", "meta_ejecutada_0_con_gasto", "sobreejecucion_financiera"])
            .format(
            {
                "monto_adjudicado": "{:,.2f}",
                "monto_ejecutado": "{:,.2f}",
            }
        ))

        st.dataframe(styled, use_container_width=True, 
                     column_config={
         "proveedor": st.column_config.LinkColumn(
            "Proveedor",
             display_text=r"query=(.*?)&"
        ),
         "alcalde": st.column_config.LinkColumn(
            "Alcalde",
             display_text=r"query=(.*?)&"
        ),
        "ratio_meta_ejecutada": st.column_config.NumberColumn("Ratio Promedio",format="%.2f%%"), })

        st.subheader("Evolución: alcaldes vs ejecución por año")

        # usar SOLO el partido seleccionado
        df_timeline = df_partidos[df_partidos["siglas_ganadora"] == partido_sel].copy()

        # filtrar años según periodo correcto
        df_timeline = df_timeline[
            (
                (df_timeline["periodo_alcalde"] == 2015) &
                (df_timeline["año"].between(2016, 2018))
            )
            |
            (
                (df_timeline["periodo_alcalde"] == 2019) &
                (df_timeline["año"].between(2019, 2022))
            )
            |
            (
                (df_timeline["periodo_alcalde"] == 2023) &
                (df_timeline["año"].between(2023, 2026))
            )
        ].copy()

        alcaldes_por_anio = (
            df_timeline[["año", "alcalde_ganador"]]
            .dropna()
            .drop_duplicates()
            .groupby("año")
            .size()
            .reset_index(name="alcaldes_unicos")
        )

        monto_por_anio = (
            df_timeline.groupby("año")["monto_ejecutado"]
            .sum()
            .reset_index(name="monto_ejecutado")
        )

        timeline = (
            alcaldes_por_anio
            .merge(monto_por_anio, on="año", how="left")
            .sort_values("año")
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=timeline["año"],
                y=timeline["alcaldes_unicos"],
                name="Alcaldes únicos",
                yaxis="y1"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=timeline["año"],
                y=timeline["monto_ejecutado"],
                name="Monto ejecutado",
                yaxis="y2",
                mode="lines+markers"
            )
        )

        fig.update_layout(
            title=f"Alcaldes vs ejecución — {partido_sel}",
            xaxis=dict(title="Año"),
            yaxis=dict(title="Número de alcaldes"),
            yaxis2=dict(
                title="Monto ejecutado",
                overlaying="y",
                side="right"
            ),
            legend=dict(x=0, y=1.1, orientation="h")
        )

        st.plotly_chart(fig, use_container_width=True)


# -------------------------
# TAB 5 - PROVEEDORES
# -------------------------

with tab5:
    st.subheader("Hallazgos clave")

    df_tmp = df_filtered.copy()
    df_tmp["diferencia"] = df_tmp["monto_adjudicado"] - df_tmp["monto_ejecutado"]

    df_tmp["sospechoso"] = np.where(
        (df_tmp["monto_adjudicado"].notna()) &
        (df_tmp["diferencia"].abs() < 1000) &
        (df_tmp["ratio_meta_ejecucion"] < 0.90),
        1,
        0
    )

    df_tmp["sin_ejecucion_fisica"] = np.where(
        (df_tmp["meta_ejecutada"] == 0) &
        (df_tmp["monto_ejecutado"] > 0),
        1,
        0
    )

    df_tmp["sobreejecucion_financiera"] = np.where(
        (df_tmp["monto_adjudicado"].notna()) &
        (df_tmp["monto_ejecutado"] > df_tmp["monto_adjudicado"]),
        1,
        0
    )

    # base de proveedores
    agg_prov = (
        df_tmp.groupby("proveedor")
        .agg(
            proyectos=("snip", "nunique"),
            municipios=("municipio", "nunique"),
            alcaldes=("alcalde_ganador", "nunique"),
            monto_total=("monto_ejecutado", "sum"),
            promedio_monto_ejecutado=("monto_ejecutado", "mean"),
            ratio_meta=("ratio_meta_ejecucion", "mean"),
        )
        .reset_index()
    )

    agg_prov = agg_prov[agg_prov["proveedor"].notna()].copy()

    # contar PROYECTOS ÚNICOS por tipo de riesgo
    sos_prov = (
        df_tmp[df_tmp["sospechoso"] == 1]
        .groupby("proveedor")["snip"]
        .nunique()
        .reset_index(name="proyectos_sospechosos")
    )

    meta0_prov = (
        df_tmp[df_tmp["sin_ejecucion_fisica"] == 1]
        .groupby("proveedor")["snip"]
        .nunique()
        .reset_index(name="casos_sin_ejecucion_fisica")
    )

    sobre_prov = (
        df_tmp[df_tmp["sobreejecucion_financiera"] == 1]
        .groupby("proveedor")["snip"]
        .nunique()
        .reset_index(name="casos_sobreejecucion")
    )

    # merge
    agg_prov = agg_prov.merge(sos_prov, on="proveedor", how="left")
    agg_prov = agg_prov.merge(meta0_prov, on="proveedor", how="left")
    agg_prov = agg_prov.merge(sobre_prov, on="proveedor", how="left")

    for col in ["proyectos_sospechosos", "casos_sin_ejecucion_fisica", "casos_sobreejecucion"]:
        agg_prov[col] = agg_prov[col].fillna(0)

    agg_prov["ratio_sospechosos"] = agg_prov["proyectos_sospechosos"] / agg_prov["proyectos"]
    agg_prov["ratio_sin_ejecucion_fisica"] = agg_prov["casos_sin_ejecucion_fisica"] / agg_prov["proyectos"]
    agg_prov["ratio_sobreejecucion"] = agg_prov["casos_sobreejecucion"] / agg_prov["proyectos"]

    # 1) Proveedor con mayor monto ejecutado
    top_monto_text = "N/D"
    if not agg_prov.empty:
        top_monto = agg_prov.sort_values("monto_total", ascending=False).iloc[0]
        top_monto_text = f"{top_monto['proveedor']} (Q {top_monto['monto_total']:,.0f})"

    # 2) Proveedor con mayor cobertura territorial
    top_municipios_text = "N/D"
    if not agg_prov.empty:
        top_mun = agg_prov.sort_values("municipios", ascending=False).iloc[0]
        top_municipios_text = f"{top_mun['proveedor']} ({top_mun['municipios']:,.0f} municipios)"

    # 3) Proveedor con peor ratio meta ejecutada
    worst_ratio_text = "N/D"
    ratio_prov = agg_prov.dropna(subset=["ratio_meta"]).sort_values("ratio_meta", ascending=True)
    if not ratio_prov.empty:
        worst_ratio = ratio_prov.iloc[0]
        worst_ratio_text = f"{worst_ratio['proveedor']} ({worst_ratio['ratio_meta']:.2%})"

    # 4) Proveedor con mayor proporción de proyectos sospechosos
    top_sos_text = "N/D"
    ratio_sos = agg_prov[agg_prov["proyectos"] >= 3].sort_values(
        ["proyectos_sospechosos", "ratio_sospechosos"], ascending=[False, False]
    )
    if not ratio_sos.empty:
        top_sos = ratio_sos.iloc[0]
        top_sos_text = (
            f"{top_sos['proveedor']} "
            f"({top_sos['proyectos_sospechosos']:,.0f} — {top_sos['ratio_sospechosos']:.2%})"
        )

    # 5) Proveedor con mayor proporción de proyectos con meta ejecutada = 0 y gasto
    top_meta0_text = "N/D"
    ratio_meta0 = agg_prov[agg_prov["proyectos"] >= 3].sort_values(
        ["casos_sin_ejecucion_fisica", "ratio_sin_ejecucion_fisica"], ascending=[False, False]
    )
    if not ratio_meta0.empty:
        top_meta0 = ratio_meta0.iloc[0]
        top_meta0_text = (
            f"{top_meta0['proveedor']} "
            f"({top_meta0['casos_sin_ejecucion_fisica']:,.0f} — {top_meta0['ratio_sin_ejecucion_fisica']:.2%})"
        )

    # 6) Proveedor con mayor proporción de proyectos con sobreejecución
    top_sobre_text = "N/D"
    ratio_sobre = agg_prov[agg_prov["proyectos"] >= 3].sort_values(
        ["casos_sobreejecucion", "ratio_sobreejecucion"], ascending=[False, False]
    )
    if not ratio_sobre.empty:
        top_sobre = ratio_sobre.iloc[0]
        top_sobre_text = (
            f"{top_sobre['proveedor']} "
            f"({top_sobre['casos_sobreejecucion']:,.0f} — {top_sobre['ratio_sobreejecucion']:.2%})"
        )

    st.markdown(f"- **Proveedor con mayor monto ejecutado:** {top_monto_text}")
    st.markdown(f"- **Proveedor con mayor cobertura territorial:** {top_municipios_text}")
    st.markdown(f"- **Proveedor con menor promedio de ratio meta ejecutada:** {worst_ratio_text}")
    st.markdown(f"- **Proveedor con mayor proporción de proyectos sospechosos:** {top_sos_text}")
    st.markdown(f"- **Proveedor con mayor proporción de proyectos con meta ejecutada = 0 y gasto:** {top_meta0_text}")
    st.markdown(f"- **Proveedor con mayor proporción de proyectos con sobreejecución:** {top_sobre_text}")

    st.subheader("Resumen de proveedores")

   # base principal por proveedor
    metricas_prov = (
        df_tmp.groupby("proveedor")
        .agg(
            proyectos=("snip", "nunique"),
            municipios=("municipio", "nunique"),
            alcaldes=("alcalde_ganador", "nunique"),
            monto_total_ejecutado=("monto_ejecutado", "sum"),
            promedio_monto_ejecutado=("monto_ejecutado", "mean"),
            promedio_ratio_meta_ejecutada=("ratio_meta_ejecucion", "mean"),
        )
        .reset_index()
    )

    metricas_prov = metricas_prov[metricas_prov["proveedor"].notna()].copy()

    # conteos de riesgo por SNIP único
    sos_prov = (
        df_tmp[df_tmp["sospechoso"] == 1]
        .groupby("proveedor")["snip"]
        .nunique()
        .reset_index(name="proyectos_sospechosos")
    )

    meta0_prov = (
        df_tmp[df_tmp["sin_ejecucion_fisica"] == 1]
        .groupby("proveedor")["snip"]
        .nunique()
        .reset_index(name="casos_sin_ejecucion_fisica")
    )

    sobre_prov = (
        df_tmp[df_tmp["sobreejecucion_financiera"] == 1]
        .groupby("proveedor")["snip"]
        .nunique()
        .reset_index(name="casos_sobreejecucion")
    )

    # merge
    metricas_prov = metricas_prov.merge(sos_prov, on="proveedor", how="left")
    metricas_prov = metricas_prov.merge(meta0_prov, on="proveedor", how="left")
    metricas_prov = metricas_prov.merge(sobre_prov, on="proveedor", how="left")

    # llenar nulos
    for col in ["proyectos_sospechosos", "casos_sin_ejecucion_fisica", "casos_sobreejecucion"]:
        metricas_prov[col] = metricas_prov[col].fillna(0).astype(int)

    # ordenar
    metricas_prov = metricas_prov.sort_values("monto_total_ejecutado", ascending=False)

    # display
    prov_display = metricas_prov.copy()
    prov_display["monto_total_ejecutado"] = pd.to_numeric(prov_display["monto_total_ejecutado"], errors="coerce")
    prov_display["promedio_monto_ejecutado"] = pd.to_numeric(prov_display["promedio_monto_ejecutado"], errors="coerce")
    prov_display["promedio_ratio_meta_ejecutada"] = prov_display["promedio_ratio_meta_ejecutada"] * 100
    prov_display["proveedor"] = prov_display["proveedor"].apply(determinar_tipo)

    styled_prov_display = (
            prov_display.style
            .format(
            {
                "monto_total_ejecutado": "{:,.2f}",
                "promedio_monto_ejecutado": "{:,.2f}",
            }
        ))
        

    st.dataframe(
        styled_prov_display,
        use_container_width=True,
        column_config={
            "proveedor": st.column_config.LinkColumn(
            "Proveedor",
             display_text=r"query=(.*?)&"
        ),
            "proyectos": st.column_config.NumberColumn(
                "Proyectos",
                format="%.0f"
            ),
            "municipios": st.column_config.NumberColumn(
                "Municipios",
                format="%.0f"
            ),
            "alcaldes": st.column_config.NumberColumn(
                "Alcaldes",
                format="%.0f"
            ),
            "promedio_ratio_meta_ejecutada": st.column_config.NumberColumn(
                "Promedio Ratio Meta Ejecutada",
                format="%.2f%%"
            ),
            "casos_sin_ejecucion_fisica": st.column_config.NumberColumn(
                "Casos meta ejecutada = 0 y gasto",
                format="%d"
            ),
            "casos_sobreejecucion": st.column_config.NumberColumn(
                "Casos con sobreejecución",
                format="%d"
            ),
            "proyectos_sospechosos": st.column_config.NumberColumn(
                "Proyectos sospechosos",
                format="%d"
            ),
        }
    )

    st.subheader("Buscar proveedor")
    proveedores_disponibles = sorted(df_filtered["proveedor"].dropna().unique().tolist())
    proveedor_sel = st.selectbox("Selecciona un proveedor", options=[""] + proveedores_disponibles)

    if proveedor_sel:
        detalle_proveedor = df_filtered[df_filtered["proveedor"] == proveedor_sel].copy()

        detalle_cols = [
        col for col in [
            "proyecto",
            "municipio",
            "departamento",
            "alcalde_ganador",
            "siglas_ganadora",
            "año",
            "monto_adjudicado",
            "monto_ejecutado",
            "ratio_meta_ejecucion",
            "meta_ejecutada",
        ] if col in detalle_proveedor.columns
    ]

        detalle_proveedor = detalle_proveedor[detalle_cols].copy()

        detalle_proveedor = detalle_proveedor.rename(
            columns={
                "alcalde_ganador": "alcalde",
                "siglas_ganadora": "partido",
                "ratio_meta_ejecucion": "ratio_meta_ejecutada",
            }
        )

        detalle_proveedor["diferencia_adjudicado_ejecutado"] = (
            detalle_proveedor["monto_adjudicado"] - detalle_proveedor["monto_ejecutado"]
        )

        detalle_proveedor["sospechoso"] = np.where(
            (detalle_proveedor["monto_adjudicado"].notna()) &
            (detalle_proveedor["diferencia_adjudicado_ejecutado"].abs() < 1000) &
            (detalle_proveedor["ratio_meta_ejecutada"] < 0.90),
            "Sí",
            "No"
        )

        detalle_proveedor["sin_ejecucion_fisica"] = np.where(
            (detalle_proveedor["meta_ejecutada"] == 0) &
            (detalle_proveedor["monto_ejecutado"] > 0),
            "Sí",
            "No"
        )

        detalle_proveedor["sobreejecucion_financiera"] = np.where(
            (detalle_proveedor["monto_adjudicado"].notna()) &
            (detalle_proveedor["monto_ejecutado"] > detalle_proveedor["monto_adjudicado"]),
            "Sí",
            "No"
        )

        pct_sospechosos = (
            (detalle_proveedor["sospechoso"] == "Sí").mean()
            if not detalle_proveedor.empty else np.nan
        )

        total_proyectos = detalle_proveedor["proyecto"].nunique() if "proyecto" in detalle_proveedor.columns else 0
        total_municipios = detalle_proveedor["municipio"].nunique() if "municipio" in detalle_proveedor.columns else 0
        total_alcaldes = detalle_proveedor["alcalde"].nunique() if "alcalde" in detalle_proveedor.columns else 0
        monto_total_ejecutado = detalle_proveedor["monto_ejecutado"].sum() if "monto_ejecutado" in detalle_proveedor.columns else 0
        ratio_promedio = detalle_proveedor["ratio_meta_ejecutada"].mean() if "ratio_meta_ejecutada" in detalle_proveedor.columns else np.nan
        proyectos_sin_ejecucion_fisica = (
            (detalle_proveedor["sin_ejecucion_fisica"] == "Sí").sum()
            if "sin_ejecucion_fisica" in detalle_proveedor.columns else 0
        )
        proyectos_sobreejecucion = (
            (detalle_proveedor["sobreejecucion_financiera"] == "Sí").sum()
            if "sobreejecucion_financiera" in detalle_proveedor.columns else 0
        )

        total_partidos = detalle_proveedor["partido"].dropna().nunique() if "partido" in detalle_proveedor.columns else 0

        min_year = detalle_proveedor["año"].min() if "año" in detalle_proveedor.columns else np.nan
        max_year = detalle_proveedor["año"].max() if "año" in detalle_proveedor.columns else np.nan

        anios_operacion = (
            int(max_year - min_year + 1)
            if pd.notna(min_year) and pd.notna(max_year) else 0
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Número de proyectos", f"{total_proyectos:,.0f}")
        k2.metric("Número de municipios", f"{total_municipios:,.0f}")
        k3.metric("% de proyectos sospechosos", f"{pct_sospechosos:.2%}" if pd.notna(pct_sospechosos) else "")
        k4.metric("Número de alcaldes", f"{total_alcaldes:,.0f}")

        k5, k6, k7,  = st.columns(3)
        k5.metric("Total monto ejecutado (Q)", f"{monto_total_ejecutado:,.0f}")
        k6.metric("Promedio ratio meta ejecutada", f"{ratio_promedio:.2%}" if pd.notna(ratio_promedio) else "")
        k7.metric("Meta ejecutada = 0 con gasto", f"{proyectos_sin_ejecucion_fisica:,.0f}")
        

        k8, k9, k10 = st.columns(3)
        k8.metric("Proyectos con sobreejecución", f"{proyectos_sobreejecucion:,.0f}")
        k9.metric("Partidos distintos", f"{total_partidos:,.0f}")
        k10.metric("Años de operación", f"{anios_operacion:,.0f}")

        detalle_proveedor["orden_meta0"] = np.where(detalle_proveedor["sin_ejecucion_fisica"] == "Sí", 0, 1)
        detalle_proveedor["orden_sospechoso"] = np.where(detalle_proveedor["sospechoso"] == "Sí", 0, 1)
        detalle_proveedor["orden_sobre"] = np.where(detalle_proveedor["sobreejecucion_financiera"] == "Sí", 0, 1)

        detalle_proveedor = detalle_proveedor.sort_values(
            by=["orden_meta0", "orden_sospechoso", "orden_sobre", "monto_ejecutado"],
            ascending=[True, True, True, False]
        ).drop(columns=["orden_meta0", "orden_sospechoso", "orden_sobre", "diferencia_adjudicado_ejecutado", "meta_ejecutada"])

        detalle_proveedor = detalle_proveedor[
        [
            "proyecto",
            "municipio",
            "departamento",
            "alcalde",
            "partido",
            "año",
            "monto_adjudicado",
            "monto_ejecutado",
            "ratio_meta_ejecutada",
            "sin_ejecucion_fisica",
            "sobreejecucion_financiera",
            "sospechoso",
        ]
        ].copy()

        detalle_display = detalle_proveedor.copy()

        def highlight_sospechoso(row):
            if row["sin_ejecucion_fisica"] == "Sí":
                return ["background-color: #fde2e2"] * len(row)
            if row["sospechoso"] == "Sí":
                return ["background-color: #f8d7da"] * len(row)
            if row["sobreejecucion_financiera"] == "Sí":
                return ["background-color: #fff3cd"] * len(row)
            return [""] * len(row)
        
        detalle_display["alcalde"] = detalle_display["alcalde"].apply(
        lambda x: (
            f"https://firmaconcerteza.com/dashboard/search?query={quote(str(x))}"
            f"&type={'natural'}&page=1&pageSize=10"
            if pd.notna(x) else ""
        ))


        detalle_display["monto_adjudicado"] = pd.to_numeric(detalle_display["monto_adjudicado"], errors="coerce")
        detalle_display["monto_ejecutado"] = pd.to_numeric(detalle_display["monto_ejecutado"], errors="coerce")
        detalle_display["ratio_meta_ejecutada"] = detalle_display["ratio_meta_ejecutada"] * 100

        styled_df = (
            detalle_display.style
            .apply(highlight_sospechoso, axis=1)
            .hide(axis="columns", subset=["sospechoso", "sin_ejecucion_fisica", "sobreejecucion_financiera"])
            .format(
                {
                    "monto_adjudicado": "{:,.2f}",
                    "monto_ejecutado": "{:,.2f}",
                }
            )
        )

        st.dataframe(styled_df, use_container_width=True, 
                     column_config={
         "alcalde": st.column_config.LinkColumn(
            "Alcalde",
             display_text=r"query=(.*?)&"
        ), "ratio_meta_ejecutada": st.column_config.NumberColumn(
                "Ratio Meta Ejecutada",
                format="%.2f%%")})

        st.caption(
            "Los proyectos resaltados muestran señales de riesgo. "
            "Rojo intenso: proyectos sospechosos. "
            "Rojo claro: meta ejecutada = 0 con gasto. "
            "Amarillo: sobreejecución financiera."
        )

        # -------------------------
        # LINE GRAPH: monto ejecutado por año y municipalidad
        # -------------------------

        # --- 1. GRÁFICO GENERAL AGREGADO (TOTAL PROVEEDOR) ---
        st.subheader(f"Evolución General de Contrataciones: {proveedor_sel}")

        # Agrupamos por año ignorando el municipio para ver el "Big Picture"
        df_total_anual = (
            detalle_proveedor.groupby(["año"])["monto_ejecutado"]
            .sum()
            .reset_index()
            .sort_values("año")
        )

        fig_general = px.line(
            df_total_anual,
            x="año",
            y="monto_ejecutado",
            markers=True,
            title="Monto Ejecutado por Año por Proveedor (Todos los municipios)",
            labels={"monto_ejecutado": "Monto Total (Q)", "año": "Año"},
            line_shape="linear", # Hace la línea un poco más curva y estética
            color_discrete_sequence=["#007BFF"] 
        )

        # Aplicar los fondos de gobierno también al general para consistencia
        gobiernos = [
            dict(nombre="G1", inicio=2016, fin=2020, color="rgba(200, 200, 200, 0.1)"),
            dict(nombre="G2", inicio=2020, fin=2024, color="rgba(0, 0, 255, 0.05)"),
            dict(nombre="G3", inicio=2024, fin=2027.5, color="rgba(255, 0, 0, 0.05)"),
        ]

        for gob in gobiernos:
            fig_general.add_vrect(
                x0=gob["inicio"], x1=gob["fin"],
                annotation_text=gob["nombre"], annotation_position="top left",
                fillcolor=gob["color"], opacity=1, layer="below", line_width=0
            )

        fig_general.update_layout(plot_bgcolor="white", hovermode="x unified")
        st.plotly_chart(fig_general, use_container_width=True)


        # --- 2. BOTÓN DE DETALLE (EXPANDER) ---
        # -------------------------
        # LINE GRAPH: monto ejecutado por año y municipalidad
        # -------------------------
        with st.expander("Ver detalle desglosado por Municipio y Alcalde"):
            # 1. Preparar los datos agregados
            df_faceta = (
            detalle_proveedor.groupby(["municipio", "año", "alcalde", "partido"])["monto_ejecutado"]
            .sum()
            .reset_index()
            .sort_values(["municipio", "año"])
            )
            df_faceta = df_faceta[df_faceta["monto_ejecutado"] > 0]

            # 2. Crear el gráfico de facetas (Small Multiples)
            # Usamos facet_col para que cada municipio tenga su propio cuadro
            fig = px.line(
            df_faceta,
            x="año",
            y="monto_ejecutado",
            facet_col="municipio",
            facet_col_wrap=1, # 4 gráficos por fila para que no sean tan pequeños
            markers=True,
            facet_row_spacing=0.01,
            category_orders={"año": sorted(df_faceta["año"].unique())},
            title=f"Evolución por Municipio y Gestión: {proveedor_sel}",
            labels={"monto_ejecutado": "Monto (Q)", "año": "Año"},
            # Agregamos alcalde y partido al hover
            hover_data={
            "alcalde": True,
            "partido": True,
            "monto_ejecutado": ":,.0f",
            "año": True
            }
            )

            # 3. Personalizar el diseño y agregar los fondos de Gobierno
            # Definimos los periodos
            gobiernos = [
            dict(nombre="G1", inicio=2016, fin=2020, color="rgba(200, 200, 200, 0.15)"),
            dict(nombre="G2", inicio=2020, fin=2024, color="rgba(0, 0, 255, 0.05)"),
            dict(nombre="G3", inicio=2024, fin=2027.5, color="rgba(255, 0, 0, 0.05)"),
            ]

            # Aplicar los fondos a cada sub-gráfico (facet)
            for gob in gobiernos:
                fig.add_vrect(
                x0=gob["inicio"], x1=gob["fin"],
                fillcolor=gob["color"], opacity=1,
                layer="below", line_width=0
                )

            # 4. Ajustes de formato para que sea legible
            fig.update_layout(
            height=300 * ((len(df_faceta["municipio"].unique()) // 4) + 1), # Altura dinámica
            showlegend=False,
            plot_bgcolor="white",
            margin=dict(t=100)
            )

            # Limpiar los títulos de las facetas (quitar "municipio=")
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

            # Hacer que los ejes Y sean independientes para que los municipios pequeños se aprecien
            fig.update_yaxes(matches=None, showgrid=True, gridcolor='lightgrey', tickformat=".2s")
            fig.update_xaxes(showgrid=False)

            st.plotly_chart(fig, use_container_width=True)

            
# -------------------------
# TAB 6
# -------------------------

with tab6:
    st.subheader("Hallazgos clave")
    df_cost = df_filtered.copy()

    # =========================
    # FILTRO UNIDAD
    # =========================
    unidades = sorted(df_cost["unidad"].dropna().unique().tolist())

    default_unit = "Metro" if "Metro" in unidades else unidades[0] if unidades else None

    unidad_sel = st.selectbox(
        "Selecciona la unidad",
        options=unidades,
        index=unidades.index(default_unit) if default_unit in unidades else 0
    )

    df_cost = df_cost[df_cost["unidad"] == unidad_sel].copy()

    if df_cost.empty:
        st.warning("No hay datos para la unidad seleccionada.")
        st.stop()

    # =========================
    # COSTO POR UNIDAD
    # =========================
    df_cost["costo_por_unidad"] = np.where(
        df_cost["meta_ejecutada"].fillna(0) > 0,
        df_cost["monto_ejecutado"] / df_cost["meta_ejecutada"],
        np.nan
    )

    df_cost = df_cost[df_cost["costo_por_unidad"].notna()]

    orden_deptos = (
    df_cost.groupby("departamento")["costo_por_unidad"]
    .mean()
    .sort_values(ascending=False)  # 👉 cambia a True si lo quieres de menor a mayor
    .index
    .tolist()
    )
    # 1) Proyecto más caro por unidad
    top_proyecto_text = "N/D"
    if not df_cost.empty:
        top_proyecto = df_cost.sort_values("costo_por_unidad", ascending=False).iloc[0]
        top_proyecto_text = (
            f"{top_proyecto['proyecto']} — {top_proyecto['proveedor']} — "
            f"{top_proyecto['municipio']} (Q {top_proyecto['costo_por_unidad']:,.2f})"
        )

    # 2) Proveedor con mayor costo promedio por unidad
    top_proveedor_text = "N/D"
    proveedores_insights = (
        df_cost.groupby("proveedor")
        .agg(
            proyectos=("snip", "nunique"),
            costo_promedio=("costo_por_unidad", "mean")
        )
        .reset_index()
    )

    proveedores_insights = proveedores_insights[proveedores_insights["proyectos"] >= 2]

    if not proveedores_insights.empty:
        top_proveedor = proveedores_insights.sort_values("costo_promedio", ascending=False).iloc[0]
        top_proveedor_text = (
            f"{top_proveedor['proveedor']} "
            f"(Q {top_proveedor['costo_promedio']:,.2f} — {top_proveedor['proyectos']:,.0f} proyectos)"
        )

    # 3) Alcalde con mayor costo promedio por unidad (NUEVO)
    top_alcalde_text = "N/D"
    alcaldes_insights = (
        df_cost.groupby(["alcalde_ganador", "siglas_ganadora", "municipio"]) # Agrupamos por alcalde y su municipio
        .agg(proyectos=("snip", "nunique"), costo_promedio=("costo_por_unidad", "mean"))
        .reset_index()
    )
    # Filtro para evitar sesgos de un solo proyecto
    alcaldes_insights = alcaldes_insights[alcaldes_insights["proyectos"] >= 2]

    if not alcaldes_insights.empty:
        top_a = alcaldes_insights.sort_values("costo_promedio", ascending=False).iloc[0]
        top_alcalde_text = (
            f"{top_a['alcalde_ganador']} — {top_a['municipio']} — {top_a['siglas_ganadora']} "
            f"(Q {top_a['costo_promedio']:,.2f} — {top_a['proyectos']:,.0f} proyectos)"
        )

    st.markdown(f"- **Proyecto con mayor costo por unidad:** {top_proyecto_text}")
    st.markdown(f"- **Proveedor con mayor costo promedio por unidad:** {top_proveedor_text}")
    st.markdown(f"- **Alcalde con mayor costo promedio por unidad:** {top_alcalde_text}")

    st.subheader("Análisis de costo por unidad ejecutada")

    # =========================
    # BOXPLOT POR DEPARTAMENTO
    # =========================
    st.subheader("Distribución de costo por unidad por departamento")

    fig = px.box(
        df_cost,
        x="departamento",
        y="costo_por_unidad",
        points="outliers",
        category_orders={"departamento": orden_deptos},
        title=f"Costo por unidad ({unidad_sel}) por departamento"
    )

    fig.update_yaxes(type="log")

    fig.update_layout(
        xaxis_title="Departamento",
        yaxis_title="Costo por unidad (escala log)",
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TABLA PROVEEDORES
    # =========================
    st.subheader("Proveedores con mayor costo promedio por unidad")

    proveedores_stats = (
        df_cost.groupby("proveedor")
        .agg(
            proyectos=("snip", "nunique"),
            costo_promedio=("costo_por_unidad", "mean"),
            costo_mediano=("costo_por_unidad", "median")
        )
        .reset_index()
        .sort_values("costo_promedio", ascending=False)
    )

    # opcional: filtrar ruido (muy recomendado)
    #proveedores_stats = proveedores_stats[proveedores_stats["proyectos"] >= 2]

    proveedores_display = proveedores_stats.copy()

    proveedores_display["costo_promedio"] = pd.to_numeric(proveedores_display["costo_promedio"], errors="coerce")
    proveedores_display["costo_mediano"] = pd.to_numeric(proveedores_display["costo_mediano"], errors="coerce")
    proveedores_display["proyectos"] = pd.to_numeric(proveedores_display["proyectos"], errors="coerce")

    proveedores_display["proveedor"] = proveedores_display["proveedor"].apply(determinar_tipo)

    styled_proveedores = proveedores_display.style.format(
        {
            "proyectos": "{:,.0f}",
            "costo_promedio": "{:,.2f}",
            "costo_mediano": "{:,.2f}",
        }
    )

    st.dataframe(
        styled_proveedores,
        use_container_width=True,
        column_config={
            "proveedor": st.column_config.LinkColumn(
                "Proveedor",
                display_text=r"query=(.*?)&"
            )
        }
    )

    # =========================
    # TABLA ALCALDES
    # =========================
    st.subheader("Alcaldes con mayor costo promedio por unidad")

    alcaldes_stats = (
        df_cost.groupby("alcalde_ganador")
        .agg(
            proyectos=("snip", "nunique"),
            costo_promedio=("costo_por_unidad", "mean"),
            costo_mediano=("costo_por_unidad", "median")
        )
        .reset_index()
        .sort_values("costo_promedio", ascending=False)
    )

    # opcional: filtrar ruido (muy recomendado)
    #alcaldes_stats = alcaldes_stats[alcaldes_stats["proyectos"] >= 2]

    alcaldes_display = alcaldes_stats.copy()
    
    alcaldes_display["costo_promedio"] = pd.to_numeric(alcaldes_display["costo_promedio"], errors="coerce")
    alcaldes_display["costo_mediano"] = pd.to_numeric(alcaldes_display["costo_mediano"], errors="coerce")
    alcaldes_display["proyectos"] = pd.to_numeric(alcaldes_display["proyectos"], errors="coerce")
    alcaldes_display["alcalde_ganador"] = alcaldes_display["alcalde_ganador"].apply(
        lambda x: (
            f"https://firmaconcerteza.com/dashboard/search?query={quote(str(x))}"
            f"&type={'natural'}&page=1&pageSize=10"
            if pd.notna(x) else ""
        ))
    
    styled_alcaldes = alcaldes_display.style.format(
        {
            "proyectos": "{:,.0f}",
            "costo_promedio": "{:,.2f}",
            "costo_mediano": "{:,.2f}",
        }
    )

    st.dataframe(styled_alcaldes, use_container_width=True, 
                 column_config={
         "alcalde_ganador": st.column_config.LinkColumn(
            "Alcalde",
             display_text=r"query=(.*?)&"
        )})

 

        
