import io

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from procesamiento import procesar_checklist

load_dotenv()

st.set_page_config(page_title="Checklist TDR / Consultas / Propuesta", layout="wide")
st.title("Checklist de cumplimiento — TDR / Consultas / Propuesta")
st.caption(
    "Sube los PDFs del proceso de contratación, indica qué es cada uno y genera "
    "el checklist de cumplimiento automáticamente con Gemini."
)

ROLES = ["TDR", "CONSULTAS", "PROPUESTA", "OTRO"]

archivos = st.file_uploader(
    "Sube tus PDFs (puedes subir varios a la vez)",
    type=["pdf"],
    accept_multiple_files=True,
)

roles_asignados: dict[str, str] = {}
if archivos:
    st.subheader("Asigna el tipo de cada archivo")
    for archivo in archivos:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(archivo.name)
        with col2:
            roles_asignados[archivo.name] = st.selectbox(
                f"Tipo para {archivo.name}",
                ROLES,
                key=f"rol_{archivo.name}",
                label_visibility="collapsed",
            )

CUMPLE_COLOR = {
    "CUMPLE": "background-color: #d4edda",
    "NO CUMPLE": "background-color: #f8d7da",
    "CUMPLE PARCIALMENTE": "background-color: #fff3cd",
    "SIN_EVALUAR": "background-color: #e2e3e5",
}


def colorear(val: str) -> str:
    return CUMPLE_COLOR.get(str(val).upper(), "")


if archivos and st.button("Procesar", type="primary"):
    archivos_por_rol: dict[str, list[bytes]] = {rol: [] for rol in ROLES}
    nombres_por_rol: dict[str, list[str]] = {rol: [] for rol in ROLES}

    for archivo in archivos:
        rol = roles_asignados[archivo.name]
        archivos_por_rol[rol].append(archivo.getvalue())
        nombres_por_rol[rol].append(archivo.name)

    with st.spinner("Subiendo archivos y generando checklist con Gemini... puede tardar un poco con PDFs grandes"):
        try:
            items = procesar_checklist(archivos_por_rol, nombres_por_rol)
        except Exception as e:
            st.error(f"Ocurrió un error procesando los documentos: {e}")
            st.stop()

    if not items:
        st.warning("Gemini no devolvió ningún ítem. Revisa que el TDR tenga requerimientos numerados.")
        st.stop()

    df = pd.DataFrame(items)
    columnas_orden = [
        "item",
        "requisito_tdr",
        "consulta",
        "requisito_efectivo",
        "propuesta_extracto",
        "cumple",
        "justificacion",
    ]
    df = df[[c for c in columnas_orden if c in df.columns]]

    st.subheader("Checklist de cumplimiento")
    if "cumple" in df.columns:
        styler = df.style
        aplicar_color = getattr(styler, "map", None) or styler.applymap
        tabla = aplicar_color(colorear, subset=["cumple"])
    else:
        tabla = df
    st.dataframe(tabla, use_container_width=True, height=600)

    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, sheet_name="Checklist")
    st.download_button(
        "Descargar checklist en Excel",
        data=buffer.getvalue(),
        file_name="checklist_tdr_consultas_propuesta.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
