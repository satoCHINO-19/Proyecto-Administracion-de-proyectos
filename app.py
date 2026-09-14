import html
import io
import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from procesamiento import generar_edt, procesar_checklist

load_dotenv()

st.set_page_config(page_title="Checklist TDR / Consultas / Propuesta", layout="wide")

# ---------------------------------------------------------------- estilos
BG = "#0b0f1a"
CARD_BG = "#141a2b"
CARD_BORDER = "#232b40"
TEXT = "#e7ecf5"
MUTED = "#8b93a8"

GREEN, GREEN_BG = "#22c55e", "rgba(34,197,94,0.14)"
AMBER, AMBER_BG = "#f5a623", "rgba(245,166,35,0.14)"
RED, RED_BG = "#ef4444", "rgba(239,68,68,0.14)"
GRAY, GRAY_BG = "#9aa4b8", "rgba(154,164,184,0.14)"

TIPO_COLOR = {
    "Técnico": ("#4f9dde", "rgba(79,157,222,0.14)"),
    "Administrativo": ("#38bdf8", "rgba(56,189,248,0.14)"),
    "Económico": ("#34d399", "rgba(52,211,153,0.14)"),
    "Plazo": ("#fb923c", "rgba(251,146,60,0.14)"),
    "Perfil profesional": ("#ec4899", "rgba(236,72,153,0.14)"),
    "Gestión de Proyecto": ("#a78bfa", "rgba(167,139,250,0.14)"),
}

CUMPLE_COLOR = {
    "CUMPLE": (GREEN, GREEN_BG),
    "NO CUMPLE": (RED, RED_BG),
    "CUMPLE PARCIALMENTE": (AMBER, AMBER_BG),
    "SIN_EVALUAR": (GRAY, GRAY_BG),
}

# Nota: por pedido explícito, el color va invertido respecto a lo intuitivo:
# "No se acoge" (se mantiene el TDR original, sin riesgo) se marca en VERDE;
# "Se acoge" (se modificó la base, hay que revisar el cambio) se marca en ROJO.
SE_ACOGE_COLOR = {
    "No se acoge": (GREEN, GREEN_BG),
    "Se acoge": (RED, RED_BG),
    "Se acoge parcialmente": (AMBER, AMBER_BG),
    "N/A": (GRAY, GRAY_BG),
}

ALERTA_STYLE = {
    "MEJORA": (GREEN, GREEN_BG, "Mejora sobre el mínimo exigido"),
    "DISCREPANCIA": (RED, RED_BG, "Variación entre documentos — revisar"),
}


def render_html(raw: str) -> None:
    # Streamlit's markdown treats indented lines as a code block; a blank line
    # (e.g. from an empty f-string placeholder) makes the next indented lines
    # get swallowed as literal text instead of rendered HTML. Stripping each
    # line's leading whitespace avoids that.
    st.markdown("\n".join(line.strip() for line in raw.strip().splitlines()), unsafe_allow_html=True)


st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    .badge {{
        display: inline-block; padding: 3px 10px; border-radius: 100px;
        font-size: 12px; font-weight: 600; white-space: nowrap;
    }}
    .card {{
        background: {CARD_BG}; border: 1px solid {CARD_BORDER}; border-radius: 12px;
        padding: 16px 18px; margin-bottom: 10px;
    }}
    .stat-num {{ font-size: 30px; font-weight: 700; line-height: 1; }}
    .stat-label {{ color: {MUTED}; font-size: 12px; letter-spacing: 0.05em; text-transform: uppercase; margin-top: 4px; }}
    .row-item {{ color: {MUTED}; font-family: monospace; font-size: 12px; }}
    .row-alert-dot {{ color: {RED}; font-size: 14px; margin-right: 4px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Checklist de cumplimiento — TDR / Consultas / Propuesta")
titulo_caso = st.text_input("Nombre del caso (opcional, para el encabezado)", value="")
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

    st.session_state["items"] = items
    if archivos_por_rol["TDR"]:
        st.session_state["tdr_bytes"] = archivos_por_rol["TDR"][0]
        st.session_state["tdr_nombre"] = nombres_por_rol["TDR"][0]

items = st.session_state.get("items")

if items:
    df = pd.DataFrame(items)
    for col in ["tipo", "se_acoge", "alerta"]:
        if col not in df.columns:
            df[col] = ""

    # ------------------------------------------------------------ encabezado + stats
    if titulo_caso:
        st.markdown(f"### {html.escape(titulo_caso)}")

    total = len(df)
    n_cumple = int((df["cumple"] == "CUMPLE").sum())
    n_parcial = int((df["cumple"] == "CUMPLE PARCIALMENTE").sum())
    n_no_cumple = int((df["cumple"] == "NO CUMPLE").sum())
    n_no_evid = int((df["cumple"] == "SIN_EVALUAR").sum())

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (c1, n_cumple, "Cumple", GREEN),
        (c2, n_parcial, "Parcial", AMBER),
        (c3, n_no_cumple, "No cumple", RED),
        (c4, n_no_evid, "No evidenciado", GRAY),
    ]
    for col, num, label, color in stats:
        col.markdown(
            f'<div class="card"><div class="stat-num" style="color:{color}">{num}</div>'
            f'<div class="stat-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    pct = int(round(100 * n_cumple / total)) if total else 0
    st.progress(pct / 100, text=f"{n_cumple} / {total} requisitos cumplidos ({pct}%)")

    # ------------------------------------------------------------ filtros
    fc1, fc2 = st.columns([3, 2])
    with fc1:
        busqueda = st.text_input("Buscar requisito o ID...", value="", label_visibility="collapsed", placeholder="Buscar requisito o ID...")
    with fc2:
        tipos_presentes = sorted(t for t in df["tipo"].unique() if t)
        tipos_sel = st.multiselect("Tipo", tipos_presentes, default=tipos_presentes, label_visibility="collapsed", placeholder="Filtrar por tipo")

    estados_presentes = ["CUMPLE", "CUMPLE PARCIALMENTE", "NO CUMPLE", "SIN_EVALUAR"]
    estados_sel = st.pills(
        "Estado",
        [e for e in estados_presentes if e in df["cumple"].unique()],
        selection_mode="multi",
        default=[e for e in estados_presentes if e in df["cumple"].unique()],
        label_visibility="collapsed",
    )

    df_f = df[df["tipo"].isin(tipos_sel) & df["cumple"].isin(estados_sel)]
    if busqueda:
        mask = df_f["item"].str.contains(busqueda, case=False, na=False) | df_f["requisito_tdr"].str.contains(busqueda, case=False, na=False)
        df_f = df_f[mask]

    st.caption(f"{len(df_f)} / {total} requisitos")

    # ------------------------------------------------------------ tabla de tarjetas
    for _, row in df_f.iterrows():
        cumple_c, cumple_bg = CUMPLE_COLOR.get(row["cumple"], (GRAY, GRAY_BG))
        tipo_c, tipo_bg = TIPO_COLOR.get(row["tipo"], (GRAY, GRAY_BG))
        acoge_c, acoge_bg = SE_ACOGE_COLOR.get(row["se_acoge"], (GRAY, GRAY_BG))

        alerta = row.get("alerta") or ""
        alert_dot = '<span class="row-alert-dot">●</span>' if alerta else ""

        m = re.search(r"Consulta\s+\d+(\s*y\s*\d+)?", str(row.get("consulta", "")))
        consulta_tag = ""
        if row["se_acoge"] != "N/A":
            etiqueta = m.group(0) if m else "Consulta"
            consulta_tag = f'<span class="badge" style="color:{AMBER};background:{AMBER_BG};margin-right:6px;">{html.escape(etiqueta)}</span>'

        alerta_html = ""
        if alerta in ALERTA_STYLE:
            a_c, a_bg, a_txt = ALERTA_STYLE[alerta]
            alerta_html = (
                f'<div style="margin-top:8px;padding:8px 12px;border-radius:8px;'
                f'background:{a_bg};color:{a_c};font-size:12.5px;font-weight:600;">'
                f"{'✓' if alerta == 'MEJORA' else '⚠'} {html.escape(a_txt)}</div>"
            )

        render_html(
            f"""
            <div class="card">
              <div style="display:flex; justify-content:space-between; gap:12px;">
                <div style="flex:1;">
                  <div>{alert_dot}<span class="row-item">{html.escape(str(row['item']))}</span></div>
                  <div style="font-weight:600; margin:4px 0 8px 0;">{html.escape(str(row['requisito_tdr']))}</div>
                  {consulta_tag}
                  <span class="badge" style="color:{tipo_c};background:{tipo_bg};">{html.escape(str(row['tipo']) or 'Sin tipo')}</span>
                  <span class="badge" style="color:{acoge_c};background:{acoge_bg};">{html.escape(str(row['se_acoge']))}</span>
                </div>
                <div style="text-align:right; min-width:150px;">
                  <span class="badge" style="color:{cumple_c};background:{cumple_bg};">{html.escape(str(row['cumple']))}</span>
                </div>
              </div>
              {alerta_html}
            </div>
            """
        )
        with st.expander("Ver detalle (requisito efectivo, propuesta y justificación)"):
            st.markdown(f"**Requisito efectivo:** {html.escape(str(row['requisito_efectivo']))}")
            st.markdown(f"**Consulta:** {html.escape(str(row['consulta'])) or '—'}")
            st.markdown(f"**Extracto de la propuesta:** {html.escape(str(row['propuesta_extracto'])) or '—'}")
            st.markdown(f"**Justificación:** {html.escape(str(row['justificacion']))}")

    # ------------------------------------------------------------ hallazgos críticos
    hallazgos = df_f[(df_f["cumple"] == "NO CUMPLE") | (df_f["alerta"] == "DISCREPANCIA")]
    if not hallazgos.empty:
        st.markdown("### Hallazgos críticos")
        for _, row in hallazgos.iterrows():
            st.markdown(
                f'<div style="border:1px solid {RED};background:{RED_BG};border-radius:8px;'
                f'padding:10px 14px;margin-bottom:6px;">'
                f'<span class="row-item" style="color:{RED};">{html.escape(str(row["item"]))}</span> '
                f'{html.escape(str(row["justificacion"]))}</div>',
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------ descarga
    buffer = io.BytesIO()
    columnas_orden = [
        "item", "tipo", "requisito_tdr", "consulta", "se_acoge",
        "requisito_efectivo", "propuesta_extracto", "cumple", "alerta", "justificacion",
    ]
    df_export = df[[c for c in columnas_orden if c in df.columns]]
    df_export.to_excel(buffer, index=False, sheet_name="Checklist")
    st.download_button(
        "Descargar checklist en Excel",
        data=buffer.getvalue(),
        file_name="checklist_tdr_consultas_propuesta.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ------------------------------------------------------------ EDT / WBS
    st.markdown("---")
    st.subheader("EDT / WBS del proyecto")
    st.caption(
        "Genera la Estructura de Desglose del Trabajo a partir del TDR, con tiempos y costos "
        "estimados cruzando la base de conocimientos de referencia (base_conocimientos.json)."
    )

    if st.session_state.get("tdr_bytes") and st.button("Generar EDT/WBS"):
        with st.spinner("Generando EDT/WBS con Gemini..."):
            try:
                st.session_state["edt_items"] = generar_edt(
                    st.session_state["tdr_bytes"], st.session_state["tdr_nombre"]
                )
            except Exception as e:
                st.error(f"Ocurrió un error generando el EDT: {e}")

    edt_items = st.session_state.get("edt_items")
    if edt_items:
        df_edt = pd.DataFrame(edt_items)
        fases = df_edt[~df_edt["codigo"].str.contains(r"\.")]
        total_dias = fases["duracion_dias"].sum()
        total_costo = fases["costo_soles"].sum()

        e1, e2 = st.columns(2)
        e1.markdown(
            f'<div class="card"><div class="stat-num" style="color:{TEXT}">{total_dias:.0f} días</div>'
            f'<div class="stat-label">Duración total estimada</div></div>',
            unsafe_allow_html=True,
        )
        e2.markdown(
            f'<div class="card"><div class="stat-num" style="color:{TEXT}">S/ {total_costo:,.0f}</div>'
            f'<div class="stat-label">Costo total estimado</div></div>',
            unsafe_allow_html=True,
        )

        # ya viene ordenado numéricamente por jerarquía desde generar_edt(); un sort_values("codigo")
        # aquí ordenaría como texto ("1.10" antes de "1.2") y rompería la jerarquía.
        for _, row in df_edt.iterrows():
            nivel = str(row["codigo"]).count(".")
            indent = nivel * 28
            render_html(
                f"""
                <div class="card" style="margin-left:{indent}px;">
                  <div style="display:flex; justify-content:space-between; gap:12px;">
                    <div>
                      <span class="row-item">{html.escape(str(row['codigo']))}</span>
                      <strong>{html.escape(str(row['nombre']))}</strong>
                      <span class="badge" style="color:{MUTED};background:{GRAY_BG};margin-left:6px;">{html.escape(str(row['tipo']))}</span>
                    </div>
                    <div style="white-space:nowrap; color:{TEXT};">
                      {row['duracion_dias']:.0f} días · S/ {row['costo_soles']:,.0f}
                    </div>
                  </div>
                  <div style="color:{MUTED}; font-size:11.5px; margin-top:4px;">
                    Fuente: {html.escape(str(row['fuente_estimacion']))}
                  </div>
                </div>
                """
            )

        buffer_edt = io.BytesIO()
        df_edt.to_excel(buffer_edt, index=False, sheet_name="EDT_WBS")
        st.download_button(
            "Descargar EDT/WBS en Excel",
            data=buffer_edt.getvalue(),
            file_name="edt_wbs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
