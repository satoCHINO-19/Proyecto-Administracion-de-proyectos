import os
import re
import tempfile
import unicodedata

from google import genai
from pydantic import BaseModel


def _nombre_seguro(nombre: str) -> str:
    """Quita tildes/ñ y cualquier caracter no-ASCII para evitar errores al subir el archivo."""
    sin_tildes = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9._-]", "_", sin_tildes) or "archivo.pdf"


MODEL = "gemini-3.6-flash"

TIPOS_VALIDOS = ["Técnico", "Administrativo", "Económico", "Plazo", "Perfil profesional", "Gestión de Proyecto"]

PROMPT = """Eres un analista de contrataciones del Estado peruano. Te doy hasta tres tipos de \
documentos de un mismo proceso de contratación:

- TDR (Términos de Referencia): contiene los requerimientos numerados (1.1, 1.2, 2.1, etc.)
- CONSULTAS: preguntas de los postores sobre el TDR y su absolución (pueden modificar, aclarar \
o mantener un requerimiento del TDR). Cada consulta tiene un campo "Estado" con el valor literal \
"Se acoge", "No se acoge" o "Se acoge parcialmente".
- PROPUESTA: la propuesta técnica de un postor, que debe responder a los requerimientos del TDR \
ya modificados por las consultas.

Tu tarea, por cada requerimiento numerado del TDR:

1. Extrae el requerimiento y clasifícalo en "tipo": Técnico, Administrativo, Económico, Plazo, \
"Perfil profesional" o "Gestión de Proyecto". IMPORTANTE: no te limites a requisitos de producto \
o técnicos — incluye también, si el TDR los exige, entregables de gestión del proyecto como plan \
de proyecto, cronograma, actas de reunión o capacitaciones, clasificándolos como "Gestión de Proyecto".

2. Si el requerimiento menciona un plazo en días, indica SIEMPRE de forma explícita si son \
"días calendario" o "días hábiles" tal como lo dice el documento fuente — nunca dejes "días" \
ambiguo sin precisar el tipo.

3. Revisa si alguna consulta modifica o aclara el requerimiento:
   - Copia en "se_acoge" el valor literal del campo Estado de esa consulta ("Se acoge", \
"No se acoge", "Se acoge parcialmente"). Si no hay consulta asociada, usa "N/A".
   - En "consulta", si el estado es "No se acoge", redacta explícitamente qué condición del TDR \
se mantiene vigente sin cambios (por ejemplo: "No se acoge: se mantiene el plazo original de 30 \
días calendario establecido en el TDR, sin modificaciones"), nunca dejes solo la etiqueta "no se \
acoge" sin explicar el efecto concreto.
   - Si la consulta o su absolución aclara que un cambio aplica solo a una etapa (por ejemplo, a \
la "prestación del servicio" y no a la "integración" o instalación), dilo explícitamente en \
"consulta", indicando qué elementos se mantienen sin cambio.
   - Con esto, redacta el "requisito_efectivo" (TDR + cambio de la consulta si la hubo).

4. Busca en la propuesta el fragmento que responde a ese requisito efectivo y cópialo en \
"propuesta_extracto".

5. Evalúa "cumple": CUMPLE, NO CUMPLE, CUMPLE PARCIALMENTE, o SIN_EVALUAR si falta información, \
citando en "justificacion" el texto concreto que sustenta tu evaluación.

6. Calcula "alerta" comparando cifras o plazos entre TDR, consulta y propuesta:
   - "MEJORA" si la propuesta ofrece una condición más favorable para la entidad que el mínimo \
exigido (ej.: TDR exige un plazo máximo de 30 días y la propuesta ofrece 14 días).
   - "DISCREPANCIA" si hay una variación entre documentos que podría ser desfavorable o requiere \
revisión manual (ej.: una garantía o plazo mínimo que la propuesta o el contrato reducen).
   - "" (cadena vacía) si no hay ninguna variación relevante que reportar.

Si falta alguno de los tres documentos, trabaja solo con lo que tengas (por ejemplo, si no hay \
propuesta, deja "propuesta_extracto" vacío y "cumple" como "SIN_EVALUAR").

Devuelve TODOS los requerimientos encontrados en el TDR, en orden."""


class ChecklistItem(BaseModel):
    item: str
    tipo: str
    requisito_tdr: str
    consulta: str
    se_acoge: str
    requisito_efectivo: str
    propuesta_extracto: str
    cumple: str
    alerta: str
    justificacion: str


class ChecklistResult(BaseModel):
    items: list[ChecklistItem]


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("No se encontró GEMINI_API_KEY en el entorno (.env)")
    return genai.Client(api_key=api_key)


def _subir_documentos(
    client: genai.Client,
    archivos_por_rol: dict[str, list[bytes]],
    nombres_por_rol: dict[str, list[str]],
    tmp_dir: str,
) -> list:
    contents: list = []
    for rol, lista_bytes in archivos_por_rol.items():
        nombres = nombres_por_rol.get(rol, [])
        for i, (data, nombre) in enumerate(zip(lista_bytes, nombres)):
            ruta = os.path.join(tmp_dir, f"{rol.lower()}_{i}_{_nombre_seguro(nombre)}")
            with open(ruta, "wb") as f:
                f.write(data)
            archivo_subido = client.files.upload(file=ruta)
            contents.append(f"--- Documento tipo {rol}: {nombre} ---")
            contents.append(archivo_subido)
    return contents


def procesar_checklist(archivos_por_rol: dict[str, list[bytes]], nombres_por_rol: dict[str, list[str]]) -> list[dict]:
    """
    archivos_por_rol: {"TDR": [bytes, ...], "CONSULTAS": [...], "PROPUESTA": [...], "OTRO": [...]}
    nombres_por_rol: mismo shape pero con nombres de archivo originales, solo para el prompt.
    """
    client = _get_client()

    with tempfile.TemporaryDirectory() as tmp_dir:
        contents = _subir_documentos(client, archivos_por_rol, nombres_por_rol, tmp_dir)
        contents.append(PROMPT)

        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": ChecklistResult,
            },
        )

    resultado: ChecklistResult = response.parsed
    return [item.model_dump() for item in resultado.items]


# ---------------------------------------------------------------- EDT / WBS

BASE_CONOCIMIENTOS_PATH = os.path.join(os.path.dirname(__file__), "base_conocimientos.json")

PROMPT_EDT = """Eres un especialista en gestión de proyectos. Te doy el TDR (Términos de \
Referencia) de un proceso de contratación del Estado peruano y una base de conocimientos con \
costos y duraciones típicas de referencia (en JSON).

Tu tarea es construir la Estructura de Desglose del Trabajo (EDT/WBS) del proyecto:

1. Descompón el alcance del TDR en una jerarquía de máximo 3 niveles: Fase (nivel 1), \
Entregable (nivel 2) y Actividad (nivel 3). Usa códigos tipo "1", "1.1", "1.1.1".
2. Para cada nodo HOJA (el que no tiene hijos propios, normalmente una Actividad), estima \
"duracion_dias" (días calendario) y "costo_soles":
   - Si el nodo se parece a un entregable de la base de conocimientos, usa ese valor de \
referencia (ajustándolo si el alcance del TDR es claramente mayor o menor) y cita en \
"fuente_estimacion" el nombre exacto del entregable de referencia que usaste.
   - Si no hay nada parecido en la base de conocimientos, estima tú mismo un valor razonable \
y escribe en "fuente_estimacion" "Estimado por IA (sin referencia en base de conocimientos)".
3. Para cada nodo que SÍ tiene hijos (Fase o Entregable), deja "duracion_dias" y "costo_soles" \
en 0 y "fuente_estimacion" vacía — se calculan automáticamente sumando a sus hijos. En su lugar, \
indica en "ejecucion_hijos" si sus hijos directos se ejecutan "secuencial" (uno después de que \
termina el anterior) o "paralelo" (al mismo tiempo, por ejemplo actividades de soporte, reportes \
mensuales y prestación continua del servicio que ocurren simultáneamente durante toda la vigencia \
del contrato). En los nodos hoja deja "ejecucion_hijos" como cadena vacía.
4. Dejar "depende_de" como cadena vacía — la dependencia entre nodos se calcula \
automáticamente después, a partir del orden de los códigos.

Base de conocimientos (JSON de referencia):
{base_conocimientos}

Devuelve la jerarquía completa cubriendo todo el alcance del TDR, no solo un resumen."""


class EDTItem(BaseModel):
    codigo: str
    nombre: str
    tipo: str
    duracion_dias: float
    costo_soles: float
    fuente_estimacion: str
    depende_de: str
    ejecucion_hijos: str


class EDTResult(BaseModel):
    items: list[EDTItem]


def _clave_codigo(codigo: str) -> tuple[int, ...]:
    return tuple(int(parte) for parte in codigo.split("."))


def _calcular_dependencias(items: list[dict]) -> None:
    """
    Sobrescribe depende_de de forma determinística a partir de la jerarquía de códigos,
    en vez de confiar en que el modelo infiera la relación correcta (podía apuntar a un
    código de otra rama del árbol). Regla: cada nodo depende del hermano anterior bajo el
    mismo padre; el primer hijo de un padre depende del propio padre.
    """
    ultimo_hijo_de: dict[str, str] = {}
    for it in items:
        codigo = it["codigo"]
        padre = codigo.rsplit(".", 1)[0] if "." in codigo else ""
        it["depende_de"] = ultimo_hijo_de.get(padre, padre)
        ultimo_hijo_de[padre] = codigo


def _calcular_rollup(items: list[dict]) -> None:
    """
    Recalcula duracion_dias y costo_soles de los nodos con hijos (Fase/Entregable) a partir de
    sus hijos directos, en vez de confiar en que el modelo los sume bien. El costo siempre se
    suma (es aditivo sin importar el paralelismo); la duración se suma si "ejecucion_hijos" es
    "secuencial" y se toma el máximo si es "paralelo" — sin esto, una fase con entregables que
    corren al mismo tiempo (ej. soporte + reportes mensuales + servicio continuo durante los
    mismos 24 meses) sumaría sus duraciones como si fueran secuenciales y triplicaría el total.
    """
    por_codigo = {it["codigo"]: it for it in items}
    hijos_de: dict[str, list[str]] = {}
    for it in items:
        codigo = it["codigo"]
        if "." in codigo:
            padre = codigo.rsplit(".", 1)[0]
            hijos_de.setdefault(padre, []).append(codigo)

    # de más profundo a menos profundo, para que un padre ya tenga a sus hijos recalculados
    for codigo in sorted(por_codigo, key=_clave_codigo, reverse=True):
        hijos = hijos_de.get(codigo)
        if not hijos:
            continue  # nodo hoja: se conserva la estimación del modelo
        nodo = por_codigo[codigo]
        duraciones = [por_codigo[h]["duracion_dias"] for h in hijos]
        nodo["costo_soles"] = sum(por_codigo[h]["costo_soles"] for h in hijos)
        nodo["duracion_dias"] = max(duraciones) if nodo.get("ejecucion_hijos") == "paralelo" else sum(duraciones)
        nodo["fuente_estimacion"] = f"Rollup de hijos ({nodo.get('ejecucion_hijos') or 'secuencial'})"


def generar_edt(tdr_bytes: bytes, tdr_nombre: str) -> list[dict]:
    """Genera un EDT/WBS con tiempos y costos a partir del TDR, cruzándolo con base_conocimientos.json."""
    client = _get_client()
    with open(BASE_CONOCIMIENTOS_PATH, "r", encoding="utf-8") as f:
        base_conocimientos = f.read()

    with tempfile.TemporaryDirectory() as tmp_dir:
        contents = _subir_documentos(client, {"TDR": [tdr_bytes]}, {"TDR": [tdr_nombre]}, tmp_dir)
        contents.append(PROMPT_EDT.format(base_conocimientos=base_conocimientos))

        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": EDTResult,
            },
        )

    resultado: EDTResult = response.parsed
    items = [item.model_dump() for item in resultado.items]
    items.sort(key=lambda it: _clave_codigo(it["codigo"]))
    _calcular_rollup(items)
    _calcular_dependencias(items)
    return items
