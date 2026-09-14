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


def procesar_checklist(archivos_por_rol: dict[str, list[bytes]], nombres_por_rol: dict[str, list[str]]) -> list[dict]:
    """
    archivos_por_rol: {"TDR": [bytes, ...], "CONSULTAS": [...], "PROPUESTA": [...], "OTRO": [...]}
    nombres_por_rol: mismo shape pero con nombres de archivo originales, solo para el prompt.
    """
    client = _get_client()

    contents: list = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for rol, lista_bytes in archivos_por_rol.items():
            nombres = nombres_por_rol.get(rol, [])
            for i, (data, nombre) in enumerate(zip(lista_bytes, nombres)):
                ruta = os.path.join(tmp_dir, f"{rol.lower()}_{i}_{_nombre_seguro(nombre)}")
                with open(ruta, "wb") as f:
                    f.write(data)
                archivo_subido = client.files.upload(file=ruta)
                contents.append(f"--- Documento tipo {rol}: {nombre} ---")
                contents.append(archivo_subido)

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
