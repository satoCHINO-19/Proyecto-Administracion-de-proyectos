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

PROMPT = """Eres un analista de contrataciones del Estado peruano. Te doy hasta tres tipos de \
documentos de un mismo proceso de contratación:

- TDR (Términos de Referencia): contiene los requerimientos numerados (1.1, 1.2, 2.1, etc.)
- CONSULTAS: preguntas de los postores sobre el TDR y su absolución (pueden modificar, aclarar \
o mantener un requerimiento del TDR).
- PROPUESTA: la propuesta técnica de un postor, que debe responder a los requerimientos del TDR \
ya modificados por las consultas.

Tu tarea:
1. Extrae del TDR cada requerimiento numerado.
2. Para cada requerimiento, revisa si alguna consulta lo modifica o aclara. Si es así, redacta \
el "requisito efectivo" (TDR + cambio de la consulta). Si no hay consulta asociada, el requisito \
efectivo es igual al del TDR.
3. Busca en la propuesta el fragmento que responde a ese requisito efectivo.
4. Evalúa si la propuesta CUMPLE, NO CUMPLE o CUMPLE PARCIALMENTE el requisito efectivo, citando \
el texto concreto de la propuesta que sustenta tu evaluación.

Si falta alguno de los tres documentos, trabaja solo con lo que tengas (por ejemplo, si no hay \
propuesta, deja "propuesta_extracto" vacío y "cumple" como "SIN_EVALUAR").

Devuelve TODOS los requerimientos encontrados en el TDR, en orden."""


class ChecklistItem(BaseModel):
    item: str
    requisito_tdr: str
    consulta: str
    requisito_efectivo: str
    propuesta_extracto: str
    cumple: str
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
