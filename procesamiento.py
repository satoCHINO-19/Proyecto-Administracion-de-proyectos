import io
import json
import os
import tempfile
import time

import pypdf
from google import genai
from google.genai import errors as genai_errors
from pydantic import BaseModel

MODEL = "gemini-3.6-flash"


def _generar_con_reintento(client: genai.Client, **kwargs):
    """
    Reintenta la llamada a Gemini ante errores transitorios del servidor (503 "alta demanda"),
    con espera creciente entre intentos. Necesario sobre todo cuando el TDR se procesa por
    varios tramos: con más llamadas, la probabilidad de toparse con un 503 pasajero en alguna
    de ellas sube, y antes bastaba con que fallara una sola para perder todo el resultado.
    """
    intentos = 3
    for intento in range(intentos):
        try:
            return client.models.generate_content(**kwargs)
        except genai_errors.ServerError:
            if intento == intentos - 1:
                raise
            time.sleep(5 * (intento + 1))

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

   Además, indica en "alcance" si el requerimiento pertenece al "Producto" (el bien o servicio \
técnico que se entrega: equipos, funcionalidades, niveles de servicio) o al "Proyecto" (la gestión \
y los entregables para ejecutarlo: plan de trabajo, cronograma, informes, actas, capacitaciones, \
personal clave, plazos de ejecución). Todo requerimiento de tipo "Gestión de Proyecto" es de alcance \
"Proyecto".

3. Revisa si alguna consulta modifica o aclara el requerimiento. VINCULACIÓN CONSULTA–ÍTEM: una \
consulta se asocia a un ítem solo si la pregunta del postor hace referencia explícita a ese \
numeral/sección del TDR o trata exactamente sobre esa exigencia; no asocies consultas por mera \
similitud de tema. Si varias consultas afectan el mismo ítem, considera todas.
   - En "consulta_ref" pon la identificación de la(s) consulta(s) asociada(s) tal como figura en el \
documento (ej.: "Consulta 12" o "Consulta 12 y 15"), o cadena vacía si no hay ninguna.
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

5. Evalúa "cumple": CUMPLE, NO CUMPLE, CUMPLE PARCIALMENTE, o SIN_EVALUAR (estado "pendiente") solo \
si realmente no se puede concluir, citando en "justificacion" el texto concreto que sustenta tu \
evaluación. Si una propuesta existe y responde al requisito con evidencia clara, NO uses SIN_EVALUAR. \
Cuando uses SIN_EVALUAR, indica en "motivo_pendiente" exactamente uno de estos valores: \
"Falta propuesta" (no se recibió el documento de propuesta), "Sin evidencia en la propuesta" (hay \
propuesta pero no menciona el requisito; en ese caso lo correcto suele ser NO CUMPLE si el requisito \
es obligatorio, usa pendiente solo si es facultativo o ambiguo), "Requiere verificación externa" \
(depende de documentos no incluidos, ej. anexos, certificados o contratos) o "Requisito ambiguo" \
(el TDR o la consulta no permiten fijar el criterio). Si "cumple" no es SIN_EVALUAR, deja \
"motivo_pendiente" vacío.

6. Calcula "alerta" comparando cifras o plazos entre TDR, consulta y propuesta:
   - "MEJORA" si la propuesta ofrece una condición más favorable para la entidad que el mínimo \
exigido (ej.: TDR exige un plazo máximo de 30 días y la propuesta ofrece 14 días).
   - "DISCREPANCIA" si hay una variación entre documentos que podría ser desfavorable o requiere \
revisión manual (ej.: una garantía o plazo mínimo que la propuesta o el contrato reducen).
   - "" (cadena vacía) si no hay ninguna variación relevante que reportar.

Si falta alguno de los tres documentos, trabaja solo con lo que tengas (por ejemplo, si no hay \
propuesta, deja "propuesta_extracto" vacío y "cumple" como "SIN_EVALUAR").

El documento TDR que recibes puede ser el documento completo o solo un tramo de páginas de un \
documento más grande (una sección). En cualquier caso, extrae TODOS y cada uno de los \
requerimientos numerados que aparezcan en las páginas que te llegaron, sin resumir ni saltarte \
ninguno — aunque el tramo empiece o termine a mitad de una sección."""


class ChecklistItem(BaseModel):
    item: str
    tipo: str
    requisito_tdr: str
    alcance: str
    consulta_ref: str
    consulta: str
    se_acoge: str
    requisito_efectivo: str
    propuesta_extracto: str
    cumple: str
    motivo_pendiente: str
    alerta: str
    justificacion: str


class ChecklistResult(BaseModel):
    items: list[ChecklistItem]


PROMPT_RECONCILIAR = """Te doy una lista de requisitos de un checklist de cumplimiento de un \
proceso de contratación del Estado peruano (código + texto del requisito, en JSON). La lista se \
generó procesando el TDR por tramos de páginas separados, así que puede contener duplicados: el \
mismo requisito extraído dos veces, con código o redacción distintos, desde tramos distintos del \
documento (por ejemplo, un requisito de cantidad de líneas telefónicas puede aparecer con un \
código en una tabla resumen y con otro código en el detalle por gamas).

Tu tarea es identificar SOLO esos duplicados genuinos: agrupa los códigos que describen \
EXACTAMENTE el mismo dato o exigencia puntual del TDR.

MUY IMPORTANTE — NO agrupes códigos que traten temas distintos aunque se parezcan o estén en la \
misma sección: por ejemplo, una certificación ISO 37001 (antisoborno) y una certificación ISO \
9001 (calidad) NO son el mismo requisito aunque ambas sean "factor de evaluación facultativo" — \
cada certificación exige un documento distinto y debe evaluarse por separado. Ante la duda, NO \
agrupes: dos ítems que podrían ser el mismo requisito pero no estás seguro deben quedar SEPARADOS \
en la respuesta (simplemente no los incluyas en ningún grupo).

Devuelve únicamente los grupos de 2 o más códigos que sí son duplicados del mismo requisito. \
Cualquier código que no menciones en ningún grupo se conserva tal cual, sin cambios.

Lista de requisitos (JSON):
{items_json}"""


class GrupoDuplicados(BaseModel):
    codigos: list[str]


class ReconciliacionResult(BaseModel):
    grupos: list[GrupoDuplicados]


def _reconciliar_items(client: genai.Client, items: list[dict]) -> list[dict]:
    """
    Llamada final (sin subir archivos, solo texto) que identifica ítems duplicados por venir de
    tramos distintos del TDR trozado. A propósito NO le pedimos al modelo que reescriba/fusione
    el contenido: solo que agrupe los códigos que son el mismo requisito. La fusión real la hace
    esta función en Python, quedándose con el ítem más completo de cada grupo — así un
    agrupamiento erróneo del modelo nunca puede inventar contenido híbrido ni hacer desaparecer
    un ítem sin dejar rastro (todo código no agrupado se conserva tal cual). Solo hace falta
    cuando el TDR se procesó en más de un tramo.
    """
    por_codigo = {it["item"]: it for it in items if it.get("item")}
    resumen = [{"item": it["item"], "requisito_tdr": it.get("requisito_tdr", "")} for it in items]
    contents = [PROMPT_RECONCILIAR.format(items_json=json.dumps(resumen, ensure_ascii=False))]
    response = _generar_con_reintento(
        client,
        model=MODEL,
        contents=contents,
        config={
            "response_mime_type": "application/json",
            "response_schema": ReconciliacionResult,
        },
    )
    resultado: ReconciliacionResult = response.parsed

    codigos_agrupados: set[str] = set()
    items_finales: list[dict] = []
    for grupo in resultado.grupos:
        codigos_grupo = [c for c in grupo.codigos if c in por_codigo and c not in codigos_agrupados]
        if len(codigos_grupo) < 2:
            continue  # grupo inválido (código repetido, desconocido, o ya usado en otro grupo)
        mejor = max(codigos_grupo, key=lambda c: len(por_codigo[c].get("propuesta_extracto") or ""))
        items_finales.append(por_codigo[mejor])
        codigos_agrupados.update(codigos_grupo)

    for codigo, it in por_codigo.items():
        if codigo not in codigos_agrupados:
            items_finales.append(it)

    return items_finales


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
            # El nombre de archivo en disco es siempre "<rol>_<i>.pdf", sin importar tildes ni
            # sufijos descriptivos en `nombre` (como "(sección 2 de 4)") — así el cliente de
            # Gemini siempre puede detectar el mime type por la extensión. El nombre original,
            # con su sufijo si lo tiene, solo se usa como etiqueta de texto en el prompt.
            ruta = os.path.join(tmp_dir, f"{rol.lower()}_{i}.pdf")
            with open(ruta, "wb") as f:
                f.write(data)
            archivo_subido = client.files.upload(file=ruta)
            contents.append(f"--- Documento tipo {rol}: {nombre} ---")
            contents.append(archivo_subido)
    return contents


def _dividir_pdf_por_paginas(pdf_bytes: bytes, tamano_chunk: int = 30, solapamiento: int = 3) -> list[bytes]:
    """
    Divide un PDF en tramos de `tamano_chunk` páginas (con `solapamiento` páginas repetidas
    entre tramos consecutivos, para no cortar un requerimiento justo en el límite). Un TDR
    grande procesado de una sola vez tiende a que el modelo devuelva solo una muestra de los
    requerimientos en vez de todos; procesarlo por tramos más chicos fuerza una extracción
    más exhaustiva. Si el documento ya es chico, devuelve una sola parte (sin trocear).
    """
    lector = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    total_paginas = len(lector.pages)
    if total_paginas <= tamano_chunk:
        return [pdf_bytes]

    tramos: list[bytes] = []
    inicio = 0
    while inicio < total_paginas:
        fin = min(inicio + tamano_chunk, total_paginas)
        escritor = pypdf.PdfWriter()
        for num_pagina in range(inicio, fin):
            escritor.add_page(lector.pages[num_pagina])
        buffer = io.BytesIO()
        escritor.write(buffer)
        tramos.append(buffer.getvalue())
        if fin >= total_paginas:
            break
        inicio = fin - solapamiento
    return tramos


def _procesar_checklist_una_llamada(
    client: genai.Client,
    archivos_por_rol: dict[str, list[bytes]],
    nombres_por_rol: dict[str, list[str]],
) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        contents = _subir_documentos(client, archivos_por_rol, nombres_por_rol, tmp_dir)
        contents.append(PROMPT)

        response = _generar_con_reintento(
            client,
            model=MODEL,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": ChecklistResult,
            },
        )

    resultado: ChecklistResult = response.parsed
    return [item.model_dump() for item in resultado.items]


def procesar_checklist(
    archivos_por_rol: dict[str, list[bytes]],
    nombres_por_rol: dict[str, list[str]],
    on_progreso=None,
) -> list[dict]:
    """
    archivos_por_rol: {"TDR": [bytes, ...], "CONSULTAS": [...], "PROPUESTA": [...], "OTRO": [...]}
    nombres_por_rol: mismo shape pero con nombres de archivo originales, solo para el prompt.
    on_progreso: callback opcional on_progreso(tramo_actual, total_tramos) para mostrar avance
    en la interfaz cuando el TDR es grande y se procesa por secciones.

    El TDR se divide en tramos de páginas (ver _dividir_pdf_por_paginas) y cada tramo se procesa
    en una llamada separada junto con las CONSULTAS y la PROPUESTA completas, para que el modelo
    no se limite a devolver una muestra de los requerimientos de un documento largo. Los
    resultados de todos los tramos se combinan, descartando ítems duplicados por su código, y
    luego se reconcilian en una llamada final (ver _reconciliar_items) para fusionar ítems que
    describen el mismo requisito pero llegaron con códigos distintos desde tramos distintos.
    """
    client = _get_client()

    tdr_bytes_lista = archivos_por_rol.get("TDR") or []
    if not tdr_bytes_lista:
        return _procesar_checklist_una_llamada(client, archivos_por_rol, nombres_por_rol)

    tramos = _dividir_pdf_por_paginas(tdr_bytes_lista[0])
    if len(tramos) <= 1:
        return _procesar_checklist_una_llamada(client, archivos_por_rol, nombres_por_rol)

    tdr_nombre = nombres_por_rol["TDR"][0]
    resto_archivos = {rol: v for rol, v in archivos_por_rol.items() if rol != "TDR"}
    resto_nombres = {rol: v for rol, v in nombres_por_rol.items() if rol != "TDR"}

    items_combinados: list[dict] = []
    codigos_vistos: set[str] = set()
    for i, tramo_bytes in enumerate(tramos):
        if on_progreso:
            on_progreso(i + 1, len(tramos))
        archivos = {"TDR": [tramo_bytes], **resto_archivos}
        nombres = {"TDR": [f"{tdr_nombre} (sección {i + 1} de {len(tramos)})"], **resto_nombres}
        items_tramo = _procesar_checklist_una_llamada(client, archivos, nombres)
        for it in items_tramo:
            codigo = it.get("item", "")
            if codigo not in codigos_vistos:
                codigos_vistos.add(codigo)
                items_combinados.append(it)

    if on_progreso:
        on_progreso(-1, len(tramos))  # -1: señal de "reconciliando", ya no es un tramo más
    return _reconciliar_items(client, items_combinados)


# ---------------------------------------------------------------- EDT / WBS

BASE_CONOCIMIENTOS_PATH = os.path.join(os.path.dirname(__file__), "base_conocimientos.json")

PROMPT_EDT = """Eres un especialista en gestión de proyectos. Te doy el TDR (Términos de \
Referencia) de un proceso de contratación del Estado peruano y una base de conocimientos con \
costos y duraciones típicas de referencia (en JSON).

Tu tarea es construir la Estructura de Desglose del Trabajo (EDT/WBS) del proyecto:

1. Te doy abajo el ALCANCE DEL PRODUCTO y el ALCANCE DEL PROYECTO ya identificados en el TDR. La EDT debe cubrir TODOS los elementos de ambos, sin omitir ni duplicar ninguno. Estructura: los nodos de nivel 1 son los ENTREGABLES MAYORES del proyecto (por ejemplo "Gestión del Proyecto", "Implementación", "Operación y soporte", "Cierre"); agrupa bajo cada uno los entregables del PROYECTO que le corresponden (plan de trabajo, cronograma, informes, actas, capacitaciones) e INTEGRA dentro del mismo entregable mayor los entregables del PRODUCTO que le corresponden (por ejemplo, el servicio o los equipos bajo "Implementación" u "Operación"), como ramas hermanas en el mismo árbol. Prohibido crear fases separadas "solo de producto" y "solo de proyecto": cada entregable mayor debe mezclar lo que sea pertinente de ambos alcances. Jerarquía de máximo 3 niveles: entregable mayor (nivel 1), entregable (nivel 2) y actividad (nivel 3), con códigos "1", "1.1", "1.1.1". En "tipo" indica nivel y alcance: los nodos de nivel 1 son siempre "Fase · Proyecto"; los demás "Entregable · Producto", "Entregable · Proyecto", "Actividad · Producto" o "Actividad · Proyecto".

Alcances identificados (JSON):
{alcances}

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


PROMPT_ALCANCES = """Eres un analista de contrataciones del Estado peruano. Basándote únicamente en el TDR (Términos de Referencia) que te doy, identifica:

1. ALCANCE DEL PRODUCTO: el bien o servicio técnico que se debe entregar (equipos, funcionalidades, niveles de servicio, cantidades). Redacta un resumen breve y lista cada elemento como entregable del producto, con su descripción y la referencia (numeral del TDR).
2. ALCANCE DEL PROYECTO: la gestión del proyecto y los entregables requeridos para ejecutarlo (plan de trabajo, cronograma, informes, actas, capacitaciones, personal clave, garantías, liquidación y cierre). Redacta un resumen breve y lista cada entregable de gestión con su descripción y referencia.

No mezcles: un elemento va en el producto o en el proyecto, no en ambos. Sé exhaustivo con lo que el TDR exige explícitamente."""


class ElementoAlcance(BaseModel):
    nombre: str
    descripcion: str
    referencia_tdr: str


class AlcancesResult(BaseModel):
    resumen_producto: str
    entregables_producto: list[ElementoAlcance]
    resumen_proyecto: str
    entregables_proyecto: list[ElementoAlcance]


def generar_alcances(tdr_bytes: bytes, tdr_nombre: str) -> dict:
    """Identifica el alcance del producto y el alcance del proyecto a partir del TDR."""
    client = _get_client()
    with tempfile.TemporaryDirectory() as tmp_dir:
        contents = _subir_documentos(client, {"TDR": [tdr_bytes]}, {"TDR": [tdr_nombre]}, tmp_dir)
        contents.append(PROMPT_ALCANCES)
        response = _generar_con_reintento(
            client, model=MODEL, contents=contents,
            config={"response_mime_type": "application/json", "response_schema": AlcancesResult},
        )
    return response.parsed.model_dump()


def generar_edt(tdr_bytes: bytes, tdr_nombre: str, alcances: dict | None = None) -> list[dict]:
    """Genera un EDT/WBS con tiempos y costos a partir del TDR, cruzándolo con base_conocimientos.json."""
    if alcances is None:
        alcances = generar_alcances(tdr_bytes, tdr_nombre)
    client = _get_client()
    with open(BASE_CONOCIMIENTOS_PATH, "r", encoding="utf-8") as f:
        base_conocimientos = f.read()

    with tempfile.TemporaryDirectory() as tmp_dir:
        contents = _subir_documentos(client, {"TDR": [tdr_bytes]}, {"TDR": [tdr_nombre]}, tmp_dir)
        contents.append(PROMPT_EDT.format(
            base_conocimientos=base_conocimientos,
            alcances=json.dumps(alcances, ensure_ascii=False, indent=1),
        ))

        response = _generar_con_reintento(
            client,
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
