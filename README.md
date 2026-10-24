# PRIVADOC · v0.10.7

PRIVADOC es un prototipo universitario de protección local de datos sensibles en documentos peruanos y comprobantes digitales.

## Tipos soportados

- DNI azul peruano.
- DNI amarillo de menor de edad.
- DNI electrónico.
- Comprobantes bancarios digitales.

## Cambios de v0.10.0

- Se elimina **Historial local** del software.
- Se mantiene **Configuración de privacidad** para método de censura, selección automática de recomendaciones y último tipo de documento.
- Después de que el usuario descarga la copia protegida aparece una invitación discreta a responder la encuesta de evaluación del proyecto.
- La encuesta no se abre automáticamente y puede descartarse con **Ahora no**.
- El enlace definitivo de Google Forms se configura en una sola línea de `frontend/app.js`:

```js
const FEEDBACK_FORM_URL = 'GOOGLE_FORMS_URL_AQUI';
```

- Se reforzó la adaptación móvil del resultado final, configuración y tarjeta de encuesta.
- Los motores de DNI, comprobantes, OCR, normalización y censura se mantienen sin cambios funcionales respecto de la base recibida.

## Ejecutar

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Abre `http://127.0.0.1:8000`.


## Ejecución OCR y timeouts (v0.10.6)

Por defecto PRIVADOC reutiliza el motor OCR dentro del proceso web mediante un hilo de trabajo. Esto evita que Windows vuelva a cargar RapidOCR/ONNX desde cero para cada documento.

Solo si necesitas aislamiento duro por proceso, configura:

```text
PRIVADOC_ISOLATE_JOBS=1
PRIVADOC_JOB_TIMEOUT=900
```

Con el aislamiento desactivado (valor predeterminado), `PRIVADOC_JOB_TIMEOUT` no cancela el OCR en curso. El valor predeterminado de `PRIVADOC_MAX_CONCURRENT` es 1 para reutilizar el mismo motor OCR con seguridad; los análisis adicionales permanecen en cola.


## OCR estructural optimizado (v0.10.7)

Los tres DNI mantienen la normalización, las plantillas y los campos existentes, pero evitan repetir el detector OCR completo sobre cada ROI. Las regiones de una sola línea intentan primero el reconocedor de RapidOCR; el detector queda como rescate.

El OCR global de DNI está limitado a dos pasadas complementarias y el resto se resuelve por la estructura de cada documento. Los comprobantes conservan el modo resiliente general porque su maquetación es menos rígida.

Para el uso normal no necesitas activar ninguna variable adicional. Mantén `PRIVADOC_ISOLATE_JOBS=0` (valor predeterminado) en Windows para reutilizar el motor OCR.
