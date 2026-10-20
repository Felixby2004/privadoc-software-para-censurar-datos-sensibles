# PRIVADOC · v0.10.1

PRIVADOC es un prototipo universitario de protección local de datos sensibles en documentos peruanos y comprobantes digitales.

## Tipos soportados

- DNI azul peruano.
- DNI amarillo de menor de edad.
- DNI electrónico.
- Comprobantes bancarios digitales.

## Cambios de v0.10.1

- OCR estructural reforzado por ROI esperada para DNI azul, amarillo y electrónico.
- La lectura global se conserva y sólo se rescanean campos débiles dentro de la estructura ya calibrada.
- Invitación final a encuesta más visible y adaptada a evaluación 1–10.

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

## Documentación de evaluación y despliegue

- `docs/EVALUACION_USUARIOS.md`: preguntas sugeridas 1–10 y separación entre métricas técnicas y aceptación.
- `docs/DEPLOY_RENDER.md`: guía de despliegue para evaluación universitaria y advertencias de privacidad.
