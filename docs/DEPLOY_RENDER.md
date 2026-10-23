# Guía de despliegue de PRIVADOC en Render

> Esta guía es para evaluación universitaria. Al desplegar el backend en Render, los documentos dejan de procesarse exclusivamente en el equipo del usuario y pasan por el servidor de Render. Antes de una evaluación pública debe actualizarse el texto de privacidad de la interfaz para reflejarlo correctamente.

## 1. Preparar el repositorio

Subir el proyecto a un repositorio Git (GitHub es la opción más cómoda, aunque Render también admite otros proveedores/repositorios públicos o imágenes Docker).

No subir:
- `.venv/`;
- `__pycache__/`;
- documentos reales de participantes;
- datasets con información personal sin anonimizar.

## 2. Python

Se recomienda fijar Python 3.13 para mantener el entorno cercano al utilizado durante el desarrollo local.

Crear `.python-version` en la raíz, por ejemplo:

```text
3.13.5
```

## 3. Crear un Web Service

En Render:

1. **New → Web Service**.
2. Conectar el repositorio.
3. Runtime: **Python**.
4. Build command:

```bash
pip install -r requirements.txt
```

5. Start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

6. Health check: `/health` (Render lo consulta periódicamente; esas peticiones son normales y no analizan documentos).
7. Mantener una sola instancia/worker durante la evaluación porque el prototipo mantiene los análisis temporalmente en memoria. Una instancia procesa hasta dos documentos en paralelo por defecto; cada trabajo se ejecuta en un proceso aislado y el modelo OCR se carga dentro de ese proceso, que se termina al finalizar para devolver la memoria al sistema.

### Concurrencia y límites

Estos valores se pueden configurar como variables de entorno en Render:

- `PRIVADOC_MAX_CONCURRENT`: trabajos simultáneos (por defecto `2`). En una instancia con poca memoria se puede bajar a `1`.
- `PRIVADOC_QUEUE_TIMEOUT`: segundos que una solicitud espera un cupo (por defecto `5`). Si no hay cupo responde `429`, en lugar de quedar esperando indefinidamente.
- `PRIVADOC_JOB_TIMEOUT`: tiempo máximo de análisis o censura en segundos (por defecto `180`). Si un proceso OCR se atasca se termina y responde `504`.
- `PRIVADOC_CACHE_MAX`: análisis recientes que se conservan para la etapa de censura (por defecto `16`), evitando que los usuarios se expulsen entre sí al trabajar a la vez.
- `PRIVADOC_MAX_IMAGE_SIDE`: opcional; por defecto no reduce las imágenes. Solo debe configurarse si se desea un límite explícito.
- `PRIVADOC_PDF_RENDER_SCALE` y `PRIVADOC_MAX_PDF_PAGES`: opcionales; por defecto conservan la resolución 3x y el límite de 10 páginas.

No se debe aumentar `PRIVADOC_MAX_CONCURRENT` sin comprobar la memoria disponible: cada proceso puede cargar el modelo OCR y conservar temporalmente las páginas del documento.

## 4. Verificar

Abrir:

```text
https://TU-SERVICIO.onrender.com/api/health
```

Después probar la interfaz principal y los cuatro tipos de documento.

## 5. Consideraciones para la investigación

- Usar documentos de prueba o contar con consentimiento explícito para documentos reales.
- Informar a los participantes si el procesamiento se realiza en servidor.
- No afirmar “procesamiento local” cuando se use la versión desplegada en Render.
- Registrar tiempos de análisis con una metodología consistente.

## 6. Después de la evaluación

Para recuperar el objetivo de privacidad local en una versión distribuible, las rutas recomendadas son:

1. aplicación de escritorio con el motor Python/OCR local; o
2. extensión de Chrome + aplicación local usando Native Messaging.

Una PWA alojada en Render es instalable y cómoda, pero seguirá utilizando el servidor mientras el OCR permanezca en Python/FastAPI.
