# v0.10.8 - Ajuste de despliegue Vercel

- Mantiene intacta la lógica OCR estructural de v0.10.7.
- Fija RapidOCR 3.10.0 para un entorno reproducible y compatible con OpenCV headless.
- Fija ONNX Runtime 1.20.1 en Python 3.12 (Vercel) y 1.23.2 en Python 3.13+.
- Cambia `uvicorn[standard]` por `uvicorn` base para no instalar uvloop/httptools innecesarios en Vercel.
- Instala `pytesseract` solo en Windows; Vercel no incluye el ejecutable Tesseract del sistema.
- Añade `uv.toml` con descarga/instalación serial para reducir el uso pico de `/tmp` durante la instalación de wheels grandes.
- Fija Python 3.12 para Vercel, reduciendo el wheel Linux de ONNX Runtime utilizado en producción.
- Añade `.vercelignore` para excluir pruebas y archivos locales del bundle.

## Variables Vercel recomendadas

Configurar en Project Settings > Environment Variables y volver a desplegar:

- `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`
- `UV_CONCURRENT_DOWNLOADS=1`
- `UV_CONCURRENT_INSTALLS=1`
- `UV_CONCURRENT_BUILDS=1`

Las tres variables `UV_*` duplican de forma intencional `uv.toml` como protección ante cambios en el directorio de trabajo del instalador de runtime de Vercel.
