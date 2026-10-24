# v0.10.7 - OCR estructural rápido

- Los DNI usan como máximo dos pasadas OCR globales: original y ampliada con CLAHE; los rescates posteriores se realizan sobre ROIs estructurales.
- RapidOCR omite el clasificador de orientación en documentos ya normalizados.
- Los campos de una sola línea usan reconocimiento directo (`use_det=False`) antes de ejecutar el detector de texto.
- Se corrigió un bug del OCR multiescala que podía disparar una pasada adaptativa adicional aun cuando la ampliación ya había recuperado texto suficiente.
- Se reutiliza el OCR global para MRZ, cabecera, fechas y anclas antes de lanzar OCR local.
- DNI amarillo: se acotaron los rescates de cabecera, fechas y dígito verificador a una sola pasada de detector después del reconocimiento directo.
- DNI electrónico: se eliminó el OCR duplicado de las fechas y se prueban normal/CLAHE/adaptativo con reconocimiento directo antes de un único detector de respaldo.
- La clasificación anverso/reverso usa primero el matching de plantilla; si es concluyente evita dos OCR adicionales de orientación.
- El polling del frontend pasa de 1.5 s a 2.5 s para reducir ruido de red/logs; no afecta el tiempo de OCR.
- No se redujo la resolución canónica ni se eliminaron campos de detección/censura.

# v0.10.6 - Corrección de timeout OCR local

- El análisis ya no crea un proceso `spawn` nuevo por documento de forma predeterminada.
- RapidOCR/ONNX se reutiliza entre análisis mediante `asyncio.to_thread`, reduciendo drásticamente el coste de arranque en Windows.
- El aislamiento por proceso queda disponible con `PRIVADOC_ISOLATE_JOBS=1`.
- El timeout duro del modo aislado sube de 180 s a 900 s por defecto.
- El frontend espera hasta 20 minutos y continúa consultando el estado sin cancelar el trabajo del servidor.

# Changelog

## v0.10.0

- Eliminado Historial local de la navegación, interfaz y JavaScript.
- Conservada Configuración de privacidad.
- Añadida invitación no invasiva a encuesta después de descargar la copia protegida.
- Google Forms configurable mediante `FEEDBACK_FORM_URL`.
- La encuesta nunca se abre automáticamente.
- Añadido botón **Ahora no** y supresión del aviso durante la sesión cuando se descarta o responde.
- Mejoras responsive específicas para 760 px y 420 px.
- Sin cambios funcionales en los motores de detección/censura.
