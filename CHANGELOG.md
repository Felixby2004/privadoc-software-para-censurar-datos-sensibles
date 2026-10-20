# Changelog

## v0.10.1

- Refuerzo OCR estructural de DNI azul, amarillo y electrónico con rescate localizado por ROI.
- Se evita OCR indiscriminado: los rescaneos se activan sólo cuando el campo esperado carece de texto confiable.
- Tarjeta final de encuesta más visible, con mayor jerarquía tipográfica y escala 1–10 destacada.
- Mejora responsive del llamado a encuesta.


## v0.10.0

- Eliminado Historial local de la navegación, interfaz y JavaScript.
- Conservada Configuración de privacidad.
- Añadida invitación no invasiva a encuesta después de descargar la copia protegida.
- Google Forms configurable mediante `FEEDBACK_FORM_URL`.
- La encuesta nunca se abre automáticamente.
- Añadido botón **Ahora no** y supresión del aviso durante la sesión cuando se descarta o responde.
- Mejoras responsive específicas para 760 px y 420 px.
- Sin cambios funcionales en los motores de detección/censura.
