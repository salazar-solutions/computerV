# Decisiones de diseño

Qué elegimos, por qué, y qué descartamos. Sirve para justificar el proyecto ante quien lo califique.

## 1. Alcance: los tres proyectos
**Elegido:** implementar los tres (asistencia, login facial, hogar inteligente).
**Por qué:** comparten ~80 % de la lógica (detectar rostro → embedding → comparar), así que el costo extra es bajo y se cubre cualquier lectura del enunciado ("para cada proyecto...").
**Descartado:** hacer solo uno (riesgo de quedar corto si piden los tres).

## 2. Estructura: monorepo con núcleo compartido
**Elegido:** `core/` (visión por computadora) + `app/` (API e interfaz de los tres proyectos), con **un solo** `requirements.txt` y `README.md`.
**Por qué:** el enunciado pide un único `requirements.txt` y un README con el que el revisor ejecute el proyecto; evita código y dependencias duplicados.
**Descartado:** un repositorio por proyecto (triplica mantenimiento y duplica el código de reconocimiento).

## 3. MoodMeter se conserva en `moodmeter/`
**Por qué:** era trabajo previo que ya funcionaba; se separó para no mezclarlo con la entrega. Usa DeepFace, así que no añade dependencias nuevas.
**Descartado:** borrarlo (se pierde trabajo) o dejarlo en la raíz (confunde al revisor).

## 4. Interfaz: web con FastAPI y cámara del navegador
**Elegido:** FastAPI + una página HTML que usa `getUserMedia`; también permite subir una foto.
**Por qué:**
- En WSL no hay acceso directo a la webcam; el navegador de Windows sí la ve y envía las imágenes al servidor en WSL.
- El revisor puede probarlo sin cámara subiendo fotos.
- Reaprovecha la arquitectura de MoodMeter.

**Descartado:** ventana de OpenCV (`cv2.imshow` + cámara en WSL requiere WSLg/usbipd, frágil) y solo CLI (no muestra tiempo real).

## 5. Librería: DeepFace (modelo Facenet512)
**Elegido:** DeepFace con Facenet512 (vectores de 512 dimensiones) y comparación por distancia coseno.
**Por qué:** se instala solo con `pip` (sin compilar nada), ya estaba en el repo y trae detección + embeddings. Facenet512 da buena precisión con pesos moderados (~90 MB).
**Descartado:** `face_recognition`/dlib (compilar dlib falla con frecuencia en instalaciones nuevas, justo el caso del revisor) y OpenCV LBPH (mucho menos preciso con cambios de luz).

### Parámetros ajustables (`core/engine.py`)
| Parámetro | Valor | Nota |
|-----------|-------|------|
| `MODEL_NAME` | `Facenet512` | Otros: `ArcFace`, `VGG-Face`. Cambiarlo exige re-registrar a todos. |
| `DETECTOR` | `yunet` | Red neuronal pequeña de OpenCV (~230 KB): ~1-2 s en CPU y robusta con lentes, contraluz y poses moderadas. Cambiarlo exige re-registrar a todos. |

**Por qué `yunet` y no `opencv`:** probamos ambos con una foto real (rostro de frente con lentes y contraluz). El detector `opencv` (Haar cascades) no la detectaba y, además, con `opencv-python` 5 ni funciona (falta `cv2.CascadeClassifier`). `yunet` la detectó con confianza 0.93. `retinaface` también acertó, pero tarda ~40 s por imagen en CPU; `ssd` no es compatible con OpenCV 5 y `mtcnn` requiere una dependencia extra.
| `THRESHOLD` | `0.30` | Distancia coseno máxima para aceptar. Es el umbral recomendado por DeepFace para Facenet512 + coseno. Bajarlo = más estricto (menos falsos positivos, más rechazos); subirlo = más permisivo. |

## 6. Almacenamiento: JSON y CSV locales
**Elegido:** `data/faces.json` (personas y embeddings) y `data/attendance.csv` (asistencia).
**Por qué:** el revisor no necesita instalar ni configurar una base de datos; son archivos legibles y fáciles de inspeccionar. Se guardan embeddings, no fotos, y `data/` está en `.gitignore` (son datos biométricos).
**Límite:** no escala a miles de personas; para eso habría que usar SQLite/una base vectorial.

## 7. Python 3.11 y versiones fijas
**Elegido:** Python 3.11 con todas las versiones fijadas en `requirements.txt` (instaladas con `uv` en WSL).
**Por qué:** el Python del sistema (3.14) no es compatible con TensorFlow. Fijar versiones asegura que el revisor obtenga el mismo entorno que probamos.

## 8. Código en el sistema de archivos de WSL (`~/computerV`)
**Por qué:** `pip` y TensorFlow son bastante más lentos sobre `/mnt/d`, además de problemas de permisos (p. ej. se pierde el bit de ejecución de `start.sh`). Se sincroniza con git.

## 9. Funcionalidades simuladas (permitido por el enunciado)
- Notificación a ausentes → `print` en consola del servidor.
- Hogar inteligente (luces, música, termostato) → acciones impresas/mostradas, sin dispositivos reales.
- Lo real (obligatorio): detección, reconocimiento y verificación de rostros.

## 10. Reglas de seguridad del login
- Exige **exactamente un** rostro en la imagen (rechaza grupos).
- Solo compara contra las muestras **del usuario indicado**, no contra toda la base.
- Token de sesión aleatorio (`secrets.token_urlsafe`).

**Limitaciones conocidas:** no hay detección de vida (una foto impresa podría engañarlo), las sesiones viven en memoria y el servidor no usa HTTPS. Es un demo académico, no para producción.

## 11. Pruebas
Los tests (`pytest`) simulan la detección de rostros para ser rápidos y no depender de modelos ni cámara; validan la lógica de asistencia (sin duplicados), login (acepta/rechaza), roles del hogar y errores de entrada. El reconocimiento real se verifica manualmente con la interfaz.
