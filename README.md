# Visión por computadora: asistencia, login facial y hogar inteligente

Tres proyectos de reconocimiento facial que comparten un mismo núcleo (`core/`) y una interfaz web (FastAPI + cámara del navegador):

| # | Proyecto | Qué hace |
|---|----------|----------|
| 1 | **Asistencia automatizada** | Reconoce rostros en tiempo real, registra la asistencia en `data/attendance.csv` (una vez por persona y día), lista presentes/ausentes y simula notificar a tutores/supervisores. |
| 2 | **Login con verificación facial** | El usuario inicia sesión con su rostro: se compara contra los rostros registrados de ese usuario y se emite un token de sesión. Rechaza fotos sin rostro o con varios rostros. |
| 3 | **Hogar inteligente** | Identifica a la persona y su perfil (adulto/niño) y dispara acciones **simuladas** con `print` (luces, música, termostato). Persona desconocida: no activa nada. |

## Cómo funciona

```
imagen/cámara -> detección de rostro (OpenCV) -> embedding 512-D (Facenet512) -> distancia coseno
                                                     └─ <= 0.30: misma persona
```

- `core/engine.py`: decodificación de imagen, detección y embeddings (DeepFace).
- `core/store.py`: personas registradas y búsqueda del rostro más parecido (`data/faces.json`).
- `app/main.py`: API de los tres proyectos. `app/static/index.html`: interfaz.
- `moodmeter/`: proyecto previo de emociones faciales (independiente, ver abajo).

## Requisitos

- Linux o WSL2 (Ubuntu) con **Python 3.11** (TensorFlow aún no soporta 3.14).
- Conexión a internet la primera vez (DeepFace descarga ~90 MB de pesos a `~/.deepface`).

## Instalación y ejecución

```bash
git clone https://github.com/salazar-solutions/computerV
cd computerV
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

(`./start.sh` hace lo mismo automáticamente.) Abre <http://127.0.0.1:8000>.

> **WSL con el repo en `/mnt/d/...`:** crea el entorno en el disco de Linux (mucho más rápido): `VENV_DIR=~/.venvs/computerV ./start.sh`.

> ¿No tienes Python 3.11? Con [uv](https://docs.astral.sh/uv/): `pip install uv && uv venv --python 3.11 .venv && uv pip install -r requirements.txt`.

### Uso

1. **Registro:** activa la cámara (o sube una imagen), escribe un nombre, elige adulto/niño y pulsa *Registrar*. Haz 2-3 registros por persona con distinto ángulo.
2. **Asistencia:** pestaña 1 → *Escanear* o *Auto* (escanea cada 2 s). *Notificar ausentes* imprime el aviso simulado en la consola del servidor.
3. **Login:** pestaña 2 → escribe tu usuario y pulsa el botón; solo entra con tu rostro.
4. **Hogar:** pestaña 3 → las acciones aparecen en pantalla y se imprimen en la consola (`[HOGAR] ...`).

La cámara del navegador funciona también en WSL (el navegador de Windows accede a `127.0.0.1:8000`). Si no tienes cámara, usa **Subir imagen**.

## Decisiones de diseño

Por qué se eligió cada tecnología y enfoque: ver [DECISIONES.md](DECISIONES.md).

## Pruebas

```bash
pytest -q
```

Las pruebas usan detección simulada, así que no necesitan modelos ni cámara.

## Limitaciones

- No hay detección de vida (anti-spoofing): una foto impresa podría engañar al login. Es un demo académico, no usar en producción.
- Las sesiones del login viven en memoria y se pierden al reiniciar.
- Datos biométricos guardados localmente en `data/` (ignorados por git).

## MoodMeter (proyecto previo)

Analizador de emociones faciales independiente en `moodmeter/`:

```bash
python -m uvicorn moodmeter.main:app --port 8001
```
