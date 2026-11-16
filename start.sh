#!/usr/bin/env bash
# Crea el entorno virtual, instala dependencias y arranca el servidor.
#
# Por defecto el entorno es ./.venv. Si el repo está en un disco montado de Windows
# (/mnt/d/...) conviene ponerlo en el disco de Linux, mucho más rápido para TensorFlow:
#   VENV_DIR=~/.venvs/computerV ./start.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
VENV="${VENV_DIR:-.venv}"

if [ ! -x "$VENV/bin/python" ]; then
    echo "[!] Creando entorno virtual en $VENV ..."
    PY=$(command -v python3.11 || command -v python3)
    "$PY" -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install -r requirements.txt
fi

echo "[*] Servidor en http://127.0.0.1:8000  (Ctrl+C para detener)"
echo "[!] La primera vez DeepFace descarga los pesos de Facenet512 (~90 MB)."
exec "$VENV/bin/python" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
