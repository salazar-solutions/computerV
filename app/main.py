"""API FastAPI con los tres proyectos de visión por computadora.

1. Asistencia automática   -> /api/attendance/*
2. Login con verificación facial -> /api/login/*
3. Hogar inteligente       -> /api/home/*
Registro de personas compartido -> /api/people

Decisiones:
- Una sola API para los tres proyectos: comparten registro de personas y motor
  de reconocimiento (core/), y el revisor solo levanta un servidor.
- Los endpoints son `def` (no `async def`) a propósito: la inferencia de DeepFace
  es CPU-bound y bloqueante; FastAPI ejecuta los `def` en un pool de hilos, así el
  servidor sigue respondiendo mientras se procesa una imagen.
- La cámara se captura en el navegador y llega como imagen base64: en WSL no hay
  acceso directo a la webcam, pero el navegador de Windows sí.
- Lo simulado (permitido por el enunciado): notificaciones y acciones del hogar se
  imprimen en consola; lo real es la detección, el reconocimiento y la verificación.
"""
import csv
import secrets
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core import engine
from core.store import DATA_DIR, FaceStore

STATIC_DIR = Path(__file__).resolve().parent / "static"
store = FaceStore()
ATTENDANCE_CSV = DATA_DIR / "attendance.csv"
sessions: dict[str, str] = {}  # token -> usuario (en memoria; es un demo)

app = FastAPI(title="Visión por computadora: asistencia, login y hogar inteligente")


class Image(BaseModel):
    image: str


class Enroll(Image):
    name: str = Field(min_length=1, max_length=60)
    role: str = "adult"


class Verify(Image):
    username: str


def analyze(image: str) -> list[dict]:
    """Decodifica la imagen y detecta rostros; un base64 inválido es error 400."""
    try:
        return engine.detect_faces(engine.decode_image(image))
    except engine.ImageError as exc:
        raise HTTPException(400, str(exc))


def single_face(image: str) -> dict:
    """Exige exactamente un rostro (error 422 si hay cero o varios).

    Se usa al registrar y al iniciar sesión: con varios rostros no sabríamos a quién
    pertenece el registro, y en el login permitiría que otra persona aparezca en el
    encuadre junto al usuario legítimo.
    """
    faces = analyze(image)
    if not faces:
        raise HTTPException(422, "No se detectó ningún rostro. Mejora la luz y mira de frente.")
    if len(faces) > 1:
        raise HTTPException(422, "Se detectó más de un rostro; debe aparecer solo uno.")
    return faces[0]


# ---------- Registro de personas (compartido) ----------
@app.post("/api/people")
def enroll(p: Enroll):
    """Registra un rostro. Llamarlo varias veces con el mismo nombre añade muestras
    (distinto ángulo/luz), lo que mejora la precisión del reconocimiento."""
    face = single_face(p.image)
    try:
        samples = store.enroll(p.name.strip(), p.role, face["embedding"])
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"name": p.name.strip(), "role": p.role, "samples": samples, "box": face["box"]}


@app.get("/api/people")
def list_people():
    return store.people()


@app.delete("/api/people/{name}")
def delete_person(name: str):
    if not store.remove(name):
        raise HTTPException(404, "Persona no registrada")
    return {"deleted": name}


# ---------- Proyecto 1: asistencia ----------
def read_attendance(day: str | None = None) -> list[dict]:
    """Lee el CSV de asistencia (opcionalmente solo un día 'YYYY-MM-DD').

    CSV en vez de base de datos: sin instalación y fácil de abrir en Excel.
    """
    if not ATTENDANCE_CSV.exists():
        return []
    with ATTENDANCE_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if day is None or r["date"] == day]


def mark_present(name: str) -> bool:
    """Registra la asistencia una sola vez por persona y por día."""
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    if any(r["name"] == name for r in read_attendance(today)):
        return False
    ATTENDANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not ATTENDANCE_CSV.exists()
    with ATTENDANCE_CSV.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "time", "name"])
        w.writerow([today, now.strftime("%H:%M:%S"), name])
    return True


@app.post("/api/attendance/scan")
def attendance_scan(p: Image):
    """Proyecto 1: reconoce todos los rostros de la imagen (identificación 1:N) y
    marca asistencia. Admite varios rostros a la vez porque en un salón o una oficina
    aparece más de una persona frente a la cámara."""
    results = []
    for f in analyze(p.image):
        name, dist = store.identify(f["embedding"])
        newly = mark_present(name) if name else False
        results.append(
            {"name": name, "distance": dist, "box": f["box"], "newly_marked": newly}
        )
    return {"faces": results}


@app.get("/api/attendance")
def attendance_today():
    today = datetime.now().strftime("%Y-%m-%d")
    present = read_attendance(today)
    names = {r["name"] for r in present}
    absent = [x["name"] for x in store.people() if x["name"] not in names]
    return {"date": today, "present": present, "absent": absent}


@app.delete("/api/attendance")
def attendance_reset_today():
    """Borra solo los registros de hoy (útil para repetir la demo)."""
    today = datetime.now().strftime("%Y-%m-%d")
    keep = [r for r in read_attendance() if r["date"] != today]
    with ATTENDANCE_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "time", "name"])
        w.writerows([r["date"], r["time"], r["name"]] for r in keep)
    return {"remaining": len(keep)}


@app.post("/api/attendance/notify-absent")
def notify_absent():
    """Simulación: avisaría a padres/supervisores de quien no asistió."""
    absent = attendance_today()["absent"]
    for name in absent:
        print(f"[NOTIFICACIÓN] Avisando a tutor/supervisor: {name} no asistió hoy")
    return {"notified": absent}


# ---------- Proyecto 2: login facial ----------
@app.post("/api/login/verify")
def login_verify(p: Verify):
    """Proyecto 2: verificación 1:1 contra el usuario indicado.

    Se compara solo con las muestras de ese usuario (no se busca en toda la base):
    así un rostro parecido a otra persona no puede entrar como el usuario pedido.
    Limitación: sin detección de vida, una foto impresa podría engañarlo.
    Las sesiones (token aleatorio) viven en memoria: es un demo, no producción.
    """
    if store.role_of(p.username) is None:
        raise HTTPException(404, "Usuario no registrado")
    face = single_face(p.image)
    dist = store.distance_to(p.username, face["embedding"])
    if dist is None or dist > engine.THRESHOLD:
        raise HTTPException(401, "El rostro no coincide con el usuario")
    token = secrets.token_urlsafe(24)
    sessions[token] = p.username
    return {"token": token, "user": p.username, "distance": dist}


@app.get("/api/login/me")
def login_me(token: str):
    user = sessions.get(token)
    if not user:
        raise HTTPException(401, "Sesión inválida")
    return {"user": user}


# ---------- Proyecto 3: hogar inteligente ----------
ACTIONS = {
    "adult": ["Encendiendo luces cálidas", "Poniendo música relajante", "Ajustando termostato a 22 °C"],
    "child": ["Encendiendo luz de colores", "Activando entretenimiento infantil", "Ajustando termostato a 24 °C"],
}


@app.post("/api/home/scan")
def home_scan(p: Image):
    """Proyecto 3: identifica a la persona y dispara acciones según su perfil.

    Usa solo el primer rostro detectado (una persona llega a la puerta). Con un
    desconocido no se activa nada del hogar: falla de forma segura.
    """
    faces = analyze(p.image)
    if not faces:
        raise HTTPException(422, "No se detectó ningún rostro.")
    f = faces[0]
    name, dist = store.identify(f["embedding"])
    if not name:
        actions = ["Persona desconocida: manteniendo la puerta cerrada y notificando a los residentes"]
        role = None
    else:
        role = store.role_of(name)
        actions = ACTIONS[role]
    for a in actions:
        print(f"[HOGAR] {a}")
    return {"name": name, "role": role, "distance": dist, "box": f["box"], "actions": actions}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
