"""Motor de visión por computadora compartido por los tres proyectos.

Pipeline: imagen -> detección de rostros -> embedding (vector 512-D) -> distancia coseno.

Decisiones (ver también DECISIONES.md):
- DeepFace: se instala solo con pip (sin compilar dlib, que suele fallar en
  instalaciones nuevas) y ya integra detector + modelo de embeddings.
- Embeddings en vez de comparar píxeles: un vector de 512 números resume la
  identidad del rostro y es robusto a luz y pose moderadas; comparar dos rostros
  se reduce a medir la distancia entre dos vectores.
- Un solo módulo compartido: asistencia, login y hogar usan exactamente la misma
  detección/comparación y solo cambian las reglas de negocio de cada uno.
"""
import base64
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # DeepFace usa tf-keras

import cv2
import numpy as np

# Facenet512: buena precisión con pesos moderados (~90 MB). Si se cambia el modelo
# hay que volver a registrar a todas las personas (los embeddings no son compatibles).
MODEL_NAME = "Facenet512"
# yunet: detector de OpenCV basado en una red neuronal pequeña (modelo ~230 KB).
# Es rápido (~1-2 s en CPU) y robusto con lentes, contraluz y poses moderadas.
# Se descartó "opencv" (Haar cascades): es poco robusto y, con opencv-python 5,
# ni siquiera funciona (falta cv2.CascadeClassifier). También se descartó
# "retinaface": igual de preciso pero ~40 s por imagen en CPU.
DETECTOR = "yunet"
# Distancia coseno máxima para considerar que dos rostros son la misma persona.
# 0.30 es el umbral recomendado por DeepFace para Facenet512 + coseno.
# Más bajo = más estricto (menos falsos positivos, más rechazos); más alto = más permisivo.
THRESHOLD = 0.30


class ImageError(ValueError):
    """La imagen recibida no se pudo decodificar."""


def decode_image(data: str) -> np.ndarray:
    """Convierte un DataURL/base64 en una imagen BGR de OpenCV."""
    try:
        if "," in data:
            data = data.split(",", 1)[1]
        buf = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise ImageError("Imagen base64 inválida") from exc
    if img is None:
        raise ImageError("No se pudo decodificar la imagen")
    return img


def detect_faces(img: np.ndarray) -> list[dict]:
    """Detecta todos los rostros de la imagen y calcula su embedding.

    Devuelve una lista de {embedding, box:{x,y,w,h}}; vacía si no hay rostros.
    Se usa enforce_detection=True para no generar embeddings de zonas que no son
    un rostro (un "rostro" falso podría coincidir por azar con alguien registrado);
    DeepFace lo señala con ValueError, que aquí se traduce a lista vacía.
    align=True endereza el rostro usando los ojos, lo que mejora la comparación.
    DeepFace se importa dentro de la función porque tarda en cargar (TensorFlow):
    así el servidor y las pruebas arrancan rápido.
    """
    from deepface import DeepFace

    try:
        results = DeepFace.represent(
            img_path=img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True,
            align=True,
        )
    except ValueError as exc:
        # Solo "no hay rostro" es un caso normal. Cualquier otro error (modelo,
        # dependencias, versión de OpenCV...) debe verse, no disfrazarse de "sin rostro".
        if "Face could not be detected" in str(exc):
            return []
        raise
    faces = []
    for r in results:
        a = r["facial_area"]
        faces.append(
            {
                "embedding": np.asarray(r["embedding"], dtype=np.float32),
                "box": {k: int(a[k]) for k in ("x", "y", "w", "h")},
            }
        )
    return faces


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - similitud coseno: 0 = mismo vector, ~1 = sin relación.

    Se usa coseno (no euclidiana) porque mide el ángulo entre vectores e ignora su
    magnitud, que es la métrica con la que se calibró el umbral de Facenet512.
    """
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
