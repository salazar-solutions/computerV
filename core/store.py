"""Base de datos local (JSON) de personas registradas y sus embeddings faciales.

Decisiones:
- JSON en vez de una base de datos: el revisor no instala ni configura nada y el
  archivo es legible. No escala a miles de personas (habría que usar SQLite o una
  base vectorial).
- Se guardan embeddings, no fotos: ocupan poco y no permiten reconstruir la imagen
  del rostro. Es dato biométrico, por eso data/ está en .gitignore.
- Varias muestras por persona: registrar el rostro con distinto ángulo/luz y quedarse
  con la distancia mínima a cualquiera de ellas mejora la tasa de aciertos.
"""
import json
import threading
from pathlib import Path

import numpy as np

from .engine import THRESHOLD, cosine_distance

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VALID_ROLES = ("adult", "child")


class FaceStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else DATA_DIR / "faces.json"
        self._lock = threading.Lock()
        self._people: dict[str, dict] = {}
        if self.path.exists():
            self._people = json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._people), encoding="utf-8")

    def enroll(self, name: str, role: str, embedding) -> int:
        """Agrega un embedding a la persona (la crea si no existe). Devuelve cuántas muestras tiene."""
        if role not in VALID_ROLES:
            raise ValueError(f"role debe ser uno de {VALID_ROLES}")
        with self._lock:
            p = self._people.setdefault(name, {"role": role, "embeddings": []})
            p["role"] = role
            p["embeddings"].append([float(x) for x in embedding])
            self._save()
            return len(p["embeddings"])

    def remove(self, name: str) -> bool:
        with self._lock:
            found = self._people.pop(name, None) is not None
            if found:
                self._save()
            return found

    def people(self) -> list[dict]:
        return [
            {"name": n, "role": p["role"], "samples": len(p["embeddings"])}
            for n, p in sorted(self._people.items())
        ]

    def role_of(self, name: str) -> str | None:
        p = self._people.get(name)
        return p["role"] if p else None

    def distance_to(self, name: str, embedding) -> float | None:
        """Verificación 1:1: menor distancia entre el embedding y las muestras de `name`.

        Devuelve None si la persona no existe o no tiene muestras.
        """
        p = self._people.get(name)
        if not p or not p["embeddings"]:
            return None
        return min(cosine_distance(embedding, e) for e in p["embeddings"])

    def identify(self, embedding) -> tuple[str | None, float | None]:
        """Identificación 1:N: busca la persona más parecida en toda la base.

        Devuelve (nombre|None, distancia). Si la mejor distancia supera THRESHOLD se
        devuelve nombre None (desconocido) en lugar del "menos malo": preferimos
        rechazar a confundir a alguien con otra persona. Lo usan asistencia y hogar;
        el login NO usa esto, sino distance_to() (verificación 1:1).
        """
        best_name, best = None, None
        for name in self._people:
            d = self.distance_to(name, embedding)
            if d is not None and (best is None or d < best):
                best_name, best = name, d
        if best is not None and best <= THRESHOLD:
            return best_name, best
        return None, best
