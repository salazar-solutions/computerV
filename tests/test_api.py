"""Pruebas de la API con detección de rostros simulada (no requieren modelos ni cámara)."""
import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from core import engine
from core.store import FaceStore

BOX = {"x": 1, "y": 2, "w": 3, "h": 4}


def vec(seed):
    return np.random.default_rng(seed).normal(size=512).astype(np.float32)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", FaceStore(tmp_path / "faces.json"))
    monkeypatch.setattr(main, "ATTENDANCE_CSV", tmp_path / "attendance.csv")
    main.sessions.clear()
    return TestClient(main.app)


def img():
    ok, buf = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def faces_are(monkeypatch, *vectors):
    monkeypatch.setattr(
        engine, "detect_faces", lambda _img: [{"embedding": v, "box": BOX} for v in vectors]
    )


def enroll(client, monkeypatch, name, seed, role="adult"):
    faces_are(monkeypatch, vec(seed))
    return client.post("/api/people", json={"name": name, "role": role, "image": img()})


def test_enroll_and_list(client, monkeypatch):
    assert enroll(client, monkeypatch, "ana", 1).status_code == 200
    assert client.get("/api/people").json() == [{"name": "ana", "role": "adult", "samples": 1}]


def test_enroll_rejects_no_face_and_bad_image(client, monkeypatch):
    faces_are(monkeypatch)
    assert client.post("/api/people", json={"name": "x", "image": img()}).status_code == 422
    assert client.post("/api/people", json={"name": "x", "image": "no-es-imagen"}).status_code == 400


def test_attendance_marks_once_and_lists_absent(client, monkeypatch):
    enroll(client, monkeypatch, "ana", 1)
    enroll(client, monkeypatch, "luis", 2)
    faces_are(monkeypatch, vec(1))
    first = client.post("/api/attendance/scan", json={"image": img()}).json()["faces"][0]
    second = client.post("/api/attendance/scan", json={"image": img()}).json()["faces"][0]
    assert first["name"] == "ana" and first["newly_marked"] is True
    assert second["newly_marked"] is False
    day = client.get("/api/attendance").json()
    assert [r["name"] for r in day["present"]] == ["ana"]
    assert day["absent"] == ["luis"]
    assert client.post("/api/attendance/notify-absent").json() == {"notified": ["luis"]}


def test_attendance_unknown_face(client, monkeypatch):
    enroll(client, monkeypatch, "ana", 1)
    faces_are(monkeypatch, vec(99))
    face = client.post("/api/attendance/scan", json={"image": img()}).json()["faces"][0]
    assert face["name"] is None


def test_login_success_and_failure(client, monkeypatch):
    enroll(client, monkeypatch, "ana", 1)
    faces_are(monkeypatch, vec(1))
    ok = client.post("/api/login/verify", json={"username": "ana", "image": img()})
    assert ok.status_code == 200
    assert client.get("/api/login/me", params={"token": ok.json()["token"]}).json() == {"user": "ana"}
    faces_are(monkeypatch, vec(2))
    assert client.post("/api/login/verify", json={"username": "ana", "image": img()}).status_code == 401
    assert client.post("/api/login/verify", json={"username": "zz", "image": img()}).status_code == 404
    assert client.get("/api/login/me", params={"token": "falso"}).status_code == 401


def test_login_rejects_multiple_faces(client, monkeypatch):
    enroll(client, monkeypatch, "ana", 1)
    faces_are(monkeypatch, vec(1), vec(2))
    assert client.post("/api/login/verify", json={"username": "ana", "image": img()}).status_code == 422


def test_home_roles(client, monkeypatch, capsys):
    enroll(client, monkeypatch, "ana", 1, "adult")
    enroll(client, monkeypatch, "leo", 2, "child")
    faces_are(monkeypatch, vec(1))
    adult = client.post("/api/home/scan", json={"image": img()}).json()
    assert adult["role"] == "adult" and any("música relajante" in a for a in adult["actions"])
    faces_are(monkeypatch, vec(2))
    child = client.post("/api/home/scan", json={"image": img()}).json()
    assert child["role"] == "child" and any("infantil" in a for a in child["actions"])
    faces_are(monkeypatch, vec(50))
    assert client.post("/api/home/scan", json={"image": img()}).json()["name"] is None
    assert "[HOGAR]" in capsys.readouterr().out


def test_attendance_reset_today(client, monkeypatch):
    enroll(client, monkeypatch, "ana", 1)
    faces_are(monkeypatch, vec(1))
    client.post("/api/attendance/scan", json={"image": img()})
    assert client.delete("/api/attendance").status_code == 200
    assert client.get("/api/attendance").json()["present"] == []
