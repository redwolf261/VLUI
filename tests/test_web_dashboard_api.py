from __future__ import annotations

import io

import scripts.web_dashboard as wd


def _multipart(file_bytes: bytes, filename: str = "sample.csv", **fields):
    data = {
        "file": (io.BytesIO(file_bytes), filename),
    }
    data.update(fields)
    return data


def test_start_rejects_invalid_sprt_mode():
    with wd.app.test_client() as client:
        resp = client.post("/api/start", json={"sprt_mode": "bad_mode"})
        assert resp.status_code == 400
        body = resp.get_json()
        assert "Invalid sprt_mode" in body["error"]


def test_reset_rejected_while_running():
    with wd._state_lock:
        wd.state["running"] = True

    with wd.app.test_client() as client:
        resp = client.post("/api/reset")
        assert resp.status_code == 409
        body = resp.get_json()
        assert "Cannot reset" in body["error"]

    with wd._state_lock:
        wd.state = wd._fresh_state()


def test_upload_rejects_non_integer_controls():
    payload = _multipart(
        b"lon,lat\n-8.6,41.15\n",
        n_points="not_an_int",
        n_queries="200",
    )
    with wd.app.test_client() as client:
        resp = client.post("/api/upload", data=payload, content_type="multipart/form-data")
        assert resp.status_code == 400
        body = resp.get_json()
        assert "must be integers" in body["error"]


def test_upload_rejects_oversized_payload():
    huge = b"x" * (wd.MAX_UPLOAD_BYTES + 1)
    payload = _multipart(huge)
    with wd.app.test_client() as client:
        resp = client.post("/api/upload", data=payload, content_type="multipart/form-data")
        assert resp.status_code == 413
        body = resp.get_json()
        assert "too large" in body["error"]
