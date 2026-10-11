"""Serveur HTTP local autonome, réservé à la démonstration sur cette machine."""

import argparse
import json
import secrets
import sys
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.inference import (  # noqa: E402
    MAX_UPLOAD,
    InputError,
    analyze,
    load_catalogue,
    load_detector,
    load_recording,
    load_reference,
    trace_points,
)

STATIC = ROOT / "app/static"


class EEGServer(ThreadingHTTPServer):
    """Garde au plus deux analyses en mémoire pendant une heure."""

    daemon_threads = True

    def __init__(self, address):
        self.pipeline, self.metadata = load_detector()
        self.reference = load_reference()
        self.catalogue = load_catalogue()
        self.analyses = OrderedDict()
        self.analysis_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        super().__init__(address, Handler)

    def remember(self, recording, result):
        with self.cache_lock:
            for key, (_, _, created) in list(self.analyses.items()):
                if time.monotonic() - created > 3600:
                    del self.analyses[key]
            key = secrets.token_hex(16)
            result["analysis_id"] = key
            self.analyses[key] = (recording, result, time.monotonic())
            while len(self.analyses) > 2:
                self.analyses.popitem(last=False)
            return result

    def recall(self, key):
        with self.cache_lock:
            value = self.analyses.get(key)
            if value is None or time.monotonic() - value[2] > 3600:
                self.analyses.pop(key, None)
                raise InputError("Cette analyse a expiré. Déposez à nouveau l'enregistrement.")
            return value


class Handler(BaseHTTPRequestHandler):
    """Fichiers statiques explicitement autorisés et API de prédiction."""

    server_version = "LecteurAbsences/2.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(180)

    def log_message(self, *_):
        """Ne journalise ni noms de fichiers ni contenu des analyses."""

    def send(self, status, body, content_type="application/json; charset=utf-8"):
        if isinstance(body, dict):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def valid_host(self):
        port = self.server.server_address[1]
        return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def do_GET(self):
        if not self.valid_host():
            self.send(403, {"error": "Accès local uniquement."})
            return
        target = urlsplit(self.path)
        routes = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        if target.path in routes:
            name, mime = routes[target.path]
            self.send(200, (STATIC / name).read_bytes(), mime)
        elif target.path == "/api/model":
            self.send(200, self.server.metadata)
        elif target.path == "/api/recordings":
            hidden = {"source_sha256", "export_sha256"}
            rows = [
                {k: v for k, v in row.items() if k not in hidden}
                for row in self.server.catalogue.values()
            ]
            self.send(200, {"recordings": rows})
        elif target.path == "/api/health":
            self.send(
                200, {"status": "ready", "configuration": self.server.metadata["configuration"]}
            )
        elif target.path == "/api/trace":
            try:
                query = parse_qs(target.query)
                recording, _, _ = self.server.recall(query.get("id", [""])[0])
                start = float(query.get("start", ["0"])[0])
                end = float(query.get("end", ["30"])[0])
                budget = int(query.get("budget", ["4000"])[0])
                end = min(end, len(recording.signal) / recording.fs)
                if not 0 <= start < end or not 200 <= budget <= 12000:
                    raise InputError("Intervalle du zoom invalide.")
                self.send(200, trace_points(recording, start, end, budget))
            except (InputError, ValueError) as exc:
                self.send(400, {"error": str(exc)})
        else:
            self.send(404, {"error": "Page introuvable."})

    def do_POST(self):
        if not self.valid_host() or self.headers.get("Origin") not in {
            f"http://127.0.0.1:{self.server.server_address[1]}",
            f"http://localhost:{self.server.server_address[1]}",
        }:
            self.send(403, {"error": "L'analyse doit être lancée depuis l'interface locale."})
            return
        if self.path not in {"/api/analyze", "/api/open"}:
            self.send(404, {"error": "Action inconnue."})
            return
        if not self.server.analysis_lock.acquire(blocking=False):
            self.send(409, {"error": "Une analyse est déjà en cours. Patientez puis réessayez."})
            return
        try:
            if self.path == "/api/open":
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024:
                    raise InputError("Requête d'ouverture invalide.")
                body = json.loads(self.rfile.read(length))
                record = body.get("record", "") if isinstance(body, dict) else ""
                record = str(record).strip().upper()
                if record not in self.server.catalogue:
                    raise InputError(
                        f"Identifiant inconnu : {record or '(vide)'}. Exemple : 190304A-E."
                    )
                if not self.server.catalogue[record]["available"]:
                    raise InputError(f"{record} : {self.server.catalogue[record]['reason']}.")
                path = ROOT / "data/exports" / f"{record.replace('-', '_')}.npz"
                payload, filename = path.read_bytes(), path.name
            else:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_UPLOAD:
                    self.send(413, {"error": "Choisissez un fichier non vide de moins de 128 Mio."})
                    return
                filename = (
                    unquote(self.headers.get("X-Filename", "")).replace("\\", "/").split("/")[-1]
                )
                if not filename or len(filename) > 200 or not filename.isprintable():
                    raise InputError("Le nom du fichier est invalide.")
                payload = self.rfile.read(length)
                if len(payload) != length:
                    raise InputError("Le transfert est incomplet. Réessayez le dépôt du fichier.")
            recording = load_recording(payload, filename, self.server.metadata["fs_hz"])
            result = analyze(
                recording,
                self.server.pipeline,
                self.server.metadata,
                self.server.reference,
                self.server.catalogue,
            )
            self.send(200, self.server.remember(recording, result))
        except InputError as exc:
            self.send(400, {"error": str(exc)})
        except (TimeoutError, ValueError):
            self.send(
                400,
                {
                    "error": "Transfert interrompu ou requête incorrecte. Réessayez avec un fichier compatible."
                },
            )
        except Exception:
            self.send(
                500,
                {
                    "error": "L'analyse n'a pas abouti. Vérifiez le fichier et redémarrez l'application si nécessaire."
                },
            )
        finally:
            self.server.analysis_lock.release()


def main():
    parser = argparse.ArgumentParser(description="Interface EEG locale")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with EEGServer(("127.0.0.1", args.port)) as server:
        print(f"Lecteur d'absences prêt : http://127.0.0.1:{server.server_address[1]}", flush=True)
        print("Arrêt : Ctrl+C. Les enregistrements déposés restent sur cette machine.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nApplication arrêtée.", flush=True)


if __name__ == "__main__":
    main()
