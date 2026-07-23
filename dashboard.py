"""Live training monitor for catch-rl — stdlib only, in the repo's from-scratch spirit.

Serves dashboard.html plus three tiny JSON endpoints over the per-step JSONL
logs that train.py appends to. Nothing is cached: the log is re-read on every
request, so the page tracks a run in flight.

    uv run python dashboard.py [--port 8765] [--root .]

then open http://<host>:<port>/ (works over LAN/tailscale; the page is fully
self-contained, no external requests).

Endpoints:
    GET /                → dashboard.html (served from this script's directory)
    GET /api/runs        → run list: `runs/` and any `runs-*/` under --root that
                           hold a log.jsonl, plus bare *.jsonl files in those
                           dirs (e.g. runs-studio/run3-log.jsonl), newest first
    GET /api/log?run=…   → the run's JSONL parsed to a JSON array (torn/partial
                           lines are skipped, since training may be mid-append)
    GET /api/evals?run=… → per-(model, fruit) catch rates aggregated from the
                           run dir's eval-episodes.jsonl, if present
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
ROOT = HERE  # overwritten in main() from --root


# ---------------------------------------------------------------- discovery

def candidate_dirs(root):
    """`runs/`, every `runs-*/`, and root itself (so --root runs-studio works)."""
    dirs = [root]
    named = root / "runs"
    if named.is_dir():
        dirs.append(named)
    dirs += sorted(p for p in root.glob("runs-*") if p.is_dir())
    # dedupe while keeping order (root may itself be a runs dir)
    seen, out = set(), []
    for d in dirs:
        r = d.resolve()
        if r not in seen:
            seen.add(r)
            out.append(d)
    return out


def discover_runs(root):
    """Each *.jsonl step log is a 'run'. log.jsonl is the canonical name; any
    other .jsonl in a runs dir (run3-log.jsonl, log.smoke1.jsonl, partials)
    counts as a fallback so studio copies show up too. eval-episodes.jsonl is
    a companion file, never a run."""
    runs = []
    for d in candidate_dirs(root):
        for f in sorted(d.glob("*.jsonl")):
            if f.name == "eval-episodes.jsonl":
                continue
            try:
                stat = f.stat()
                with open(f, "rb") as fh:
                    steps = sum(1 for _ in fh)
            except OSError:
                continue
            rel = f.relative_to(root).as_posix()
            runs.append({
                "id": rel,
                # "runs" for runs/log.jsonl; otherwise keep the file visible
                "name": rel[:-len("/log.jsonl")] if rel.endswith("/log.jsonl") else rel,
                "steps": steps,
                "mtime": stat.st_mtime,
                "age_s": max(0.0, time.time() - stat.st_mtime),
            })
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def resolve_run(root, run_id):
    """Map a run id to its log file, refusing anything outside --root."""
    if not run_id:
        return None
    target = (root / run_id).resolve()
    if not (target == root or root in target.parents):  # path-traversal guard
        return None
    if target.is_dir():
        target = target / "log.jsonl"
    if target.suffix != ".jsonl" or not target.is_file():
        return None
    return target


def read_jsonl(path):
    """Parse a JSONL file, skipping unparseable lines — the final line may be
    torn mid-write by the training process."""
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return rows


def aggregate_evals(log_path):
    """Per-(model, fruit) catch rates from eval-episodes.jsonl beside the log.
    Preserves first-seen order for both axes (train fruits come first in
    train.py's eval loop, held-out after)."""
    ep_file = log_path.parent / "eval-episodes.jsonl"
    if not ep_file.is_file():
        return {"models": [], "fruits": [], "rates": {}}
    models, fruits, acc = [], [], {}
    for row in read_jsonl(ep_file):
        m, fr = row.get("model"), row.get("fruit")
        if m is None or fr is None:
            continue
        if m not in acc:
            acc[m] = {}
            models.append(m)
        if fr not in fruits:
            fruits.append(fr)
        caught, n = acc[m].get(fr, (0, 0))
        acc[m][fr] = (caught + int(row.get("caught", 0)), n + 1)
    rates = {m: {fr: {"rate": c / n, "n": n} for fr, (c, n) in by.items() if n}
             for m, by in acc.items()}
    return {"models": models, "fruits": fruits, "rates": rates}


# ------------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        run = q.get("run", [None])[0]
        try:
            if url.path == "/":
                self.send_file(HERE / "dashboard.html", "text/html; charset=utf-8")
            elif url.path == "/api/runs":
                self.send_json({"runs": discover_runs(ROOT)})
            elif url.path == "/api/log":
                log = resolve_run(ROOT, run)
                if log is None:
                    self.send_json({"error": "unknown run"}, status=404)
                else:
                    self.send_json(read_jsonl(log))
            elif url.path == "/api/evals":
                log = resolve_run(ROOT, run)
                if log is None:
                    self.send_json({"error": "unknown run"}, status=404)
                else:
                    self.send_json(aggregate_evals(log))
            else:
                self.send_json({"error": "not found"}, status=404)
        except BrokenPipeError:
            pass  # client went away mid-response; nothing to do

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, ctype):
        try:
            body = path.read_bytes()
        except OSError:
            return self.send_json({"error": f"{path.name} not found"}, status=404)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass


def main():
    global ROOT
    ap = argparse.ArgumentParser(description="catch-rl live training monitor")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", default=".", help="directory containing runs/ (or the runs dir itself)")
    args = ap.parse_args()
    ROOT = Path(args.root).resolve()
    if not ROOT.is_dir():
        raise SystemExit(f"--root {args.root!r} is not a directory")
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"catch-rl monitor: http://0.0.0.0:{args.port}/  (root: {ROOT})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
