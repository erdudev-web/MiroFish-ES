#!/usr/bin/env python3
"""
Enrutador LLM local para MiroFish (compatible con la API de OpenAI).

Reparte las llamadas entre varios proveedores gratuitos respetando los limites
de cada uno (RPM, TPM, RPD, TPD), conmuta ante 429/5xx y persiste el consumo en
disco para poder fraccionar una simulacion en varios dias.

Uso:
    ROUTER_CONFIG=providers.json python scripts/llm_router.py --port 4000
    # En .env de MiroFish:
    #   LLM_BASE_URL=http://127.0.0.1:4000/v1
    #   LLM_API_KEY=router-local        (cualquier texto; no se usa)
    #   LLM_MODEL_NAME=router           (el enrutador sustituye el modelo real)

Las claves reales se leen de variables de entorno (campo key_env), nunca del JSON.
"""

import argparse
import json
import os
import sys
import threading
import time
import uuid
from collections import deque

import httpx
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

SAFETY = float(os.environ.get("ROUTER_SAFETY", "0.85"))  # usa el 85% del limite declarado
MAX_WAIT = float(os.environ.get("ROUTER_MAX_WAIT", "75"))  # espera maxima por capacidad (s)
TOTAL_TOKEN_CAP = int(os.environ.get("ROUTER_TOTAL_TOKEN_CAP", "0"))  # 0 = sin tope global
UPSTREAM_TIMEOUT = float(os.environ.get("ROUTER_UPSTREAM_TIMEOUT", "170"))
STATE_PATH = os.environ.get("ROUTER_STATE", "router_state.json")
DAY = 86400.0


class Provider:
    def __init__(self, cfg):
        self.name = cfg["name"]
        self.base_url = cfg["base_url"].rstrip("/")
        self.key_env = cfg.get("key_env", "")
        self.model = cfg["model"]
        self.rpm = cfg.get("rpm")
        self.tpm = cfg.get("tpm")
        self.rpd = cfg.get("rpd")
        self.tpd = cfg.get("tpd")
        # Tamano maximo de una peticion (tokens estimados); por defecto, el TPM.
        self.max_req = cfg.get("max_req_tokens") or self.tpm
        self.priority = cfg.get("priority", 5)  # menor = preferido a igualdad de carga
        self.extra_headers = cfg.get("extra_headers", {})
        self.events = deque()  # (timestamp, tokens)
        self.cooldown_until = 0.0
        self.errors = 0

    @property
    def key(self):
        return os.environ.get(self.key_env, "") if self.key_env else "none"

    @property
    def enabled(self):
        return bool(self.key)

    def _prune(self, now):
        while self.events and now - self.events[0][0] > DAY:
            self.events.popleft()

    def usage(self, now, window):
        self._prune(now)
        reqs = toks = 0
        for ts, t in self.events:
            if now - ts <= window:
                reqs += 1
                toks += t
        return reqs, toks

    def wait_for(self, tokens, now, size=None):
        """Segundos hasta poder enviar `tokens`; None si esperar no sirve (limite diario)."""
        if now < self.cooldown_until:
            return self.cooldown_until - now
        if self.max_req and (size if size is not None else tokens) > self.max_req * SAFETY:
            return None
        r24, t24 = self.usage(now, DAY)
        if self.rpd and r24 + 1 > self.rpd * SAFETY:
            return None
        if self.tpd and t24 + tokens > self.tpd * SAFETY:
            return None
        wait = 0.0
        rm, tm = self.usage(now, 60)
        if self.rpm and rm + 1 > self.rpm * SAFETY:
            wait = max(wait, self._release(now, lambda e: 1, rm + 1 - self.rpm * SAFETY))
        if self.tpm and tm + tokens > self.tpm * SAFETY:
            wait = max(wait, self._release(now, lambda e: e[1], tm + tokens - self.tpm * SAFETY))
        return wait

    def _release(self, now, weight, excess):
        """Tiempo hasta que salgan de la ventana de 60 s eventos suficientes."""
        acc = 0.0
        for e in self.events:
            if now - e[0] <= 60:
                acc += weight(e)
                if acc >= excess:
                    return max(0.0, 60 - (now - e[0])) + 0.2
        return 5.0

    def headroom(self, now):
        _, t24 = self.usage(now, DAY)
        if self.tpd:
            return 1 - t24 / self.tpd
        r24, _ = self.usage(now, DAY)
        return 1 - r24 / self.rpd if self.rpd else 1.0


LOCK = threading.Lock()
PROVIDERS = []
TOTAL_USED = 0


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    return [Provider(p) for p in cfg["providers"]]


def load_state():
    global TOTAL_USED
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            st = json.load(f)
    except Exception:
        return
    TOTAL_USED = st.get("total_used", 0)
    now = time.time()
    for p in PROVIDERS:
        for ts, t in st.get("providers", {}).get(p.name, {}).get("events", []):
            if now - ts <= DAY:
                p.events.append((ts, t))


def save_state():
    st = {"total_used": TOTAL_USED, "providers": {p.name: {"events": list(p.events)} for p in PROVIDERS}}
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f)
    os.replace(tmp, STATE_PATH)


def estimate_input(body):
    text = json.dumps(body.get("messages", []), ensure_ascii=False)
    if body.get("tools"):
        text += json.dumps(body["tools"], ensure_ascii=False)
    return int(len(text) / 3.2)


def estimate_tokens(body):
    text = json.dumps(body.get("messages", []), ensure_ascii=False)
    if body.get("tools"):
        text += json.dumps(body["tools"], ensure_ascii=False)
    out = body.get("max_tokens") or body.get("max_completion_tokens") or 512
    return int(len(text) / 3.2) + min(int(out), 1500)


def pick(tokens, exclude, size=None):
    """Devuelve (provider, wait). provider=None y wait=None -> sin capacidad diaria."""
    now = time.time()
    best, best_wait, soonest = None, None, None
    with LOCK:
        for p in PROVIDERS:
            if not p.enabled or p.name in exclude:
                continue
            w = p.wait_for(tokens, now, size)
            if w is None:
                continue
            if w == 0:
                score = (-p.headroom(now), p.priority)
                if best is None or score < best[0]:
                    best = (score, p)
            elif soonest is None or w < soonest:
                soonest = w
    if best:
        return best[1], 0.0
    return None, soonest


def record(p, tokens):
    global TOTAL_USED
    with LOCK:
        p.events.append((time.time(), tokens))
        TOTAL_USED += tokens
        save_state()


def sanitize_tools(payload):
    """Normaliza esquemas de herramientas para proveedores estrictos (p. ej. Groq).

    Algunos clientes (camel/OASIS) envian parametros con 'required' pero sin
    'properties'; Groq responde 400. Se completa 'properties' y se descarta un
    'required' vacio o invalido.
    """
    tools = payload.get("tools")
    if not tools:
        return
    fixed = []
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        params = fn.get("parameters") if isinstance(fn, dict) else None
        if isinstance(params, dict):
            params = dict(params)
            if params.get("type") == "object" or "type" not in params:
                params.setdefault("type", "object")
                if not isinstance(params.get("properties"), dict):
                    params["properties"] = {}
                req = params.get("required")
                if not isinstance(req, list) or not req:
                    params.pop("required", None)
            t = {**t, "function": {**fn, "parameters": params}}
        fixed.append(t)
    payload["tools"] = fixed


def forward(p, body, inp=0):
    payload = dict(body)
    sanitize_tools(payload)
    # Ajusta max_tokens para que entrada + salida quepan en el limite por peticion del proveedor
    if p.max_req:
        req = payload.get("max_tokens") or payload.get("max_completion_tokens") or 1024
        room = int(p.max_req * 0.97) - inp
        payload.pop("max_completion_tokens", None)
        payload["max_tokens"] = max(200, min(int(req), room))
    payload["model"] = p.model
    payload["stream"] = False
    payload.pop("stream_options", None)
    headers = {"Authorization": f"Bearer {p.key}", "Content-Type": "application/json", **p.extra_headers}
    return httpx.post(f"{p.base_url}/chat/completions", json=payload, headers=headers, timeout=UPSTREAM_TIMEOUT)


def error(status, msg, reason, retry_after=None):
    print(f"[router] {status} {reason}: {msg}", file=sys.stderr, flush=True)
    resp = jsonify({"error": {"message": msg, "type": reason, "code": status}})
    resp.status_code = status
    resp.headers["X-Router-Reason"] = reason
    if retry_after is not None:
        resp.headers["Retry-After"] = str(int(retry_after) + 1)
    return resp


@app.post("/v1/chat/completions")
def chat():
    body = request.get_json(force=True)
    want_stream = bool(body.get("stream"))
    tokens = estimate_tokens(body)
    inp = estimate_input(body)
    if TOTAL_TOKEN_CAP and TOTAL_USED + tokens > TOTAL_TOKEN_CAP:
        return error(429, "Tope global de tokens del enrutador alcanzado", "global_cap", 3600)

    tried, deadline = set(), time.time() + MAX_WAIT
    last = None
    for _ in range(8):
        p, wait = pick(tokens, tried, inp + 400)
        if p is None:
            if wait is None:
                reason = "daily_budget_or_size"
                return error(429, f"Sin capacidad diaria o peticion demasiado grande (~{tokens} tokens)", reason, 900)
            if time.time() + wait > deadline:
                return error(429, "Todos los proveedores en espera por limite por minuto", "rate_wait", wait)
            time.sleep(wait)
            continue
        try:
            r = forward(p, body, inp)
        except httpx.HTTPError as e:
            p.errors += 1
            p.cooldown_until = time.time() + 20
            tried.add(p.name)
            last = f"{p.name}: {e}"
            continue
        if r.status_code == 200:
            data = r.json()
            used = (data.get("usage") or {}).get("total_tokens") or tokens
            record(p, used)
            data["router_provider"] = p.name
            if want_stream:
                return stream_response(data)
            resp = jsonify(data)
            resp.headers["X-Router-Provider"] = p.name
            return resp
        # Errores del proveedor: 429/413 -> enfriar y probar otro; 5xx -> probar otro
        tried.add(p.name)
        last = f"{p.name}: HTTP {r.status_code} {r.text[:200]}"
        if r.status_code in (429, 413):
            ra = float(r.headers.get("retry-after", "30") or 30)
            p.cooldown_until = time.time() + min(ra, 3600)
            record(p, 0)
        elif r.status_code >= 500:
            p.cooldown_until = time.time() + 20
        elif r.status_code in (400, 401, 403, 404):
            return Response(r.content, status=r.status_code, content_type="application/json")
    return error(429, f"Ningun proveedor pudo atender la peticion. Ultimo: {last}", "all_failed", 60)


def stream_response(data):
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    chunk = {
        "id": data.get("id", "chatcmpl-" + uuid.uuid4().hex[:8]),
        "object": "chat.completion.chunk",
        "created": data.get("created", int(time.time())),
        "model": data.get("model", "router"),
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": msg.get("content"),
                                              "tool_calls": msg.get("tool_calls")},
                     "finish_reason": choice.get("finish_reason", "stop")}],
    }
    body = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n"
    return Response(body, content_type="text/event-stream")


@app.get("/v1/models")
def models():
    return jsonify({"object": "list", "data": [{"id": "router", "object": "model", "owned_by": "local"}]})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "providers_enabled": [p.name for p in PROVIDERS if p.enabled]})


@app.get("/router/status")
def status():
    now = time.time()
    out = []
    with LOCK:
        for p in PROVIDERS:
            r24, t24 = p.usage(now, DAY)
            out.append({
                "name": p.name, "model": p.model, "enabled": p.enabled,
                "requests_24h": r24, "tokens_24h": t24,
                "tpd": p.tpd, "rpd": p.rpd,
                "tokens_left_24h": (p.tpd - t24) if p.tpd else None,
                "cooldown_s": max(0, round(p.cooldown_until - now)), "errors": p.errors,
            })
    return jsonify({"total_tokens_used": TOTAL_USED, "cap": TOTAL_TOKEN_CAP or None, "providers": out})


def main():
    global PROVIDERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.environ.get("ROUTER_CONFIG", "providers.json"))
    ap.add_argument("--port", type=int, default=4000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    PROVIDERS = load_config(args.config)
    load_state()
    on = [p.name for p in PROVIDERS if p.enabled]
    print(f"[router] proveedores activos: {on or 'ninguno (faltan claves)'}", file=sys.stderr)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
