"""FastAPI serving (Phase 4) — tanpa web UI, standar OpenAPI."""

from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager

import torch

try:
    from fastapi import FastAPI, UploadFile
    from PIL import Image

    _HAS_FASTAPI = True
except ImportError:
    FastAPI = None
    UploadFile = None
    Image = None
    _HAS_FASTAPI = False

from rak.network.competitive import CompetitiveNetwork, NetworkConfig
from rak.readout.linear import RidgeReadout


def load_model(weights_path, readout_path=None, device="cpu"):
    blob = torch.load(weights_path, map_location=device, weights_only=False)
    state = blob["net"]
    n_e = int(state["w_e"].shape[1])
    net = CompetitiveNetwork(NetworkConfig(n_e=n_e), seed=1, device=device)
    net.load_state_dict(state)
    readout = None
    if readout_path:
        readout = RidgeReadout.load(readout_path)
    return net, readout, blob.get("assignments")


@asynccontextmanager
async def lifespan(app):
    app.state.net = None
    app.state.readout = None
    app.state.assignments = None
    wp = getattr(app.state, "weights_path", None)
    rp = getattr(app.state, "readout_path", None)
    if wp:
        net, readout, assignments = load_model(wp, rp)
        app.state.net = net
        app.state.readout = readout
        app.state.assignments = assignments
    yield
    app.state.net = None
    app.state.readout = None


if _HAS_FASTAPI:
    app = FastAPI(
        lifespan=lifespan,
        title="RAK — Recurrent Adaptive Knowledge",
        description="Pure spiking architecture with local learning. No backprop, no pretrained, no API LLM.",
        version="0.1.0",
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": app.state.net is not None}

    @app.get("/ready")
    async def ready():
        if app.state.net is None:
            return {"status": "not_ready"}
        return {"status": "ready"}

    @app.get("/v1/model/info")
    async def model_info():
        if app.state.net is None:
            return {"error": "no model loaded"}
        net = app.state.net
        return {
            "name": "rak-mnist",
            "n_neurons": net.config.n_e,
            "device": str(net.device),
            "classes": 10,
        }

    @app.post("/v1/predict")
    async def predict(file: UploadFile):
        if app.state.net is None:
            return {"error": "no model loaded"}
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("L").resize((28, 28))
        x = torch.tensor(list(img.getdata()), dtype=torch.float32).unsqueeze(0) / 255.0
        from rak.data.encode import EncoderConfig, present_adaptive
        enc = EncoderConfig()
        gen = torch.Generator().manual_seed(0)
        counts, _ = present_adaptive(app.state.net, x, enc, gen, learn=False)
        if app.state.readout is not None:
            pred = int(app.state.readout.predict(counts).item())
        else:
            pred = 0
        return {"class": pred, "classes": 10}

    @app.post("/v1/generate")
    async def generate(class_idx: int = 0, seed: int = 0):
        if app.state.net is None:
            return {"error": "no model loaded"}
        img = app.state.net.generate_class(app.state.assignments, class_idx, seed=seed)
        b64 = base64.b64encode(img.numpy().tobytes()).decode()
        return {"image": b64, "shape": [28, 28], "format": "raw_float32"}

    @app.post("/v1/reconstruct")
    async def reconstruct(file: UploadFile):
        if app.state.net is None:
            return {"error": "no model loaded"}
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("L").resize((28, 28))
        x = torch.tensor(list(img.getdata()), dtype=torch.float32).unsqueeze(0) / 255.0
        from rak.data.encode import EncoderConfig, present_adaptive
        enc = EncoderConfig()
        gen = torch.Generator().manual_seed(0)
        counts, _ = present_adaptive(app.state.net, x, enc, gen, learn=False)
        rec = app.state.net.reconstruct(counts)
        b64 = base64.b64encode(rec[0].numpy().tobytes()).decode()
        return {"image": b64, "shape": [28, 28], "format": "raw_float32"}
else:
    app = None