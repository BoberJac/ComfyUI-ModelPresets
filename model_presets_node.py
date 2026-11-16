"""ComfyUI Model Presets node.

Keeps per-model settings (model file, sage/compile options, base + hires sampler
settings, resolution) in ``model_presets.json`` next to this file and loads the
selected model as MODEL. CLIP/VAE are pass-through inputs: Anima models are
usually loaded as diffusion models, with the text encoder and VAE shared.

The JSON file is the source of truth for the model. The sampler settings are
applied to real KSampler nodes by the frontend (see web/model_presets.js).
"""

import json
import logging
import os
import threading

try:
    import folder_paths
except ImportError:  # imported outside a running ComfyUI (lint, tests)
    folder_paths = None

try:
    import server
    from aiohttp import web
except ImportError:
    server = None
    web = None

LOGGER = logging.getLogger(__name__)

NODE_DIR = os.path.dirname(os.path.realpath(__file__))
PRESETS_FILE = os.path.join(NODE_DIR, "model_presets.json")

_FILE_LOCK = threading.RLock()

DEFAULT_PRESET = {
    "source": "model",
    "model": "",
    "weight_dtype": "default",
    "sage_attention": "auto",
    "torch_compile": False,
    "width": 1024,
    "height": 1536,
    "divide": False,
    "round_to": 16,
    "hires_width": 1536,
    "hires_height": 2304,
    "base": {
        "steps": 25,
        "cfg": 3.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
    },
    "hires": {
        "enabled": False,
        "steps": 15,
        "cfg": 3.0,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
        "denoise": 0.45,
    },
}


def load_presets():
    """Read the preset file. Returns an empty dict on a missing or broken file."""
    with _FILE_LOCK:
        if not os.path.exists(PRESETS_FILE):
            return {}
        try:
            with open(PRESETS_FILE, "r", encoding="utf-8-sig") as handle:
                presets = json.load(handle)
        except (OSError, ValueError) as exc:
            LOGGER.warning("ModelPresets: cannot read %s: %s", PRESETS_FILE, exc)
            return {}
    return presets if isinstance(presets, dict) else {}


def save_presets(presets):
    """Write the preset file atomically so a crash cannot truncate it."""
    temp_path = f"{PRESETS_FILE}.tmp"
    with _FILE_LOCK:
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(presets, handle, indent=2, ensure_ascii=False)
            os.replace(temp_path, PRESETS_FILE)
        except OSError as exc:
            LOGGER.warning("ModelPresets: cannot write %s: %s", PRESETS_FILE, exc)
            return False
    return True


def preset_names():
    return list(load_presets().keys()) or ["Default"]


# --- Optional Anima Booster helpers (sage attention / torch.compile) ---------

def _anima_booster_module(name):
    """Import a submodule of the ANIMA_BOOSTER custom node.

    ComfyUI registers custom node packages under their absolute path, so the
    package is not importable as plain ``ANIMA_BOOSTER``. Find the loaded
    package by its file path and import the submodule inside it.
    """
    import importlib
    import sys

    target = os.path.normcase(os.path.join("ANIMA_BOOSTER", "__init__.py"))
    for module in list(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        if os.path.normcase(os.path.normpath(module_file)).endswith(target):
            try:
                return importlib.import_module(f"{module.__name__}.{name}")
            except ImportError:
                continue
    for package in ("ANIMA_BOOSTER", "ANIMA_x_BOOSTER"):
        try:
            return importlib.import_module(f"{package}.{name}")
        except ImportError:
            continue
    return None


def sage_modes():
    module = _anima_booster_module("core.attention_sage")
    modes = getattr(module, "SAGE_MODES", None)
    if isinstance(modes, (list, tuple)) and modes:
        return list(modes)
    return ["auto", "disabled"]


def apply_sage(model, mode):
    if not mode or mode == "disabled":
        return model
    module = _anima_booster_module("core.attention_sage")
    apply_fn = getattr(module, "apply_sage_attention_to_model", None) if module else None
    if apply_fn is None:
        LOGGER.warning(
            "ModelPresets: sage_attention=%r requested, but ANIMA Booster is not installed; "
            "using native attention.", mode,
        )
        return model
    model, success = apply_fn(model, mode)
    if not success:
        LOGGER.info("ModelPresets: SageAttention unavailable, using native attention")
    return model


def apply_compile(model):
    module = _anima_booster_module("core.compile_utils")
    compile_fn = getattr(module, "detect_and_compile_blocks", None) if module else None
    if compile_fn is None:
        LOGGER.warning("ModelPresets: torch_compile requested, but ANIMA Booster is not installed")
        return model
    try:
        return compile_fn(model, mode="default")
    except Exception as exc:  # noqa: BLE001 - a failed compile must not break loading
        LOGGER.error("ModelPresets: torch.compile failed: %s", exc)
        return model


def load_model(preset):
    """Load the MODEL for a preset. Returns (model, clip, vae)."""
    import comfy.sd
    import comfy.utils
    import torch

    source = preset.get("source", "model")
    sage = preset.get("sage_attention", "auto")
    compile_flag = bool(preset.get("torch_compile", False))

    if source == "checkpoint":
        name = preset.get("checkpoint") or preset.get("model") or ""
        if not name:
            raise ValueError("ModelPresets: no checkpoint file in preset")
        path = folder_paths.get_full_path_or_raise("checkpoints", name)
        out = comfy.sd.load_checkpoint_guess_config(
            path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        model, clip, vae = out[0], out[1], out[2]
        if compile_flag:
            model = apply_compile(model)
        return model, clip, vae

    name = preset.get("model") or ""
    if not name:
        raise ValueError("ModelPresets: no diffusion model file in preset")

    weight_dtype = preset.get("weight_dtype", "default")
    model_options = {}
    if weight_dtype == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2

    path = folder_paths.get_full_path_or_raise("diffusion_models", name)
    model = comfy.sd.load_diffusion_model(path, model_options=model_options)
    model = apply_sage(model, sage)
    if compile_flag:
        model = apply_compile(model)
    return model, None, None


# --- HTTP API ---------------------------------------------------------------

_prompt_server = getattr(getattr(server, "PromptServer", None), "instance", None)

if _prompt_server is not None:
    routes = _prompt_server.routes

    @routes.get("/model_presets/get_presets")
    async def get_presets(request):
        return web.json_response(load_presets())

    @routes.post("/model_presets/save_preset")
    async def save_preset(request):
        try:
            data = await request.json()
        except ValueError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        name = str(data.get("name", "")).strip()
        preset = data.get("preset")
        if not name:
            return web.json_response({"error": "Preset name required"}, status=400)
        if len(name) > 200:
            return web.json_response({"error": "Preset name is too long"}, status=400)
        if not isinstance(preset, dict):
            return web.json_response({"error": "Preset payload must be an object"}, status=400)

        presets = load_presets()
        presets[name] = preset
        if not save_presets(presets):
            return web.json_response({"error": "Failed to save"}, status=500)
        return web.json_response({"success": True, "message": f"Preset '{name}' saved"})

    @routes.post("/model_presets/delete_preset")
    async def delete_preset(request):
        try:
            data = await request.json()
        except ValueError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        name = str(data.get("name", "")).strip()
        if not name:
            return web.json_response({"error": "Invalid preset name"}, status=400)

        presets = load_presets()
        if name not in presets:
            return web.json_response({"error": "Preset not found"}, status=404)
        del presets[name]
        if not save_presets(presets):
            return web.json_response({"error": "Failed to save"}, status=500)
        return web.json_response({"success": True, "message": f"Preset '{name}' deleted"})

    @routes.get("/model_presets/options")
    async def get_options(request):
        import comfy.samplers

        def list_files(folder):
            try:
                return folder_paths.get_filename_list(folder)
            except Exception:  # noqa: BLE001 - a missing folder must not break the UI
                return []

        return web.json_response({
            "models": list_files("diffusion_models"),
            "checkpoints": list_files("checkpoints"),
            "vaes": list_files("vae"),
            "upscale_models": list_files("upscale_models"),
            "samplers": list(comfy.samplers.KSampler.SAMPLERS),
            "schedulers": list(comfy.samplers.KSampler.SCHEDULERS),
            "sage_modes": sage_modes(),
            "weight_dtypes": ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
        })


# --- Node -------------------------------------------------------------------

class ModelPresetsNode:
    """Load a diffusion model or checkpoint selected from a preset."""

    @classmethod
    def INPUT_TYPES(cls):
        names = preset_names()
        return {
            "required": {
                "preset": (names, {"default": names[0]}),
            },
            "optional": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_preset"
    CATEGORY = "loaders"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # The dropdown is built from a file that changes while a workflow is open,
        # so a preset name that no longer exists must not block queueing.
        return True

    @classmethod
    def IS_CHANGED(cls, preset, **kwargs):
        return preset

    def load_preset(self, preset, clip=None, vae=None):
        presets = load_presets()
        data = presets.get(preset)
        if data is None:
            raise ValueError(
                f"ModelPresets: preset {preset!r} not found in {PRESETS_FILE}. "
                "Save the preset in the node panel first."
            )

        model, loaded_clip, loaded_vae = load_model(data)
        LOGGER.info(
            "ModelPresets: preset %r -> %s",
            preset, data.get("model") or data.get("checkpoint") or "?",
        )
        return (
            model,
            clip if clip is not None else loaded_clip,
            vae if vae is not None else loaded_vae,
        )


NODE_CLASS_MAPPINGS = {
    "ModelPresetsNode": ModelPresetsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelPresetsNode": "Model Presets",
}
