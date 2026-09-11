"""Local CLIP inference for unannotated point-cloud candidate renderings.

``load_clip`` downloads an official public checkpoint once and pins its resolved
Hub commit. ``score_candidate`` passes only rendered PNG pixels and the supplied
class descriptions to CLIP. Candidate labels, review notes, and LAS files never
enter the model. There are no uploads, hosted inference calls, or paid APIs.

Cosine similarities are not calibrated probabilities. CLIP was trained using
image/text pairs, so sparse point-cloud renderings are a different visual domain;
rankings and agreement across viewpoints require manual review.

References:
https://huggingface.co/openai/clip-vit-base-patch32
https://huggingface.co/docs/transformers/model_doc/clip
"""

from __future__ import annotations

import hashlib
import fnmatch
import importlib.metadata
import json
import os
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CHECKPOINT = "openai/clip-vit-base-patch32"
PREPROCESS_VERSION = "clip-official-processor-slow-rgb-l2-mean-cosine-v1"
MAX_VIEWS = 12
MAX_DESCRIPTIONS = 100
_MODELS: dict[tuple, dict] = {}


def _privacy() -> None:
    # Configure before importing Hugging Face. These affect downloads, not pixels.
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _versions() -> dict:
    versions = {"python": platform.python_version()}
    for package in ("torch", "torchvision", "transformers", "huggingface_hub", "numpy", "Pillow", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _dependencies() -> tuple:
    _privacy()
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except Exception as error:
        raise RuntimeError(
            "Local CLIP cannot load its dependencies in this kernel. Select the project's "
            ".venv-ground kernel with CPU torch/torchvision, transformers<5, Pillow and "
            f"huggingface_hub installed. Actual import error: {type(error).__name__}: {error}"
        ) from error
    return torch, CLIPModel, CLIPProcessor


def _checkpoint_info(root: Path, checkpoint: str) -> dict:
    """Resolve a public OpenAI checkpoint once; subsequent calls use its pinned commit."""
    _privacy()
    if not isinstance(checkpoint, str) or not checkpoint.startswith("openai/") or checkpoint.count("/") != 1:
        raise ValueError("Choose an official public openai/ CLIP checkpoint, e.g. openai/clip-vit-base-patch32")
    model_cache = root / ".cache-object/models"
    model_cache.mkdir(parents=True, exist_ok=True)
    info_path = model_cache / f"checkpoint_{hashlib.sha256(checkpoint.encode()).hexdigest()[:20]}.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("checkpoint") != checkpoint or not re.fullmatch(r"[0-9a-f]{40}", info.get("resolved_revision", "")):
            raise ValueError(f"Invalid pinned checkpoint metadata: {info_path}")
        return info
    try:
        from huggingface_hub import HfApi
        remote = HfApi(token=False).model_info(checkpoint)
    except Exception as error:
        raise RuntimeError(
            f"Cannot resolve the public checkpoint {checkpoint}; no local pinned metadata exists yet. "
            f"The first run needs a download connection to huggingface.co. Actual error: {error}"
        ) from error
    if not re.fullmatch(r"[0-9a-f]{40}", remote.sha or ""):
        raise RuntimeError(f"The Hub did not provide a resolved commit for {checkpoint}")
    files = sorted(item.rfilename for item in remote.siblings)
    safe_files = [name for name in files if name.endswith(".safetensors")]
    if safe_files:
        weight_format = "safetensors"
    elif "pytorch_model.bin" in files:
        weight_format = "pytorch_bin"
    else:
        raise RuntimeError(f"No supported PyTorch checkpoint weights are published for {checkpoint}")
    info = {"checkpoint": checkpoint, "resolved_revision": remote.sha,
            "weight_format": weight_format, "published_files": files,
            "resolved_utc": datetime.now(timezone.utc).isoformat(),
            "revision_policy": "resolved public Hub commit pinned locally; cached runs do not follow main"}
    with info_path.open("x", encoding="utf-8") as stream:
        json.dump(info, stream, indent=2)
    return info


def load_clip(project_dir: str | Path, checkpoint: str = DEFAULT_CHECKPOINT,
              device: str = "cpu") -> dict:
    """Load the real pretrained CLIP model, reusing it within the current process.

    The default public model is roughly 600 MB. Downloads and metadata stay in
    ``.cache-object/models``. Model and processor load from the exact downloaded
    commit with remote code disabled. CPU inference uses at most eight threads.
    A checkpoint with only ``pytorch_model.bin`` requires torch >= 2.6 and uses
    ``weights_only=True``. No randomly initialized fallback is provided.
    """
    started = time.perf_counter()
    root = Path(project_dir).resolve()
    torch, model_type, processor_type = _dependencies()
    if device != "cpu" and not re.fullmatch(r"cuda(?::\d+)?", device):
        raise ValueError("device must be 'cpu' or an available CUDA device")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this kernel has no available CUDA device; use device='cpu'")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    info = _checkpoint_info(root, checkpoint)
    key = (str(root), checkpoint, info["resolved_revision"], device)
    if key in _MODELS:
        return {**_MODELS[key], "model_reused": True, "load_runtime_seconds": 0.0}
    if info["weight_format"] == "pytorch_bin":
        version = tuple(int(p) for p in re.match(r"(\d+)\.(\d+)", torch.__version__).groups())
        if version < (2, 6):
            raise RuntimeError(f"{checkpoint} publishes PyTorch .bin weights; torch >= 2.6 is required (installed {torch.__version__})")
    try:
        from huggingface_hub import snapshot_download
        patterns = ["config.json", "preprocessor_config.json", "tokenizer*.json",
                    "special_tokens_map.json", "vocab.json", "merges.txt"] + (
                        ["*.safetensors", "*.safetensors.index.json"]
                        if info["weight_format"] == "safetensors" else ["pytorch_model.bin"])
        local_snapshot = (root / ".cache-object/models/hub" /
                          ("models--" + checkpoint.replace("/", "--")) /
                          "snapshots" / info["resolved_revision"])
        required = [name for name in info["published_files"]
                    if any(fnmatch.fnmatch(name, pattern) for pattern in patterns)]
        if required and all((local_snapshot / name).is_file() for name in required):
            snapshot = str(local_snapshot)
        else:
            snapshot = snapshot_download(
                repo_id=checkpoint, revision=info["resolved_revision"],
                cache_dir=str(root / ".cache-object/models/hub"), token=False,
                allow_patterns=patterns,
            )
        processor = processor_type.from_pretrained(snapshot, local_files_only=True,
                                                   use_fast=False, trust_remote_code=False)
        model = model_type.from_pretrained(snapshot, local_files_only=True,
                                           use_safetensors=info["weight_format"] == "safetensors",
                                           weights_only=True, trust_remote_code=False)
        model = model.to(device).eval()
    except Exception as error:
        raise RuntimeError(
            f"Failed to download/load {checkpoint}@{info['resolved_revision']} locally. "
            f"Cache directory: {root / '.cache-object/models'}. "
            f"Actual error: {type(error).__name__}: {error}"
        ) from error
    preprocessing = {"version": PREPROCESS_VERSION,
                     "image_processor": processor.image_processor.to_dict(),
                     "tokenizer_class": type(processor.tokenizer).__name__,
                     "text_max_tokens": int(model.config.text_config.max_position_embeddings),
                     "rgb_conversion": "Pillow convert('RGB')", "embedding_normalization": "L2",
                     "aggregation": "arithmetic mean of per-view cosine similarities; no softmax"}
    bundle = {"model": model, "processor": processor, "checkpoint": checkpoint,
              "resolved_revision": info["resolved_revision"], "revision": info["resolved_revision"],
              "weight_format": info["weight_format"], "versions": _versions(),
              "device": device, "cpu_threads": int(torch.get_num_threads()),
              "hardware": {"device": device, "cpu_threads": int(torch.get_num_threads()),
                           "processor": platform.processor(), "machine": platform.machine()},
              "preprocessing": preprocessing, "snapshot_path": snapshot,
              "load_runtime_seconds": time.perf_counter() - started, "model_reused": False}
    bundle["initial_load_runtime_seconds"] = bundle["load_runtime_seconds"]
    _MODELS[key] = bundle
    return bundle


def _rank(scores: np.ndarray, prompts: list[str]) -> list[dict]:
    order = np.argsort(-scores, kind="stable")
    return [{"rank": rank, "class_description": prompts[int(index)],
             "cosine_similarity": float(scores[index])}
            for rank, index in enumerate(order, start=1)]


def score_candidate(project_dir: str | Path, candidate: dict, render_record: dict,
                    class_descriptions: list[str], model_bundle: dict | None = None,
                    checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cpu") -> dict:
    """Return real CLIP cosine rankings for each view and their mean.

    Inference is cached by the actual ordered PNG bytes, render key, resolved
    checkpoint commit, exact text descriptions, this helper's code hash, and
    preprocessing/library versions. Human labels/notes are not inputs or keys.
    Cache hits are checked before loading model weights whenever possible.
    ``ranking`` includes every supplied class; ``top5`` and ``per_view[*].top3``
    make small notebook tables. Disagreement is the fraction of viewpoints whose
    top class differs from the aggregate top class. This is not accuracy.
    """
    started = time.perf_counter()
    root = Path(project_dir).resolve()
    _privacy()
    if (not isinstance(class_descriptions, (list, tuple)) or
            not 2 <= len(class_descriptions) <= MAX_DESCRIPTIONS or
            any(not isinstance(text, str) or not text.strip() for text in class_descriptions)):
        raise ValueError(f"Supply 2 to {MAX_DESCRIPTIONS} nonempty English class descriptions")
    prompts = list(class_descriptions)
    if len(set(prompts)) != len(prompts):
        raise ValueError("Each class description must be unique")
    candidate_id = str(candidate.get("id", candidate.get("candidate_id", "")))
    if not candidate_id:
        raise ValueError("candidate must have an id or candidate_id")
    if render_record.get("candidate_id", candidate_id) != candidate_id:
        raise ValueError("Render record belongs to a different candidate")
    paths = [Path(path) if Path(path).is_absolute() else root / path for path in render_record["paths"]]
    views = render_record["views"]
    if not 1 <= len(paths) <= MAX_VIEWS or len(paths) != len(views):
        raise ValueError(f"Provide 1 to {MAX_VIEWS} rendered PNG paths and one view descriptor per path")
    image_hashes = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Candidate render is missing: {path}")
        image_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    if model_bundle is not None:
        if model_bundle["checkpoint"] != checkpoint or model_bundle["device"] != device:
            raise ValueError("model_bundle checkpoint/device differ from the requested settings")
        revision = model_bundle["resolved_revision"]
    else:
        revision = _checkpoint_info(root, checkpoint)["resolved_revision"]
    provenance = {"render_key": render_record["cache_key"], "image_sha256": image_hashes,
                  "checkpoint": checkpoint, "resolved_revision": revision, "prompts": prompts,
                  "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "preprocess_version": PREPROCESS_VERSION, "versions": _versions(),
                  "device": device}
    inference_key = _hash_json(provenance)
    folder = root / "outputs/objects/inference" / inference_key
    cache_path = folder / "inference.json"
    if cache_path.is_file():
        record = json.loads(cache_path.read_text(encoding="utf-8"))
        if record.get("inference_key") != inference_key or record.get("cache_provenance") != provenance:
            raise ValueError(f"Inference cache metadata mismatch: {cache_path}")
        record.update(cache_hit=True, candidate_id=candidate_id,
                      cache_lookup_seconds=time.perf_counter() - started, cache_path=str(cache_path))
        return record
    model_load_started = time.perf_counter()
    bundle = model_bundle or load_clip(root, checkpoint=checkpoint, device=device)
    model_load_seconds = time.perf_counter() - model_load_started if model_bundle is None else 0.0
    torch, _, _ = _dependencies()
    from PIL import Image
    processor, model = bundle["processor"], bundle["model"]
    tick = time.perf_counter()
    images = []
    for path in paths:
        with Image.open(path) as source:
            if source.format != "PNG":
                raise ValueError(f"Expected an unannotated PNG point-cloud rendering: {path}")
            images.append(source.convert("RGB"))
    text_inputs = processor.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    max_length = int(model.config.text_config.max_position_embeddings)
    if text_inputs["input_ids"].shape[1] > max_length:
        raise ValueError(f"A class description exceeds CLIP's {max_length}-token text limit; shorten it (no prompt was silently truncated)")
    image_inputs = processor(images=images, return_tensors="pt")
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
    preprocessing_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    with torch.inference_mode():
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)
        image_features = torch.nn.functional.normalize(image_features.float(), p=2, dim=-1)
        text_features = torch.nn.functional.normalize(text_features.float(), p=2, dim=-1)
        scores = (image_features @ text_features.T).cpu().numpy()
    inference_seconds = time.perf_counter() - tick
    if scores.shape != (len(paths), len(prompts)) or not np.all(np.isfinite(scores)):
        raise RuntimeError("CLIP returned invalid similarity scores; no inference cache was written")
    mean_scores = scores.mean(axis=0)
    ranking = _rank(mean_scores, prompts)
    per_view = []
    for index, (view, row) in enumerate(zip(views, scores)):
        view_ranking = _rank(row, prompts)
        label = (view.get("label", view.get("name", f"view {index + 1}"))
                 if isinstance(view, dict) else str(view))
        per_view.append({"view_index": index, "view": _jsonable(view), "view_label": label,
                         "image_sha256": image_hashes[index], "top1": view_ranking[0]["class_description"],
                         "top3": view_ranking[:3], "ranking": view_ranking,
                         "cosine_by_class": row.tolist()})
    top1 = ranking[0]["class_description"]
    disagreement = sum(view["top1"] != top1 for view in per_view) / len(per_view)
    record = {"schema_version": 1, "candidate_id": candidate_id,
              "render_key": render_record["cache_key"], "inference_key": inference_key,
              "cache_hit": False, "cache_path": str(cache_path), "cache_provenance": provenance,
              "checkpoint": checkpoint, "resolved_revision": revision, "revision": revision,
              "prompts": prompts, "ranking": ranking, "top5": ranking[:5], "top1": top1,
              "aggregate_top1": top1, "top1_cosine": ranking[0]["cosine_similarity"],
              "per_view": per_view, "per_view_top1": [view["top1"] for view in per_view],
              "top1_disagreement_fraction": disagreement, "disagreement_fraction": disagreement,
              "score_matrix": scores.tolist(), "aggregate_scores": mean_scores.tolist(),
              "aggregation": "mean of L2-normalized image/text cosine scores over views",
              "interpretation": "similarities, not calibrated probabilities; view disagreement is not an accuracy measure",
              "hardware": bundle["hardware"], "versions": bundle["versions"],
              "preprocessing": _jsonable(bundle["preprocessing"]),
              "render_settings": _jsonable(render_record.get("settings", {})),
              "render_manifest_path": str(render_record.get("manifest_path", "")),
              "runtimes": {"model_load_seconds_this_call": model_load_seconds,
                           "initial_model_load_seconds": bundle["initial_load_runtime_seconds"],
                           "image_read_and_preprocess_seconds": preprocessing_seconds,
                           "inference_seconds": inference_seconds,
                           "call_seconds_excluding_cache_write": time.perf_counter() - started},
              "created_utc": datetime.now(timezone.utc).isoformat()}
    folder.mkdir(parents=True, exist_ok=True)
    with cache_path.open("x", encoding="utf-8") as stream:
        json.dump(_jsonable(record), stream, indent=2, allow_nan=False)
    return _jsonable(record)


def show_predictions(record: dict) -> None:
    """Display aggregate top five, per-view top three, and disagreement evidence."""
    from ground_experiment_views import show_table

    show_table([{"candidate": record["candidate_id"], "checkpoint": record["checkpoint"],
                 "resolved_commit": record["resolved_revision"], "cache_hit": record["cache_hit"],
                 "aggregate_top1": record["top1"],
                 "view_top1_disagreement_fraction": record["top1_disagreement_fraction"],
                 "score_meaning": "Cosine similarity; not a calibrated probability"}])
    show_table(record["top5"])
    show_table([{"view": view["view_label"], "top1": view["top1"],
                 "agrees_with_aggregate": view["top1"] == record["top1"],
                 "top3": view["top3"]} for view in record["per_view"]])
