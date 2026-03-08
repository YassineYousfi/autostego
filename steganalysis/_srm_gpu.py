from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from ._srm import _load_image, _q_suffix, parse_feaname, post_processing

Array = np.ndarray
Tensor = torch.Tensor


def srm(
    image: str | Path | Array,
    device: str | torch.device | None = None,
) -> dict[str, Array]:
    x = _load_image(image)
    tensor = _to_image_tensor(x, device=device)

    result = post_processing(all1st(tensor, 1), "f1", 1)
    result = post_processing(all1st(tensor, 2), "f1", 2, result)

    for q in (1, 1.5, 2):
        result = post_processing(all2nd(tensor, q * 2), "f2", q, result)
    for q in (1, 1.5, 2):
        result = post_processing(all3rd(tensor, q * 3), "f3", q, result)
    for q in (1, 1.5, 2):
        result = post_processing(all3x3(tensor, q * 4), "f3x3", q, result)
    for q in (1, 1.5, 2):
        result = post_processing(all5x5(tensor, q * 12), "f5x5", q, result)

    return result


def benchmark(
    image: str | Path | Array,
    repeats: int = 5,
    warmup: int = 1,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    from time import perf_counter

    from ._srm import srm as cpu_srm

    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    x = _load_image(image)
    tensor = _to_image_tensor(x, device=device)
    device_name = str(tensor.device)

    cpu_result = cpu_srm(x)
    for _ in range(warmup):
        _ = srm(x, device=tensor.device)
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    gpu_times: list[float] = []
    last_gpu_result: dict[str, Array] | None = None
    for _ in range(repeats):
        start = perf_counter()
        last_gpu_result = srm(x, device=tensor.device)
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        gpu_times.append(perf_counter() - start)

    cpu_times: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        _ = cpu_srm(x)
        cpu_times.append(perf_counter() - start)

    assert last_gpu_result is not None
    compare(cpu_result, last_gpu_result)

    return {
        "device": device_name,
        "repeats": repeats,
        "warmup": warmup,
        "cpu_mean_seconds": float(np.mean(cpu_times)),
        "gpu_mean_seconds": float(np.mean(gpu_times)),
        "speedup": float(np.mean(cpu_times) / np.mean(gpu_times)),
        "feature_count": len(cpu_result),
    }


def benchmark_directory(
    image_dir: str | Path,
    *,
    limit: int = 1000,
    devices: Sequence[str | torch.device] | None = None,
    verify_count: int = 3,
) -> dict[str, Any]:
    image_root = Path(image_dir)
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    image_paths = sorted(image_root.glob("*.pgm"))[:limit]
    if not image_paths:
        raise FileNotFoundError(f"No .pgm images found in {image_root}")

    resolved_devices = _normalize_devices(devices)
    verify_paths = image_paths[: min(verify_count, len(image_paths))]
    verification = verify_images(verify_paths, device=resolved_devices[0])

    single_device = run_distributed_benchmark(image_paths, devices=[resolved_devices[0]])
    multi_device = run_distributed_benchmark(image_paths, devices=resolved_devices)

    return {
        "image_dir": str(image_root),
        "image_count": len(image_paths),
        "devices": resolved_devices,
        "verification": verification,
        "single_gpu": single_device,
        "multi_gpu": multi_device,
        "multi_gpu_speedup_vs_single": multi_device["wall_seconds"] / single_device["wall_seconds"] if False else single_device["wall_seconds"] / multi_device["wall_seconds"],
    }


def run_distributed_benchmark(
    image_paths: Sequence[str | Path],
    *,
    devices: Sequence[str | torch.device],
) -> dict[str, Any]:
    resolved_devices = _normalize_devices(devices)
    paths = [str(Path(path)) for path in image_paths]
    partitions = _partition_round_robin(paths, len(resolved_devices))
    jobs = [(resolved_devices[index], chunk) for index, chunk in enumerate(partitions) if chunk]
    if not jobs:
        raise ValueError("No work items were assigned to GPU workers.")

    context = get_context("spawn")
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=context) as executor:
        worker_results = list(executor.map(_benchmark_worker, jobs))
    wall_seconds = perf_counter() - started

    image_count = sum(result["image_count"] for result in worker_results)
    return {
        "devices": resolved_devices,
        "worker_count": len(jobs),
        "image_count": image_count,
        "wall_seconds": wall_seconds,
        "images_per_second": image_count / wall_seconds,
        "workers": worker_results,
    }


def verify_images(
    image_paths: Iterable[str | Path],
    *,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    from ._srm import srm as cpu_srm

    checked: list[str] = []
    for image_path in image_paths:
        path = Path(image_path)
        compare(cpu_srm(path), srm(path, device=device))
        checked.append(str(path))
    return {"verified_images": checked, "verified_count": len(checked)}


def _benchmark_worker(job: tuple[str, list[str]]) -> dict[str, Any]:
    device, paths = job
    torch.cuda.set_device(torch.device(device))
    started = perf_counter()
    first_image_started = perf_counter()
    _ = srm(paths[0], device=device)
    torch.cuda.synchronize(torch.device(device))
    first_image_seconds = perf_counter() - first_image_started

    for image_path in paths[1:]:
        _ = srm(image_path, device=device)
    torch.cuda.synchronize(torch.device(device))
    elapsed = perf_counter() - started

    return {
        "device": device,
        "image_count": len(paths),
        "wall_seconds": elapsed,
        "images_per_second": len(paths) / elapsed,
        "first_image_seconds": first_image_seconds,
    }


def _normalize_devices(devices: Sequence[str | torch.device] | None) -> list[str]:
    if devices is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for GPU benchmarking.")
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    resolved = [str(torch.device(device)) for device in devices]
    if not resolved:
        raise ValueError("At least one device must be provided.")
    return resolved


def _partition_round_robin(paths: Sequence[str], partition_count: int) -> list[list[str]]:
    partitions = [[] for _ in range(partition_count)]
    for index, path in enumerate(paths):
        partitions[index % partition_count].append(path)
    return partitions


def compare(cpu_result: dict[str, Array], gpu_result: dict[str, Array]) -> None:
    if set(cpu_result) != set(gpu_result):
        raise AssertionError("GPU SRM feature keys do not match CPU SRM feature keys.")
    for key in sorted(cpu_result):
        np.testing.assert_allclose(cpu_result[key], gpu_result[key], rtol=0.0, atol=0.0, err_msg=key)


def _to_image_tensor(image: Array, device: str | torch.device | None = None) -> Tensor:
    target = torch.device(device) if device is not None else _default_device()
    return torch.as_tensor(image, dtype=torch.float64, device=target)


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def all1st(x: Tensor, q: float) -> dict[str, Array]:
    t = 2
    order = 4

    center = x[1:-1, 1:-1]
    rq = quant(x[1:-1, 2:] - center, q, t)
    lq = quant(x[1:-1, :-2] - center, q, t)
    uq = quant(x[:-2, 1:-1] - center, q, t)
    dq = quant(x[2:, 1:-1] - center, q, t)

    ruq = quant(x[:-2, 2:] - center, q, t)
    rdq = quant(x[2:, 2:] - center, q, t)
    luq = quant(x[:-2, :-2] - center, q, t)
    ldq = quant(x[2:, :-2] - center, q, t)

    rlq_min = torch.minimum(rq, lq)
    udq_min = torch.minimum(uq, dq)
    rlq_max = torch.maximum(rq, lq)
    udq_max = torch.maximum(uq, dq)

    result: dict[str, Array] = {}
    result["min22h"] = _flatten_feature(cooc(rlq_min, order, "hor", t) + cooc(udq_min, order, "ver", t))
    result["max22h"] = _flatten_feature(cooc(rlq_max, order, "hor", t) + cooc(udq_max, order, "ver", t))

    uq_min = torch.minimum(torch.minimum(lq, uq), rq)
    rq_min = torch.minimum(torch.minimum(uq, rq), dq)
    dq_min = torch.minimum(torch.minimum(rq, dq), lq)
    lq_min = torch.minimum(torch.minimum(dq, lq), uq)
    uq_max = torch.maximum(torch.maximum(lq, uq), rq)
    rq_max = torch.maximum(torch.maximum(uq, rq), dq)
    dq_max = torch.maximum(torch.maximum(rq, dq), lq)
    lq_max = torch.maximum(torch.maximum(dq, lq), uq)

    result["min34h"] = _flatten_feature(
        cooc(torch.vstack((uq_min, dq_min)), order, "hor", t)
        + cooc(torch.hstack((lq_min, rq_min)), order, "ver", t)
    )
    result["max34h"] = _flatten_feature(
        cooc(torch.vstack((uq_max, dq_max)), order, "hor", t)
        + cooc(torch.hstack((rq_max, lq_max)), order, "ver", t)
    )

    result["spam14h"] = _flatten_feature(cooc(rq, order, "hor", t) + cooc(uq, order, "ver", t))
    result["spam14v"] = _flatten_feature(cooc(rq, order, "ver", t) + cooc(uq, order, "hor", t))
    result["min22v"] = _flatten_feature(cooc(rlq_min, order, "ver", t) + cooc(udq_min, order, "hor", t))
    result["max22v"] = _flatten_feature(cooc(rlq_max, order, "ver", t) + cooc(udq_max, order, "hor", t))

    ruq_min = torch.minimum(rq, uq)
    rdq_min = torch.minimum(rq, dq)
    luq_min = torch.minimum(lq, uq)
    ldq_min = torch.minimum(lq, dq)
    ruq_max = torch.maximum(rq, uq)
    rdq_max = torch.maximum(rq, dq)
    luq_max = torch.maximum(lq, uq)
    ldq_max = torch.maximum(lq, dq)

    result["min24"] = _flatten_feature(
        cooc(torch.vstack((ruq_min, rdq_min, luq_min, ldq_min)), order, "hor", t)
        + cooc(torch.hstack((ruq_min, rdq_min, luq_min, ldq_min)), order, "ver", t)
    )
    result["max24"] = _flatten_feature(
        cooc(torch.vstack((ruq_max, rdq_max, luq_max, ldq_max)), order, "hor", t)
        + cooc(torch.hstack((ruq_max, rdq_max, luq_max, ldq_max)), order, "ver", t)
    )

    result["min34v"] = _flatten_feature(
        cooc(torch.hstack((uq_min, dq_min)), order, "ver", t)
        + cooc(torch.vstack((rq_min, lq_min)), order, "hor", t)
    )
    result["max34v"] = _flatten_feature(
        cooc(torch.hstack((uq_max, dq_max)), order, "ver", t)
        + cooc(torch.vstack((rq_max, lq_max)), order, "hor", t)
    )

    r_min = torch.minimum(rlq_min, udq_min)
    r_max = torch.maximum(rlq_max, udq_max)
    result["min41"] = _flatten_feature(cooc(r_min, order, "hor", t) + cooc(r_min, order, "ver", t))
    result["max41"] = _flatten_feature(cooc(r_max, order, "hor", t) + cooc(r_max, order, "ver", t))

    ruq_min = torch.minimum(ruq_min, ruq)
    rdq_min = torch.minimum(rdq_min, rdq)
    luq_min = torch.minimum(luq_min, luq)
    ldq_min = torch.minimum(ldq_min, ldq)
    ruq_max = torch.maximum(ruq_max, ruq)
    rdq_max = torch.maximum(rdq_max, rdq)
    luq_max = torch.maximum(luq_max, luq)
    ldq_max = torch.maximum(ldq_max, ldq)

    result["min34"] = _flatten_feature(
        cooc(torch.vstack((ruq_min, rdq_min, luq_min, ldq_min)), order, "hor", t)
        + cooc(torch.hstack((ruq_min, rdq_min, luq_min, ldq_min)), order, "ver", t)
    )
    result["max34"] = _flatten_feature(
        cooc(torch.vstack((ruq_max, rdq_max, luq_max, ldq_max)), order, "hor", t)
        + cooc(torch.hstack((ruq_max, rdq_max, luq_max, ldq_max)), order, "ver", t)
    )

    ruq_min2 = torch.minimum(ruq_min, luq)
    rdq_min2 = torch.minimum(rdq_min, ruq)
    ldq_min2 = torch.minimum(ldq_min, rdq)
    luq_min2 = torch.minimum(luq_min, ldq)
    ruq_min3 = torch.minimum(ruq_min, rdq)
    rdq_min3 = torch.minimum(rdq_min, ldq)
    ldq_min3 = torch.minimum(ldq_min, luq)
    luq_min3 = torch.minimum(luq_min, ruq)
    result["min48h"] = _flatten_feature(
        cooc(torch.vstack((ruq_min2, ldq_min2, rdq_min3, luq_min3)), order, "hor", t)
        + cooc(torch.hstack((rdq_min2, luq_min2, ruq_min3, ldq_min3)), order, "ver", t)
    )
    result["min48v"] = _flatten_feature(
        cooc(torch.vstack((rdq_min2, luq_min2, ruq_min3, ldq_min3)), order, "hor", t)
        + cooc(torch.hstack((ruq_min2, ldq_min2, rdq_min3, luq_min3)), order, "ver", t)
    )

    ruq_max2 = torch.maximum(ruq_max, luq)
    rdq_max2 = torch.maximum(rdq_max, ruq)
    ldq_max2 = torch.maximum(ldq_max, rdq)
    luq_max2 = torch.maximum(luq_max, ldq)
    ruq_max3 = torch.maximum(ruq_max, rdq)
    rdq_max3 = torch.maximum(rdq_max, ldq)
    ldq_max3 = torch.maximum(ldq_max, luq)
    luq_max3 = torch.maximum(luq_max, ruq)
    result["max48h"] = _flatten_feature(
        cooc(torch.vstack((ruq_max2, ldq_max2, rdq_max3, luq_max3)), order, "hor", t)
        + cooc(torch.hstack((rdq_max2, luq_max2, ruq_max3, ldq_max3)), order, "ver", t)
    )
    result["max48v"] = _flatten_feature(
        cooc(torch.vstack((rdq_max2, luq_max2, ruq_max3, ldq_max3)), order, "hor", t)
        + cooc(torch.hstack((ruq_max2, ldq_max2, rdq_max3, luq_max3)), order, "ver", t)
    )

    ruq_min4 = torch.minimum(ruq_min2, rdq)
    rdq_min4 = torch.minimum(rdq_min2, ldq)
    ldq_min4 = torch.minimum(ldq_min2, luq)
    luq_min4 = torch.minimum(luq_min2, ruq)
    ruq_min5 = torch.minimum(ruq_min3, luq)
    rdq_min5 = torch.minimum(rdq_min3, ruq)
    ldq_min5 = torch.minimum(ldq_min3, rdq)
    luq_min5 = torch.minimum(luq_min3, ldq)
    result["min54"] = _flatten_feature(
        cooc(torch.vstack((ruq_min4, ldq_min4, rdq_min5, luq_min5)), order, "hor", t)
        + cooc(torch.hstack((rdq_min4, luq_min4, ruq_min5, ldq_min5)), order, "ver", t)
    )

    ruq_max4 = torch.maximum(ruq_max2, rdq)
    rdq_max4 = torch.maximum(rdq_max2, ldq)
    ldq_max4 = torch.maximum(ldq_max2, luq)
    luq_max4 = torch.maximum(luq_max2, ruq)
    ruq_max5 = torch.maximum(ruq_max3, luq)
    rdq_max5 = torch.maximum(rdq_max3, ruq)
    ldq_max5 = torch.maximum(ldq_max3, rdq)
    luq_max5 = torch.maximum(luq_max3, ldq)
    result["max54"] = _flatten_feature(
        cooc(torch.vstack((ruq_max4, ldq_max4, rdq_max5, luq_max5)), order, "hor", t)
        + cooc(torch.hstack((rdq_max4, luq_max4, ruq_max5, ldq_max5)), order, "ver", t)
    )

    return result


def all2nd(x: Tensor, q: float) -> dict[str, Array]:
    t = 2
    order = 4

    dh = residual(x, 2, "hor")
    dv = residual(x, 2, "ver")
    dd = residual(x, 2, "diag")
    dm = residual(x, 2, "mdiag")
    yh = quant(dh, q, t)
    yv = quant(dv, q, t)
    yd = quant(dd, q, t)
    ym = quant(dm, q, t)

    result: dict[str, Array] = {}
    result["spam12h"] = _flatten_feature(cooc(yh, order, "hor", t) + cooc(yv, order, "ver", t))
    result["spam12v"] = _flatten_feature(cooc(yh, order, "ver", t) + cooc(yv, order, "hor", t))

    dmin = torch.minimum(yh, yv)
    dmax = torch.maximum(yh, yv)
    result["min21"] = _flatten_feature(cooc(dmin, order, "hor", t) + cooc(dmin, order, "ver", t))
    result["max21"] = _flatten_feature(cooc(dmax, order, "hor", t) + cooc(dmax, order, "ver", t))

    dmin2 = torch.minimum(dmin, torch.minimum(yd, ym))
    dmax2 = torch.maximum(dmax, torch.maximum(yd, ym))
    result["min41"] = _flatten_feature(cooc(dmin2, order, "hor", t) + cooc(dmin2, order, "ver", t))
    result["max41"] = _flatten_feature(cooc(dmax2, order, "hor", t) + cooc(dmax2, order, "ver", t))

    ruq_min = torch.minimum(dmin, ym)
    rdq_min = torch.minimum(dmin, yd)
    ruq_max = torch.maximum(dmax, ym)
    rdq_max = torch.maximum(dmax, yd)
    result["min32"] = _flatten_feature(
        cooc(torch.vstack((ruq_min, rdq_min)), order, "hor", t)
        + cooc(torch.hstack((ruq_min, rdq_min)), order, "ver", t)
    )
    result["max32"] = _flatten_feature(
        cooc(torch.vstack((ruq_max, rdq_max)), order, "hor", t)
        + cooc(torch.hstack((ruq_max, rdq_max)), order, "ver", t)
    )

    ruq_min2 = torch.minimum(ym, yh)
    rdq_min2 = torch.minimum(yd, yh)
    ruq_min3 = torch.minimum(ym, yv)
    luq_min3 = torch.minimum(yd, yv)
    result["min24h"] = _flatten_feature(
        cooc(torch.vstack((ruq_min2, rdq_min2)), order, "hor", t)
        + cooc(torch.hstack((ruq_min3, luq_min3)), order, "ver", t)
    )
    result["min24v"] = _flatten_feature(
        cooc(torch.hstack((ruq_min2, rdq_min2)), order, "ver", t)
        + cooc(torch.vstack((ruq_min3, luq_min3)), order, "hor", t)
    )

    ruq_max2 = torch.maximum(ym, yh)
    rdq_max2 = torch.maximum(yd, yh)
    ruq_max3 = torch.maximum(ym, yv)
    luq_max3 = torch.maximum(yd, yv)
    result["max24h"] = _flatten_feature(
        cooc(torch.vstack((ruq_max2, rdq_max2)), order, "hor", t)
        + cooc(torch.hstack((ruq_max3, luq_max3)), order, "ver", t)
    )
    result["max24v"] = _flatten_feature(
        cooc(torch.hstack((ruq_max2, rdq_max2)), order, "ver", t)
        + cooc(torch.vstack((ruq_max3, luq_max3)), order, "hor", t)
    )

    return result


def all3rd(x: Tensor, q: float) -> dict[str, Array]:
    t = 2
    order = 4

    center = x[2:-2, 2:-2]
    rq = quant(-x[2:-2, 4:] + 3 * x[2:-2, 3:-1] - 3 * center + x[2:-2, 1:-3], q, t)
    lq = quant(-x[2:-2, :-4] + 3 * x[2:-2, 1:-3] - 3 * center + x[2:-2, 3:-1], q, t)
    uq = quant(-x[:-4, 2:-2] + 3 * x[1:-3, 2:-2] - 3 * center + x[3:-1, 2:-2], q, t)
    dq = quant(-x[4:, 2:-2] + 3 * x[3:-1, 2:-2] - 3 * center + x[1:-3, 2:-2], q, t)

    ruq = quant(-x[:-4, 4:] + 3 * x[1:-3, 3:-1] - 3 * center + x[3:-1, 1:-3], q, t)
    luq = quant(-x[:-4, :-4] + 3 * x[1:-3, 1:-3] - 3 * center + x[3:-1, 3:-1], q, t)
    rdq = quant(-x[4:, 4:] + 3 * x[3:-1, 3:-1] - 3 * center + x[1:-3, 1:-3], q, t)
    ldq = quant(-x[4:, :-4] + 3 * x[3:-1, 1:-3] - 3 * center + x[1:-3, 3:-1], q, t)

    rlq_min = torch.minimum(rq, lq)
    udq_min = torch.minimum(uq, dq)
    rlq_max = torch.maximum(rq, lq)
    udq_max = torch.maximum(uq, dq)

    result: dict[str, Array] = {}
    result["min22h"] = _flatten_feature(cooc(rlq_min, order, "hor", t) + cooc(udq_min, order, "ver", t))
    result["max22h"] = _flatten_feature(cooc(rlq_max, order, "hor", t) + cooc(udq_max, order, "ver", t))

    uq_min = torch.minimum(rlq_min, uq)
    rq_min = torch.minimum(udq_min, rq)
    dq_min = torch.minimum(rlq_min, dq)
    lq_min = torch.minimum(udq_min, lq)
    uq_max = torch.maximum(rlq_max, uq)
    rq_max = torch.maximum(udq_max, rq)
    dq_max = torch.maximum(rlq_max, dq)
    lq_max = torch.maximum(udq_max, lq)

    result["min34h"] = _flatten_feature(
        cooc(torch.vstack((uq_min, dq_min)), order, "hor", t)
        + cooc(torch.hstack((rq_min, lq_min)), order, "ver", t)
    )
    result["max34h"] = _flatten_feature(
        cooc(torch.vstack((uq_max, dq_max)), order, "hor", t)
        + cooc(torch.hstack((rq_max, lq_max)), order, "ver", t)
    )

    result["spam14h"] = _flatten_feature(cooc(rq, order, "hor", t) + cooc(uq, order, "ver", t))
    result["spam14v"] = _flatten_feature(cooc(rq, order, "ver", t) + cooc(uq, order, "hor", t))
    result["min22v"] = _flatten_feature(cooc(rlq_min, order, "ver", t) + cooc(udq_min, order, "hor", t))
    result["max22v"] = _flatten_feature(cooc(rlq_max, order, "ver", t) + cooc(udq_max, order, "hor", t))

    ruq_min = torch.minimum(rq, uq)
    rdq_min = torch.minimum(rq, dq)
    luq_min = torch.minimum(lq, uq)
    ldq_min = torch.minimum(lq, dq)
    ruq_max = torch.maximum(rq, uq)
    rdq_max = torch.maximum(rq, dq)
    luq_max = torch.maximum(lq, uq)
    ldq_max = torch.maximum(lq, dq)

    result["min24"] = _flatten_feature(
        cooc(torch.vstack((ruq_min, rdq_min, luq_min, ldq_min)), order, "hor", t)
        + cooc(torch.hstack((ruq_min, rdq_min, luq_min, ldq_min)), order, "ver", t)
    )
    result["max24"] = _flatten_feature(
        cooc(torch.vstack((ruq_max, rdq_max, luq_max, ldq_max)), order, "hor", t)
        + cooc(torch.hstack((ruq_max, rdq_max, luq_max, ldq_max)), order, "ver", t)
    )

    result["min34v"] = _flatten_feature(
        cooc(torch.hstack((uq_min, dq_min)), order, "ver", t)
        + cooc(torch.vstack((rq_min, lq_min)), order, "hor", t)
    )
    result["max34v"] = _flatten_feature(
        cooc(torch.hstack((uq_max, dq_max)), order, "ver", t)
        + cooc(torch.vstack((rq_max, lq_max)), order, "hor", t)
    )

    r_min = torch.minimum(ruq_min, ldq_min)
    r_max = torch.maximum(ruq_max, ldq_max)
    result["min41"] = _flatten_feature(cooc(r_min, order, "hor", t) + cooc(r_min, order, "ver", t))
    result["max41"] = _flatten_feature(cooc(r_max, order, "hor", t) + cooc(r_max, order, "ver", t))

    ruq_min2 = torch.minimum(ruq_min, ruq)
    rdq_min2 = torch.minimum(rdq_min, rdq)
    luq_min2 = torch.minimum(luq_min, luq)
    ldq_min2 = torch.minimum(ldq_min, ldq)
    ruq_max2 = torch.maximum(ruq_max, ruq)
    rdq_max2 = torch.maximum(rdq_max, rdq)
    luq_max2 = torch.maximum(luq_max, luq)
    ldq_max2 = torch.maximum(ldq_max, ldq)

    result["min34"] = _flatten_feature(
        cooc(torch.vstack((ruq_min2, rdq_min2, luq_min2, ldq_min2)), order, "hor", t)
        + cooc(torch.hstack((ruq_min2, rdq_min2, luq_min2, ldq_min2)), order, "ver", t)
    )
    result["max34"] = _flatten_feature(
        cooc(torch.vstack((ruq_max2, rdq_max2, luq_max2, ldq_max2)), order, "hor", t)
        + cooc(torch.hstack((ruq_max2, rdq_max2, luq_max2, ldq_max2)), order, "ver", t)
    )

    ruq_min3 = torch.minimum(ruq_min2, luq)
    rdq_min3 = torch.minimum(rdq_min2, ruq)
    ldq_min3 = torch.minimum(ldq_min2, rdq)
    luq_min3 = torch.minimum(luq_min2, ldq)
    ruq_min4 = torch.minimum(ruq_min2, rdq)
    rdq_min4 = torch.minimum(rdq_min2, ldq)
    ldq_min4 = torch.minimum(ldq_min2, luq)
    luq_min4 = torch.minimum(luq_min2, ruq)
    result["min48h"] = _flatten_feature(
        cooc(torch.vstack((ruq_min3, ldq_min3, rdq_min4, luq_min4)), order, "hor", t)
        + cooc(torch.hstack((rdq_min3, luq_min3, ruq_min4, ldq_min4)), order, "ver", t)
    )
    result["min48v"] = _flatten_feature(
        cooc(torch.hstack((ruq_min3, ldq_min3, rdq_min4, luq_min4)), order, "ver", t)
        + cooc(torch.vstack((rdq_min3, luq_min3, ruq_min4, ldq_min4)), order, "hor", t)
    )

    ruq_max3 = torch.maximum(ruq_max2, luq)
    rdq_max3 = torch.maximum(rdq_max2, ruq)
    ldq_max3 = torch.maximum(ldq_max2, rdq)
    luq_max3 = torch.maximum(luq_max2, ldq)
    ruq_max4 = torch.maximum(ruq_max2, rdq)
    rdq_max4 = torch.maximum(rdq_max2, ldq)
    ldq_max4 = torch.maximum(ldq_max2, luq)
    luq_max4 = torch.maximum(luq_max2, ruq)
    result["max48h"] = _flatten_feature(
        cooc(torch.vstack((ruq_max3, ldq_max3, rdq_max4, luq_max4)), order, "hor", t)
        + cooc(torch.hstack((rdq_max3, luq_max3, ruq_max4, ldq_max4)), order, "ver", t)
    )
    result["max48v"] = _flatten_feature(
        cooc(torch.hstack((ruq_max3, ldq_max3, rdq_max4, luq_max4)), order, "ver", t)
        + cooc(torch.vstack((rdq_max3, luq_max3, ruq_max4, ldq_max4)), order, "hor", t)
    )

    ruq_min5 = torch.minimum(ruq_min3, rdq)
    rdq_min5 = torch.minimum(rdq_min3, ldq)
    ldq_min5 = torch.minimum(ldq_min3, luq)
    luq_min5 = torch.minimum(luq_min3, ruq)
    ruq_max5 = torch.maximum(ruq_max3, rdq)
    rdq_max5 = torch.maximum(rdq_max3, ldq)
    ldq_max5 = torch.maximum(ldq_max3, luq)
    luq_max5 = torch.maximum(luq_max3, ruq)
    result["min54"] = _flatten_feature(
        cooc(torch.vstack((ruq_min5, ldq_min5, rdq_min5, luq_min5)), order, "hor", t)
        + cooc(torch.hstack((rdq_min5, luq_min5, ruq_min5, ldq_min5)), order, "ver", t)
    )
    result["max54"] = _flatten_feature(
        cooc(torch.vstack((ruq_max5, ldq_max5, rdq_max5, luq_max5)), order, "hor", t)
        + cooc(torch.hstack((rdq_max5, luq_max5, ruq_max5, ldq_max5)), order, "ver", t)
    )

    return result


def all3x3(x: Tensor, q: float) -> dict[str, Array]:
    t = 2
    order = 4

    y = quant(residual(x, 2, "KB"), q, t)
    result: dict[str, Array] = {}
    result["spam11"] = _flatten_feature(cooc(y, order, "hor", t) + cooc(y, order, "ver", t))

    edge_h = residual(x, 2, "edge-h")
    midpoint = edge_h.shape[1] // 2
    du, db = edge_h[:, :midpoint], edge_h[:, midpoint:]
    edge_v = residual(x, 2, "edge-v")
    midpoint = edge_v.shape[1] // 2
    dl, dr = edge_v[:, :midpoint], edge_v[:, midpoint:]
    yu, yb, yl, yr = (quant(part, q, t) for part in (du, db, dl, dr))

    result["spam14v"] = _flatten_feature(
        cooc(torch.hstack((yu, yb)), order, "ver", t) + cooc(torch.vstack((yl, yr)), order, "hor", t)
    )
    result["spam14h"] = _flatten_feature(
        cooc(torch.vstack((yu, yb)), order, "hor", t) + cooc(torch.hstack((yl, yr)), order, "ver", t)
    )

    dmin1 = torch.minimum(yu, yl)
    dmin2 = torch.minimum(yb, yr)
    dmin3 = torch.minimum(yu, yr)
    dmin4 = torch.minimum(yb, yl)
    result["min24"] = _flatten_feature(
        cooc(torch.hstack((dmin1, dmin2, dmin3, dmin4)), order, "ver", t)
        + cooc(torch.vstack((dmin1, dmin2, dmin3, dmin4)), order, "hor", t)
    )

    dmax1 = torch.maximum(yu, yl)
    dmax2 = torch.maximum(yb, yr)
    dmax3 = torch.maximum(yu, yr)
    dmax4 = torch.maximum(yb, yl)
    result["max24"] = _flatten_feature(
        cooc(torch.hstack((dmax1, dmax2, dmax3, dmax4)), order, "ver", t)
        + cooc(torch.vstack((dmax1, dmax2, dmax3, dmax4)), order, "hor", t)
    )

    ueq_min = torch.minimum(yu, yb)
    req_min = torch.minimum(yr, yl)
    result["min22h"] = _flatten_feature(cooc(ueq_min, order, "hor", t) + cooc(req_min, order, "ver", t))
    result["min22v"] = _flatten_feature(cooc(ueq_min, order, "ver", t) + cooc(req_min, order, "hor", t))

    ueq_max = torch.maximum(yu, yb)
    req_max = torch.maximum(yr, yl)
    result["max22h"] = _flatten_feature(cooc(ueq_max, order, "hor", t) + cooc(req_max, order, "ver", t))
    result["max22v"] = _flatten_feature(cooc(ueq_max, order, "ver", t) + cooc(req_max, order, "hor", t))

    dmin5 = torch.minimum(dmin1, dmin2)
    dmax5 = torch.maximum(dmax1, dmax2)
    result["min41"] = _flatten_feature(cooc(dmin5, order, "ver", t) + cooc(dmin5, order, "hor", t))
    result["max41"] = _flatten_feature(cooc(dmax5, order, "ver", t) + cooc(dmax5, order, "hor", t))

    return result


def all5x5(x: Tensor, q: float) -> dict[str, Array]:
    t = 2
    order = 4

    m, n = x.shape
    i = slice(2, m - 2)
    j = slice(2, n - 2)
    center = x[i, j]

    y = quant(residual(x, 3, "KV"), q, t)
    result: dict[str, Array] = {}
    result["spam11"] = _flatten_feature(cooc(y, order, "hor", t) + cooc(y, order, "ver", t))

    du = (
        8 * x[i, 1:-3] + 8 * x[1:-3, j] + 8 * x[i, 3:-1]
        - 6 * x[1:-3, 1:-3] - 6 * x[1:-3, 3:-1]
        - 2 * x[i, :-4] - 2 * x[i, 4:] - 2 * x[:-4, j]
        + 2 * x[1:-3, :-4] + 2 * x[:-4, 1:-3] + 2 * x[:-4, 3:-1] + 2 * x[1:-3, 4:]
        - x[:-4, :-4] - x[:-4, 4:] - 12 * center
    )
    dr = (
        8 * x[1:-3, j] + 8 * x[i, 3:-1] + 8 * x[3:-1, j]
        - 6 * x[1:-3, 3:-1] - 6 * x[3:-1, 3:-1]
        - 2 * x[:-4, j] - 2 * x[4:, j] - 2 * x[i, 4:]
        + 2 * x[:-4, 3:-1] + 2 * x[1:-3, 4:] + 2 * x[3:-1, 4:] + 2 * x[4:, 3:-1]
        - x[:-4, 4:] - x[4:, 4:] - 12 * center
    )
    db = (
        8 * x[i, 3:-1] + 8 * x[3:-1, j] + 8 * x[i, 1:-3]
        - 6 * x[3:-1, 3:-1] - 6 * x[3:-1, 1:-3]
        - 2 * x[i, :-4] - 2 * x[i, 4:] - 2 * x[4:, j]
        + 2 * x[3:-1, 4:] + 2 * x[4:, 3:-1] + 2 * x[4:, 1:-3] + 2 * x[3:-1, :-4]
        - x[4:, 4:] - x[4:, :-4] - 12 * center
    )
    dl = (
        8 * x[3:-1, j] + 8 * x[i, 1:-3] + 8 * x[1:-3, j]
        - 6 * x[3:-1, 1:-3] - 6 * x[1:-3, 1:-3]
        - 2 * x[:-4, j] - 2 * x[4:, j] - 2 * x[i, :-4]
        + 2 * x[4:, 1:-3] + 2 * x[3:-1, :-4] + 2 * x[1:-3, :-4] + 2 * x[:-4, 1:-3]
        - x[4:, :-4] - x[:-4, :-4] - 12 * center
    )
    yu, yb, yl, yr = (quant(part, q, t) for part in (du, db, dl, dr))

    result["spam14v"] = _flatten_feature(
        cooc(torch.hstack((yu, yb)), order, "ver", t) + cooc(torch.vstack((yl, yr)), order, "hor", t)
    )
    result["spam14h"] = _flatten_feature(
        cooc(torch.vstack((yu, yb)), order, "hor", t) + cooc(torch.hstack((yl, yr)), order, "ver", t)
    )

    dmin1 = torch.minimum(yu, yl)
    dmin2 = torch.minimum(yb, yr)
    dmin3 = torch.minimum(yu, yr)
    dmin4 = torch.minimum(yb, yl)
    result["min24"] = _flatten_feature(
        cooc(torch.hstack((dmin1, dmin2, dmin3, dmin4)), order, "ver", t)
        + cooc(torch.vstack((dmin1, dmin2, dmin3, dmin4)), order, "hor", t)
    )

    dmax1 = torch.maximum(yu, yl)
    dmax2 = torch.maximum(yb, yr)
    dmax3 = torch.maximum(yu, yr)
    dmax4 = torch.maximum(yb, yl)
    result["max24"] = _flatten_feature(
        cooc(torch.hstack((dmax1, dmax2, dmax3, dmax4)), order, "ver", t)
        + cooc(torch.vstack((dmax1, dmax2, dmax3, dmax4)), order, "hor", t)
    )

    ueq_min = torch.minimum(yu, yb)
    req_min = torch.minimum(yr, yl)
    result["min22h"] = _flatten_feature(cooc(ueq_min, order, "hor", t) + cooc(req_min, order, "ver", t))
    result["min22v"] = _flatten_feature(cooc(ueq_min, order, "ver", t) + cooc(req_min, order, "hor", t))

    ueq_max = torch.maximum(yu, yb)
    req_max = torch.maximum(yr, yl)
    result["max22h"] = _flatten_feature(cooc(ueq_max, order, "hor", t) + cooc(req_max, order, "ver", t))
    result["max22v"] = _flatten_feature(cooc(ueq_max, order, "ver", t) + cooc(req_max, order, "hor", t))

    dmin5 = torch.minimum(dmin1, dmin2)
    dmax5 = torch.maximum(dmax1, dmax2)
    result["min41"] = _flatten_feature(cooc(dmin5, order, "ver", t) + cooc(dmin5, order, "hor", t))
    result["max41"] = _flatten_feature(cooc(dmax5, order, "ver", t) + cooc(dmax5, order, "hor", t))

    return result


def cooc(d: Tensor, order: int, kind: str, t: int) -> Tensor:
    bins = 2 * t + 1
    arrays = _cooc_arrays(d, order, kind)
    linear = torch.zeros_like(_ravel_fortran(arrays[0]), dtype=torch.int64)
    for part in arrays:
        linear = linear * bins + (_ravel_fortran(part).to(torch.int64) + t)
    counts = torch.bincount(linear, minlength=bins**order).to(dtype=torch.float64)
    counts = counts.reshape((bins,) * order)
    total = counts.sum()
    if total.item() > 0:
        counts = counts / total
    return counts


def _cooc_arrays(d: Tensor, order: int, kind: str) -> tuple[Tensor, ...]:
    if order == 1:
        return (d,)

    if order == 2:
        if kind == "hor":
            return d[:, :-1], d[:, 1:]
        if kind == "ver":
            return d[:-1, :], d[1:, :]
        if kind == "diag":
            return d[:-1, :-1], d[1:, 1:]
        if kind == "mdiag":
            return d[:-1, 1:], d[1:, :-1]

    if order == 3:
        if kind == "hor":
            return d[:, :-2], d[:, 1:-1], d[:, 2:]
        if kind == "ver":
            return d[:-2, :], d[1:-1, :], d[2:, :]
        if kind == "diag":
            return d[:-2, :-2], d[1:-1, 1:-1], d[2:, 2:]
        if kind == "mdiag":
            return d[:-2, 2:], d[1:-1, 1:-1], d[2:, :-2]

    if order == 4:
        if kind == "hor":
            return d[:, :-3], d[:, 1:-2], d[:, 2:-1], d[:, 3:]
        if kind == "ver":
            return d[:-3, :], d[1:-2, :], d[2:-1, :], d[3:, :]
        if kind == "diag":
            return d[:-3, :-3], d[1:-2, 1:-2], d[2:-1, 2:-1], d[3:, 3:]
        if kind == "mdiag":
            return d[3:, :-3], d[2:-1, 1:-2], d[1:-2, 2:-1], d[:-3, 3:]

    if order == 5:
        if kind == "hor":
            return d[:, :-4], d[:, 1:-3], d[:, 2:-2], d[:, 3:-1], d[:, 4:]
        if kind == "ver":
            return d[:-4, :], d[1:-3, :], d[2:-2, :], d[3:-1, :], d[4:, :]

    raise ValueError(f"Unsupported co-occurrence mode: order={order}, kind={kind}")


def quant(x: Tensor, q: float | Array, t: int) -> Tensor:
    if np.isscalar(q):
        step = Fraction(str(float(q))).limit_denominator()
        if step <= 0:
            raise ValueError("Quantization step must be positive.")
        numerator = step.numerator
        denominator = step.denominator
        abs_x = torch.round(torch.abs(x)).to(torch.int64)
        rounded = (2 * abs_x * denominator + numerator) // (2 * numerator)
        signed = torch.sign(x).to(torch.int64) * rounded
        return torch.clamp(signed, -t, t).to(torch.int16)

    q_array = np.rint(np.asarray(q)).astype(np.int16)
    if np.any(np.diff(q_array) <= 0):
        raise ValueError("Quantization vector must be strictly increasing.")
    if np.min(q_array) < 0:
        raise ValueError("Quantization vector must be non-negative.")

    t = int(q_array[-1])
    values = np.zeros(2 * t + 1, dtype=np.int16)
    y = torch.clamp(torch.round(x), -t, t).to(torch.int64) + t

    if q_array[0] == 0:
        values[t] = 0
        z = 1
        index = t + 1
        for current, previous in zip(q_array[1:], q_array[:-1]):
            span = int(current - previous)
            values[index:index + span] = z
            index += span
            z += 1
        values[:t] = -values[:t:-1]
    else:
        start = t - int(q_array[0])
        stop = t + int(q_array[0]) + 1
        values[start:stop] = 0
        z = 1
        index = t + 1 + int(q_array[0])
        for current, previous in zip(q_array[1:], q_array[:-1]):
            span = int(current - previous)
            values[index:index + span] = z
            index += span
            z += 1
        values[: t - int(q_array[0])] = -values[: t + 1 : -1]

    lookup = torch.as_tensor(values, device=x.device, dtype=torch.int16)
    return lookup[y]


def residual(x: Tensor, order: int, kind: str) -> Tensor:
    m, n = x.shape
    border = int(np.ceil(order / 2))
    i = slice(border, m - border)
    j = slice(border, n - border)

    if kind == "hor":
        if order == 1:
            return -x[i, j] + x[i, border + 1 : n - border + 1]
        if order == 2:
            return x[i, border - 1 : n - border - 1] - 2 * x[i, j] + x[i, border + 1 : n - border + 1]
        if order == 3:
            return x[i, border - 1 : n - border - 1] - 3 * x[i, j] + 3 * x[i, border + 1 : n - border + 1] - x[i, border + 2 : n - border + 2]
        if order == 4:
            return -x[i, border - 2 : n - border - 2] + 4 * x[i, border - 1 : n - border - 1] - 6 * x[i, j] + 4 * x[i, border + 1 : n - border + 1] - x[i, border + 2 : n - border + 2]
        if order == 5:
            return -x[i, border - 2 : n - border - 2] + 5 * x[i, border - 1 : n - border - 1] - 10 * x[i, j] + 10 * x[i, border + 1 : n - border + 1] - 5 * x[i, border + 2 : n - border + 2] + x[i, border + 3 : n - border + 3]
        if order == 6:
            return x[i, border - 3 : n - border - 3] - 6 * x[i, border - 2 : n - border - 2] + 15 * x[i, border - 1 : n - border - 1] - 20 * x[i, j] + 15 * x[i, border + 1 : n - border + 1] - 6 * x[i, border + 2 : n - border + 2] + x[i, border + 3 : n - border + 3]

    if kind == "ver":
        if order == 1:
            return -x[i, j] + x[border + 1 : m - border + 1, j]
        if order == 2:
            return x[border - 1 : m - border - 1, j] - 2 * x[i, j] + x[border + 1 : m - border + 1, j]
        if order == 3:
            return x[border - 1 : m - border - 1, j] - 3 * x[i, j] + 3 * x[border + 1 : m - border + 1, j] - x[border + 2 : m - border + 2, j]
        if order == 4:
            return -x[border - 2 : m - border - 2, j] + 4 * x[border - 1 : m - border - 1, j] - 6 * x[i, j] + 4 * x[border + 1 : m - border + 1, j] - x[border + 2 : m - border + 2, j]
        if order == 5:
            return -x[border - 2 : m - border - 2, j] + 5 * x[border - 1 : m - border - 1, j] - 10 * x[i, j] + 10 * x[border + 1 : m - border + 1, j] - 5 * x[border + 2 : m - border + 2, j] + x[border + 3 : m - border + 3, j]
        if order == 6:
            return x[border - 3 : m - border - 3, j] - 6 * x[border - 2 : m - border - 2, j] + 15 * x[border - 1 : m - border - 1, j] - 20 * x[i, j] + 15 * x[border + 1 : m - border + 1, j] - 6 * x[border + 2 : m - border + 2, j] + x[border + 3 : m - border + 3, j]

    if kind == "diag":
        if order == 1:
            return -x[i, j] + x[border + 1 : m - border + 1, border + 1 : n - border + 1]
        if order == 2:
            return x[border - 1 : m - border - 1, border - 1 : n - border - 1] - 2 * x[i, j] + x[border + 1 : m - border + 1, border + 1 : n - border + 1]
        if order == 3:
            return x[border - 1 : m - border - 1, border - 1 : n - border - 1] - 3 * x[i, j] + 3 * x[border + 1 : m - border + 1, border + 1 : n - border + 1] - x[border + 2 : m - border + 2, border + 2 : n - border + 2]
        if order == 4:
            return -x[border - 2 : m - border - 2, border - 2 : n - border - 2] + 4 * x[border - 1 : m - border - 1, border - 1 : n - border - 1] - 6 * x[i, j] + 4 * x[border + 1 : m - border + 1, border + 1 : n - border + 1] - x[border + 2 : m - border + 2, border + 2 : n - border + 2]
        if order == 5:
            return -x[border - 2 : m - border - 2, border - 2 : n - border - 2] + 5 * x[border - 1 : m - border - 1, border - 1 : n - border - 1] - 10 * x[i, j] + 10 * x[border + 1 : m - border + 1, border + 1 : n - border + 1] - 5 * x[border + 2 : m - border + 2, border + 2 : n - border + 2] + x[border + 3 : m - border + 3, border + 3 : n - border + 3]
        if order == 6:
            return x[border - 3 : m - border - 3, border - 3 : n - border - 3] - 6 * x[border - 2 : m - border - 2, border - 2 : n - border - 2] + 15 * x[border - 1 : m - border - 1, border - 1 : n - border - 1] - 20 * x[i, j] + 15 * x[border + 1 : m - border + 1, border + 1 : n - border + 1] - 6 * x[border + 2 : m - border + 2, border + 2 : n - border + 2] + x[border + 3 : m - border + 3, border + 3 : n - border + 3]

    if kind == "mdiag":
        if order == 1:
            return -x[i, j] + x[border - 1 : m - border - 1, border + 1 : n - border + 1]
        if order == 2:
            return x[border - 1 : m - border - 1, border + 1 : n - border + 1] - 2 * x[i, j] + x[border + 1 : m - border + 1, border - 1 : n - border - 1]
        if order == 3:
            return x[border - 1 : m - border - 1, border + 1 : n - border + 1] - 3 * x[i, j] + 3 * x[border + 1 : m - border + 1, border - 1 : n - border - 1] - x[border + 2 : m - border + 2, border - 2 : n - border - 2]
        if order == 4:
            return -x[border - 2 : m - border - 2, border + 2 : n - border + 2] + 4 * x[border - 1 : m - border - 1, border + 1 : n - border + 1] - 6 * x[i, j] + 4 * x[border + 1 : m - border + 1, border - 1 : n - border - 1] - x[border + 2 : m - border + 2, border - 2 : n - border - 2]
        if order == 5:
            return -x[border - 2 : m - border - 2, border + 2 : n - border + 2] + 5 * x[border - 1 : m - border - 1, border + 1 : n - border + 1] - 10 * x[i, j] + 10 * x[border + 1 : m - border + 1, border - 1 : n - border - 1] - 5 * x[border + 2 : m - border + 2, border - 2 : n - border - 2] + x[border + 3 : m - border + 3, border - 3 : n - border - 3]
        if order == 6:
            return x[border - 3 : m - border - 3, border + 3 : n - border + 3] - 6 * x[border - 2 : m - border - 2, border + 2 : n - border + 2] + 15 * x[border - 1 : m - border - 1, border + 1 : n - border + 1] - 20 * x[i, j] + 15 * x[border + 1 : m - border + 1, border - 1 : n - border - 1] - 6 * x[border + 2 : m - border + 2, border - 2 : n - border - 2] + x[border + 3 : m - border + 3, border - 3 : n - border - 3]

    if kind == "KB":
        return (
            -x[border - 1 : m - border - 1, border - 1 : n - border - 1]
            + 2 * x[border - 1 : m - border - 1, j]
            - x[border - 1 : m - border - 1, border + 1 : n - border + 1]
            + 2 * x[i, border - 1 : n - border - 1]
            - 4 * x[i, j]
            + 2 * x[i, border + 1 : n - border + 1]
            - x[border + 1 : m - border + 1, border - 1 : n - border - 1]
            + 2 * x[border + 1 : m - border + 1, j]
            - x[border + 1 : m - border + 1, border + 1 : n - border + 1]
        )

    if kind == "edge-h":
        du = 2 * x[border - 1 : m - border - 1, j] + 2 * x[i, border - 1 : n - border - 1] + 2 * x[i, border + 1 : n - border + 1] - x[border - 1 : m - border - 1, border - 1 : n - border - 1] - x[border - 1 : m - border - 1, border + 1 : n - border + 1] - 4 * x[i, j]
        db = 2 * x[border + 1 : m - border + 1, j] + 2 * x[i, border - 1 : n - border - 1] + 2 * x[i, border + 1 : n - border + 1] - x[border + 1 : m - border + 1, border - 1 : n - border - 1] - x[border + 1 : m - border + 1, border + 1 : n - border + 1] - 4 * x[i, j]
        return torch.hstack((du, db))

    if kind == "edge-v":
        dl = 2 * x[i, border - 1 : n - border - 1] + 2 * x[border - 1 : m - border - 1, j] + 2 * x[border + 1 : m - border + 1, j] - x[border - 1 : m - border - 1, border - 1 : n - border - 1] - x[border + 1 : m - border + 1, border - 1 : n - border - 1] - 4 * x[i, j]
        dr = 2 * x[i, border + 1 : n - border + 1] + 2 * x[border - 1 : m - border - 1, j] + 2 * x[border + 1 : m - border + 1, j] - x[border - 1 : m - border - 1, border + 1 : n - border + 1] - x[border + 1 : m - border + 1, border + 1 : n - border + 1] - 4 * x[i, j]
        return torch.hstack((dl, dr))

    if kind == "edge-m":
        dlu = 2 * x[i, border - 1 : n - border - 1] + 2 * x[border - 1 : m - border - 1, j] - x[border - 1 : m - border - 1, border - 1 : n - border - 1] - x[border + 1 : m - border + 1, border - 1 : n - border - 1] - x[border - 1 : m - border - 1, border + 1 : n - border + 1] - x[i, j]
        drb = 2 * x[i, border + 1 : n - border + 1] + 2 * x[border + 1 : m - border + 1, j] - x[border + 1 : m - border + 1, border + 1 : n - border + 1] - x[border + 1 : m - border + 1, border - 1 : n - border - 1] - x[border - 1 : m - border - 1, border + 1 : n - border + 1] - x[i, j]
        return torch.hstack((dlu, drb))

    if kind == "edge-d":
        dru = 2 * x[border - 1 : m - border - 1, j] + 2 * x[i, border + 1 : n - border + 1] - x[border - 1 : m - border - 1, border + 1 : n - border + 1] - x[border - 1 : m - border - 1, border - 1 : n - border - 1] - x[border + 1 : m - border + 1, border + 1 : n - border + 1] - x[i, j]
        dlb = 2 * x[i, border - 1 : n - border - 1] + 2 * x[border + 1 : m - border + 1, j] - x[border + 1 : m - border + 1, border - 1 : n - border - 1] - x[border + 1 : m - border + 1, border + 1 : n - border + 1] - x[border - 1 : m - border - 1, border - 1 : n - border - 1] - x[i, j]
        return torch.hstack((dru, dlb))

    if kind == "KV":
        result = 8 * x[border - 1 : m - border - 1, j] + 8 * x[border + 1 : m - border + 1, j] + 8 * x[i, border - 1 : n - border - 1] + 8 * x[i, border + 1 : n - border + 1]
        result = result - 6 * x[border - 1 : m - border - 1, border + 1 : n - border + 1] - 6 * x[border - 1 : m - border - 1, border - 1 : n - border - 1] - 6 * x[border + 1 : m - border + 1, border - 1 : n - border - 1] - 6 * x[border + 1 : m - border + 1, border + 1 : n - border + 1]
        result = result - 2 * x[border - 2 : m - border - 2, j] - 2 * x[border + 2 : m - border + 2, j] - 2 * x[i, border + 2 : n - border + 2] - 2 * x[i, border - 2 : n - border - 2]
        result = result + 2 * x[border - 1 : m - border - 1, border - 2 : n - border - 2] + 2 * x[border - 2 : m - border - 2, border - 1 : n - border - 1] + 2 * x[border - 2 : m - border - 2, border + 1 : n - border + 1] + 2 * x[border - 1 : m - border - 1, border + 2 : n - border + 2] + 2 * x[border + 1 : m - border + 1, border + 2 : n - border + 2] + 2 * x[border + 2 : m - border + 2, border + 1 : n - border + 1] + 2 * x[border + 2 : m - border + 2, border - 1 : n - border - 1] + 2 * x[border + 1 : m - border + 1, border - 2 : n - border - 2]
        result = result - x[border - 2 : m - border - 2, border - 2 : n - border - 2] - x[border - 2 : m - border - 2, border + 2 : n - border + 2] - x[border + 2 : m - border + 2, border - 2 : n - border - 2] - x[border + 2 : m - border + 2, border + 2 : n - border + 2] - 12 * x[i, j]
        return result

    raise ValueError(f"Unsupported residual kind: {kind}")


def _flatten_feature(value: Tensor) -> Array:
    return value.detach().cpu().numpy().reshape(-1, order="F")


def _ravel_fortran(value: Tensor) -> Tensor:
    return value.transpose(0, 1).contiguous().reshape(-1)
