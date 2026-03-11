#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import scipy.signal

from steganography.common import apply_wet_cost, embedding_simulator, read_grayscale_image, save_grayscale_image
from steganography.hill import cost_fn as hill_cost
from steganography.suniward import cost_fn as suniward_cost
from steganography.wow import wow_cost


WET_COST = 10**10
EPS = 1e-6


def _normalize_cost(cost: np.ndarray) -> np.ndarray:
    finite = np.isfinite(cost) & (cost < WET_COST)
    if not np.any(finite):
        return np.full_like(cost, WET_COST, dtype=np.float64)
    scale = float(np.median(cost[finite]))
    if scale <= 0:
        scale = 1.0
    normalized = np.asarray(cost, dtype=np.float64) / scale
    normalized[~np.isfinite(normalized)] = WET_COST
    return normalized


def _texture_gate(cover: np.ndarray) -> np.ndarray:
    cover = np.asarray(cover, dtype=np.float64)
    lap = scipy.signal.convolve2d(
        cover,
        np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        mode="same",
        boundary="symm",
    )
    local_var = scipy.signal.convolve2d(
        (cover - cover.mean()) ** 2,
        np.ones((7, 7), dtype=np.float64) / 49.0,
        mode="same",
        boundary="symm",
    )
    score = np.abs(lap) + 0.35 * np.sqrt(np.maximum(local_var, 0.0))
    lo = float(np.percentile(score, 15))
    hi = float(np.percentile(score, 92))
    spread = max(hi - lo, EPS)
    normalized = np.clip((score - lo) / spread, 0.0, 1.0)
    return 1.8 - 1.15 * normalized


def cost_fn(cover: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wow_rho, _, _ = wow_cost(cover)
    hill_plus, hill_minus = hill_cost(cover)
    suniward_plus, suniward_minus = suniward_cost(cover)

    hill_rho = np.minimum(hill_plus, hill_minus)
    suniward_rho = np.minimum(suniward_plus, suniward_minus)

    stacked = np.stack(
        (
            _normalize_cost(wow_rho),
            _normalize_cost(hill_rho),
            _normalize_cost(suniward_rho),
        ),
        axis=0,
    )

    strict = np.max(stacked, axis=0)
    support = np.exp(np.mean(np.log(stacked + EPS), axis=0))
    combined = 0.8 * strict + 0.2 * support
    combined *= _texture_gate(cover)
    combined = scipy.signal.convolve2d(
        combined,
        np.ones((5, 5), dtype=np.float64) / 25.0,
        mode="same",
        boundary="symm",
    )
    return apply_wet_cost(combined, cover, wet_cost=WET_COST)


def masquerade(cover_path: str, stego_path: str, payload: float) -> None:
    cover = read_grayscale_image(cover_path)
    rho_plus, rho_minus = cost_fn(cover)
    stego = embedding_simulator(cover, rho_plus, rho_minus, round(payload * cover.size))
    save_grayscale_image(stego_path, stego)
