#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import scipy.signal

from .common import embedding_simulator, read_grayscale_image, save_grayscale_image


DEFAULT_P = -1.0
WET_COST = 10**10


def wow_cost(cover: np.ndarray, p: float = DEFAULT_P) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hpdf = np.array([
        -0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837,
         0.0158291053, -0.2840155430, -0.0004724846, 0.1287474266,
         0.0173693010, -0.0440882539, -0.0139810279, 0.0087460940,
         0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768,
    ], dtype=np.float64)
    signs = np.array([1 if index % 2 == 0 else -1 for index in range(len(hpdf))], dtype=np.float64)
    lpdf = hpdf[::-1] * signs

    filters = [
        np.outer(lpdf, hpdf),
        np.outer(hpdf, lpdf),
        np.outer(hpdf, hpdf),
    ]

    cover = np.asarray(cover, dtype=np.float64)
    height, width = cover.shape
    pad_size = max(max(kernel.shape) for kernel in filters)
    cover_padded = np.pad(cover, ((pad_size, pad_size), (pad_size, pad_size)), mode="symmetric")

    xis: list[np.ndarray] = []
    for kernel in filters:
        residual = scipy.signal.convolve2d(cover_padded, kernel, mode="same")
        suitability = scipy.signal.convolve2d(np.abs(residual), np.rot90(np.abs(kernel), 2), mode="same")

        if kernel.shape[0] % 2 == 0:
            suitability = np.roll(suitability, 1, axis=0)
        if kernel.shape[1] % 2 == 0:
            suitability = np.roll(suitability, 1, axis=1)

        row_margin = (suitability.shape[0] - height) // 2
        col_margin = (suitability.shape[1] - width) // 2
        suitability = suitability[row_margin:row_margin + height, col_margin:col_margin + width]
        xis.append(suitability)

    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (xis[0] ** p + xis[1] ** p + xis[2] ** p) ** (-1.0 / p)

    rho[rho > WET_COST] = WET_COST
    rho[~np.isfinite(rho)] = WET_COST

    rho_p1 = rho.copy()
    rho_m1 = rho.copy()
    rho_p1[cover == 255] = WET_COST
    rho_m1[cover == 0] = WET_COST
    return rho, rho_p1, rho_m1


def wow(cover_path: str, stego_path: str, payload: float, p: float = DEFAULT_P) -> None:
    cover = read_grayscale_image(cover_path)
    _, rho_p1, rho_m1 = wow_cost(cover, p=p)
    stego = embedding_simulator(cover, rho_p1, rho_m1, payload * cover.size)
    save_grayscale_image(stego_path, stego)