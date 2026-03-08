#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import scipy.signal

from .common import apply_wet_cost, embedding_simulator, read_grayscale_image, save_grayscale_image


def cost_fn(cover: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    HF1 = np.array([
        [-1, 2, -1],
        [ 2,-4,  2],
        [-1, 2, -1]
    ])
    H2 = np.ones((3, 3), dtype=np.float64) / 3**2
    HW = np.ones((15, 15), dtype=np.float64) / 15**2

    R1 = scipy.signal.convolve2d(cover, HF1, mode='same', boundary='symm')
    W1 = scipy.signal.convolve2d(np.abs(R1), H2, mode='same', boundary='symm')
    rho = 1.0 / (W1 + 10**(-10))
    rho = scipy.signal.convolve2d(rho, HW, mode='same', boundary='symm')
    return apply_wet_cost(rho, cover, wet_cost=10**10)

def hill(cover_path: str, stego_path: str, payload: float) -> None:
    cover = read_grayscale_image(cover_path)
    rho_p1, rho_m1 = cost_fn(cover)
    stego = embedding_simulator(cover, rho_p1, rho_m1, payload * cover.size)
    save_grayscale_image(stego_path, stego)







