#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import scipy.signal

from steganography.common import apply_wet_cost, embedding_simulator, read_grayscale_image, save_grayscale_image


np.seterr(divide='ignore')



def cost_fn(cover: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k, l = cover.shape

    hpdf = np.array([
        -0.0544158422,  0.3128715909, -0.6756307363,  0.5853546837,
         0.0158291053, -0.2840155430, -0.0004724846,  0.1287474266,
         0.0173693010, -0.0440882539, -0.0139810279,  0.0087460940,
         0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768
    ])

    sign = np.array([-1 if i%2 else 1 for i in range(len(hpdf))])
    lpdf = hpdf[::-1] * sign

    filters = [
        np.outer(lpdf.T, hpdf),
        np.outer(hpdf.T, lpdf),
        np.outer(hpdf.T, hpdf),
    ]

    sgm = 1
    pad_size = 16

    rho = np.zeros((k, l))
    for kernel in filters:
        cover_padded = np.pad(cover, (pad_size, pad_size), 'symmetric').astype('float32')

        residual = scipy.signal.convolve2d(cover_padded, kernel, mode="same")
        suitability = scipy.signal.convolve2d(1.0 / (np.abs(residual) + sgm), np.rot90(np.abs(kernel), 2), 'same')

        if kernel.shape[0] % 2 == 0:
            suitability = np.roll(suitability, 1, axis=0)

        if kernel.shape[1] % 2 == 0:
            suitability = np.roll(suitability, 1, axis=1)

        suitability = suitability[
            (suitability.shape[0] - k) // 2:-(suitability.shape[0] - k) // 2,
            (suitability.shape[1] - l) // 2:-(suitability.shape[1] - l) // 2,
        ]
        rho += suitability

    return apply_wet_cost(rho, cover, wet_cost=10**13)


def suniward(cover_path: str, stego_path: str, payload: float) -> None:
    cover = read_grayscale_image(cover_path)
    rho_p1, rho_m1 = cost_fn(cover)
    stego = embedding_simulator(cover, rho_p1, rho_m1, round(payload * cover.size))
    save_grayscale_image(stego_path, stego)


