from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

Array = np.ndarray


def read_grayscale_image(path: str | Path) -> Array:
    with Image.open(path) as handle:
        image = np.asarray(handle)
    if image.ndim != 2:
        raise ValueError("Only grayscale images are supported.")
    return image


def save_grayscale_image(path: str | Path, image: Array) -> None:
    output = np.asarray(np.clip(image, 0, 255), dtype=np.uint8)
    Image.fromarray(output, mode="L").save(path)


def ternary_entropyf(p_plus: Array, p_minus: Array) -> float:
    p_zero = 1 - p_plus - p_minus
    probabilities = np.concatenate((p_zero.ravel(), p_plus.ravel(), p_minus.ravel()))
    entropy = np.zeros_like(probabilities, dtype=np.float64)
    valid = (probabilities > 0) & (probabilities < 1)
    entropy[valid] = -probabilities[valid] * np.log2(probabilities[valid])
    eps = 2.2204e-16
    entropy[probabilities < eps] = 0
    entropy[probabilities > 1 - eps] = 0
    return float(np.sum(entropy))


def calc_lambda(rho_plus: Array, rho_minus: Array, message_length: float, n: int) -> float:
    upper_lambda = 1e3
    upper_message = float(message_length + 1)
    iterations = 0

    while upper_message > message_length:
        upper_lambda *= 2
        p_plus, p_minus = change_probabilities(rho_plus, rho_minus, upper_lambda)
        upper_message = ternary_entropyf(p_plus, p_minus)
        iterations += 1
        if iterations > 10:
            return upper_lambda

    lower_lambda = 0.0
    lower_message = float(n)
    alpha = float(message_length) / n
    current_lambda = 0.0
    iterations = 0

    while float(lower_message - upper_message) / n > alpha / 1000.0 and iterations < 300:
        current_lambda = lower_lambda + (upper_lambda - lower_lambda) / 2
        p_plus, p_minus = change_probabilities(rho_plus, rho_minus, current_lambda)
        middle_message = ternary_entropyf(p_plus, p_minus)
        if middle_message < message_length:
            upper_lambda = current_lambda
            upper_message = middle_message
        else:
            lower_lambda = current_lambda
            lower_message = middle_message
        iterations += 1

    return current_lambda


def change_probabilities(rho_plus: Array, rho_minus: Array, lamb: float) -> tuple[Array, Array]:
    exp_plus = np.exp(-lamb * rho_plus)
    exp_minus = np.exp(-lamb * rho_minus)
    denominator = 1 + exp_plus + exp_minus
    return exp_plus / denominator, exp_minus / denominator


def embedding_simulator(x: Array, rho_plus: Array, rho_minus: Array, message_length: float) -> Array:
    n = x.shape[0] * x.shape[1]
    lamb = calc_lambda(rho_plus, rho_minus, message_length, n)
    p_plus, p_minus = change_probabilities(rho_plus, rho_minus, lamb)

    y = x.copy()
    random_change = np.random.rand(*y.shape)
    y[random_change < p_plus] += 1
    mask = (random_change >= p_plus) & (random_change < p_plus + p_minus)
    y[mask] -= 1
    return y


def apply_wet_cost(rho: Array, cover: Array, wet_cost: float) -> tuple[Array, Array]:
    rho_plus = rho.copy()
    rho_minus = rho.copy()

    rho_plus[np.isnan(rho_plus)] = wet_cost
    rho_plus[rho_plus > wet_cost] = wet_cost
    rho_plus[cover == 255] = wet_cost

    rho_minus[np.isnan(rho_minus)] = wet_cost
    rho_minus[rho_minus > wet_cost] = wet_cost
    rho_minus[cover == 0] = wet_cost

    return rho_plus, rho_minus
