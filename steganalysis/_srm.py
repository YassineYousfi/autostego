from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

Array = np.ndarray


def srm(image: str | Path | Array) -> dict[str, Array]:
    x = _load_image(image)

    result = post_processing(all1st(x, 1), "f1", 1)
    result = post_processing(all1st(x, 2), "f1", 2, result)

    for q in (1, 1.5, 2):
        result = post_processing(all2nd(x, q * 2), "f2", q, result)
    for q in (1, 1.5, 2):
        result = post_processing(all3rd(x, q * 3), "f3", q, result)
    for q in (1, 1.5, 2):
        result = post_processing(all3x3(x, q * 4), "f3x3", q, result)
    for q in (1, 1.5, 2):
        result = post_processing(all5x5(x, q * 12), "f5x5", q, result)

    return result


def _load_image(image: str | Path | Array) -> Array:
    if isinstance(image, np.ndarray):
        return image.astype(np.float64, copy=False)
    with Image.open(image) as handle:
        return np.asarray(handle, dtype=np.float64)


def post_processing(
    data: dict[str, Array],
    prefix: str,
    q: float,
    result: dict[str, Array] | None = None,
) -> dict[str, Array]:
    if result is None:
        result = {}

    for name, value in data.items():
        var_name = f"{prefix}_{name}_q{_q_suffix(q)}"
        result[var_name] = _flatten_feature(value).astype(np.float32)

    for name in list(result):
        if name.startswith("s"):
            continue
        family, feature_name, quant_name = parse_feaname(name)
        if not family:
            continue

        if feature_name.startswith("min") or feature_name.startswith("max"):
            out = f"s{family[1:]}_minmax{feature_name[3:]}_{quant_name}"
            if out in result:
                continue
            fmin = result[name.replace("max", "min", 1)]
            fmax = result[name.replace("min", "max", 1)]
            merged = symfea(np.concatenate((fmin, fmax))[:, None], 2, 4, "mnmx")
            result[out] = merged.ravel().astype(np.float32)
        elif feature_name.startswith("spam"):
            out = f"s{family[1:]}_{feature_name}_{quant_name}"
            if out in result:
                continue
            merged = symm1(result[name][:, None], 2, 4)
            result[out] = merged.ravel().astype(np.float32)

    for name in [key for key in result if key.startswith("f")]:
        del result[name]

    for name in list(result):
        family, feature_name, quant_name = parse_feaname(name)
        if not family or not feature_name.startswith("spam"):
            continue

        if feature_name.endswith("v") or (feature_name == "spam11" and family == "s5x5"):
            continue
        if feature_name.endswith("h"):
            out = f"{family}_{feature_name}v_{quant_name}"
            if out in result:
                continue
            other = name.replace("h_", "v_", 1)
            result[out] = np.concatenate((result[name], result[other])).astype(np.float32)
            del result[name]
            del result[other]
        elif feature_name == "spam11":
            out = f"s35_{feature_name}_{quant_name}"
            if out in result:
                continue
            name_3x3 = name.replace("5x5", "3x3", 1)
            name_5x5 = name.replace("3x3", "5x5", 1)
            if name_3x3 not in result or name_5x5 not in result:
                continue
            result[out] = np.concatenate((result[name_3x3], result[name_5x5])).astype(np.float32)
            del result[name_3x3]
            del result[name_5x5]

    return result


def parse_feaname(name: str) -> tuple[str, str, str]:
    parts = name.split("_")
    if len(parts) != 3:
        return "", "", ""
    return parts[0], parts[1], parts[2]


def all1st(x: Array, q: float) -> dict[str, Array]:
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

    rlq_min = np.minimum(rq, lq)
    udq_min = np.minimum(uq, dq)
    rlq_max = np.maximum(rq, lq)
    udq_max = np.maximum(uq, dq)

    result: dict[str, Array] = {}
    result["min22h"] = _flatten_feature(cooc(rlq_min, order, "hor", t) + cooc(udq_min, order, "ver", t))
    result["max22h"] = _flatten_feature(cooc(rlq_max, order, "hor", t) + cooc(udq_max, order, "ver", t))

    uq_min = np.minimum(np.minimum(lq, uq), rq)
    rq_min = np.minimum(np.minimum(uq, rq), dq)
    dq_min = np.minimum(np.minimum(rq, dq), lq)
    lq_min = np.minimum(np.minimum(dq, lq), uq)
    uq_max = np.maximum(np.maximum(lq, uq), rq)
    rq_max = np.maximum(np.maximum(uq, rq), dq)
    dq_max = np.maximum(np.maximum(rq, dq), lq)
    lq_max = np.maximum(np.maximum(dq, lq), uq)

    result["min34h"] = _flatten_feature(
        cooc(np.vstack((uq_min, dq_min)), order, "hor", t)
        + cooc(np.hstack((lq_min, rq_min)), order, "ver", t)
    )
    result["max34h"] = _flatten_feature(
        cooc(np.vstack((uq_max, dq_max)), order, "hor", t)
        + cooc(np.hstack((rq_max, lq_max)), order, "ver", t)
    )

    result["spam14h"] = _flatten_feature(cooc(rq, order, "hor", t) + cooc(uq, order, "ver", t))
    result["spam14v"] = _flatten_feature(cooc(rq, order, "ver", t) + cooc(uq, order, "hor", t))
    result["min22v"] = _flatten_feature(cooc(rlq_min, order, "ver", t) + cooc(udq_min, order, "hor", t))
    result["max22v"] = _flatten_feature(cooc(rlq_max, order, "ver", t) + cooc(udq_max, order, "hor", t))

    ruq_min = np.minimum(rq, uq)
    rdq_min = np.minimum(rq, dq)
    luq_min = np.minimum(lq, uq)
    ldq_min = np.minimum(lq, dq)
    ruq_max = np.maximum(rq, uq)
    rdq_max = np.maximum(rq, dq)
    luq_max = np.maximum(lq, uq)
    ldq_max = np.maximum(lq, dq)

    result["min24"] = _flatten_feature(
        cooc(np.vstack((ruq_min, rdq_min, luq_min, ldq_min)), order, "hor", t)
        + cooc(np.hstack((ruq_min, rdq_min, luq_min, ldq_min)), order, "ver", t)
    )
    result["max24"] = _flatten_feature(
        cooc(np.vstack((ruq_max, rdq_max, luq_max, ldq_max)), order, "hor", t)
        + cooc(np.hstack((ruq_max, rdq_max, luq_max, ldq_max)), order, "ver", t)
    )

    result["min34v"] = _flatten_feature(
        cooc(np.hstack((uq_min, dq_min)), order, "ver", t)
        + cooc(np.vstack((rq_min, lq_min)), order, "hor", t)
    )
    result["max34v"] = _flatten_feature(
        cooc(np.hstack((uq_max, dq_max)), order, "ver", t)
        + cooc(np.vstack((rq_max, lq_max)), order, "hor", t)
    )

    r_min = np.minimum(rlq_min, udq_min)
    r_max = np.maximum(rlq_max, udq_max)
    result["min41"] = _flatten_feature(cooc(r_min, order, "hor", t) + cooc(r_min, order, "ver", t))
    result["max41"] = _flatten_feature(cooc(r_max, order, "hor", t) + cooc(r_max, order, "ver", t))

    ruq_min = np.minimum(ruq_min, ruq)
    rdq_min = np.minimum(rdq_min, rdq)
    luq_min = np.minimum(luq_min, luq)
    ldq_min = np.minimum(ldq_min, ldq)
    ruq_max = np.maximum(ruq_max, ruq)
    rdq_max = np.maximum(rdq_max, rdq)
    luq_max = np.maximum(luq_max, luq)
    ldq_max = np.maximum(ldq_max, ldq)

    result["min34"] = _flatten_feature(
        cooc(np.vstack((ruq_min, rdq_min, luq_min, ldq_min)), order, "hor", t)
        + cooc(np.hstack((ruq_min, rdq_min, luq_min, ldq_min)), order, "ver", t)
    )
    result["max34"] = _flatten_feature(
        cooc(np.vstack((ruq_max, rdq_max, luq_max, ldq_max)), order, "hor", t)
        + cooc(np.hstack((ruq_max, rdq_max, luq_max, ldq_max)), order, "ver", t)
    )

    ruq_min2 = np.minimum(ruq_min, luq)
    rdq_min2 = np.minimum(rdq_min, ruq)
    ldq_min2 = np.minimum(ldq_min, rdq)
    luq_min2 = np.minimum(luq_min, ldq)
    ruq_min3 = np.minimum(ruq_min, rdq)
    rdq_min3 = np.minimum(rdq_min, ldq)
    ldq_min3 = np.minimum(ldq_min, luq)
    luq_min3 = np.minimum(luq_min, ruq)
    result["min48h"] = _flatten_feature(
        cooc(np.vstack((ruq_min2, ldq_min2, rdq_min3, luq_min3)), order, "hor", t)
        + cooc(np.hstack((rdq_min2, luq_min2, ruq_min3, ldq_min3)), order, "ver", t)
    )
    result["min48v"] = _flatten_feature(
        cooc(np.vstack((rdq_min2, luq_min2, ruq_min3, ldq_min3)), order, "hor", t)
        + cooc(np.hstack((ruq_min2, ldq_min2, rdq_min3, luq_min3)), order, "ver", t)
    )

    ruq_max2 = np.maximum(ruq_max, luq)
    rdq_max2 = np.maximum(rdq_max, ruq)
    ldq_max2 = np.maximum(ldq_max, rdq)
    luq_max2 = np.maximum(luq_max, ldq)
    ruq_max3 = np.maximum(ruq_max, rdq)
    rdq_max3 = np.maximum(rdq_max, ldq)
    ldq_max3 = np.maximum(ldq_max, luq)
    luq_max3 = np.maximum(luq_max, ruq)
    result["max48h"] = _flatten_feature(
        cooc(np.vstack((ruq_max2, ldq_max2, rdq_max3, luq_max3)), order, "hor", t)
        + cooc(np.hstack((rdq_max2, luq_max2, ruq_max3, ldq_max3)), order, "ver", t)
    )
    result["max48v"] = _flatten_feature(
        cooc(np.vstack((rdq_max2, luq_max2, ruq_max3, ldq_max3)), order, "hor", t)
        + cooc(np.hstack((ruq_max2, ldq_max2, rdq_max3, luq_max3)), order, "ver", t)
    )

    ruq_min4 = np.minimum(ruq_min2, rdq)
    rdq_min4 = np.minimum(rdq_min2, ldq)
    ldq_min4 = np.minimum(ldq_min2, luq)
    luq_min4 = np.minimum(luq_min2, ruq)
    ruq_min5 = np.minimum(ruq_min3, luq)
    rdq_min5 = np.minimum(rdq_min3, ruq)
    ldq_min5 = np.minimum(ldq_min3, rdq)
    luq_min5 = np.minimum(luq_min3, ldq)
    result["min54"] = _flatten_feature(
        cooc(np.vstack((ruq_min4, ldq_min4, rdq_min5, luq_min5)), order, "hor", t)
        + cooc(np.hstack((rdq_min4, luq_min4, ruq_min5, ldq_min5)), order, "ver", t)
    )

    ruq_max4 = np.maximum(ruq_max2, rdq)
    rdq_max4 = np.maximum(rdq_max2, ldq)
    ldq_max4 = np.maximum(ldq_max2, luq)
    luq_max4 = np.maximum(luq_max2, ruq)
    ruq_max5 = np.maximum(ruq_max3, luq)
    rdq_max5 = np.maximum(rdq_max3, ruq)
    ldq_max5 = np.maximum(ldq_max3, rdq)
    luq_max5 = np.maximum(luq_max3, ldq)
    result["max54"] = _flatten_feature(
        cooc(np.vstack((ruq_max4, ldq_max4, rdq_max5, luq_max5)), order, "hor", t)
        + cooc(np.hstack((rdq_max4, luq_max4, ruq_max5, ldq_max5)), order, "ver", t)
    )

    return result


def all2nd(x: Array, q: float) -> dict[str, Array]:
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

    dmin = np.minimum(yh, yv)
    dmax = np.maximum(yh, yv)
    result["min21"] = _flatten_feature(cooc(dmin, order, "hor", t) + cooc(dmin, order, "ver", t))
    result["max21"] = _flatten_feature(cooc(dmax, order, "hor", t) + cooc(dmax, order, "ver", t))

    dmin2 = np.minimum(dmin, np.minimum(yd, ym))
    dmax2 = np.maximum(dmax, np.maximum(yd, ym))
    result["min41"] = _flatten_feature(cooc(dmin2, order, "hor", t) + cooc(dmin2, order, "ver", t))
    result["max41"] = _flatten_feature(cooc(dmax2, order, "hor", t) + cooc(dmax2, order, "ver", t))

    ruq_min = np.minimum(dmin, ym)
    rdq_min = np.minimum(dmin, yd)
    ruq_max = np.maximum(dmax, ym)
    rdq_max = np.maximum(dmax, yd)
    result["min32"] = _flatten_feature(
        cooc(np.vstack((ruq_min, rdq_min)), order, "hor", t)
        + cooc(np.hstack((ruq_min, rdq_min)), order, "ver", t)
    )
    result["max32"] = _flatten_feature(
        cooc(np.vstack((ruq_max, rdq_max)), order, "hor", t)
        + cooc(np.hstack((ruq_max, rdq_max)), order, "ver", t)
    )

    ruq_min2 = np.minimum(ym, yh)
    rdq_min2 = np.minimum(yd, yh)
    ruq_min3 = np.minimum(ym, yv)
    luq_min3 = np.minimum(yd, yv)
    result["min24h"] = _flatten_feature(
        cooc(np.vstack((ruq_min2, rdq_min2)), order, "hor", t)
        + cooc(np.hstack((ruq_min3, luq_min3)), order, "ver", t)
    )
    result["min24v"] = _flatten_feature(
        cooc(np.hstack((ruq_min2, rdq_min2)), order, "ver", t)
        + cooc(np.vstack((ruq_min3, luq_min3)), order, "hor", t)
    )

    ruq_max2 = np.maximum(ym, yh)
    rdq_max2 = np.maximum(yd, yh)
    ruq_max3 = np.maximum(ym, yv)
    luq_max3 = np.maximum(yd, yv)
    result["max24h"] = _flatten_feature(
        cooc(np.vstack((ruq_max2, rdq_max2)), order, "hor", t)
        + cooc(np.hstack((ruq_max3, luq_max3)), order, "ver", t)
    )
    result["max24v"] = _flatten_feature(
        cooc(np.hstack((ruq_max2, rdq_max2)), order, "ver", t)
        + cooc(np.vstack((ruq_max3, luq_max3)), order, "hor", t)
    )

    return result


def all3rd(x: Array, q: float) -> dict[str, Array]:
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

    rlq_min = np.minimum(rq, lq)
    udq_min = np.minimum(uq, dq)
    rlq_max = np.maximum(rq, lq)
    udq_max = np.maximum(uq, dq)

    result: dict[str, Array] = {}
    result["min22h"] = _flatten_feature(cooc(rlq_min, order, "hor", t) + cooc(udq_min, order, "ver", t))
    result["max22h"] = _flatten_feature(cooc(rlq_max, order, "hor", t) + cooc(udq_max, order, "ver", t))

    uq_min = np.minimum(rlq_min, uq)
    rq_min = np.minimum(udq_min, rq)
    dq_min = np.minimum(rlq_min, dq)
    lq_min = np.minimum(udq_min, lq)
    uq_max = np.maximum(rlq_max, uq)
    rq_max = np.maximum(udq_max, rq)
    dq_max = np.maximum(rlq_max, dq)
    lq_max = np.maximum(udq_max, lq)

    result["min34h"] = _flatten_feature(
        cooc(np.vstack((uq_min, dq_min)), order, "hor", t)
        + cooc(np.hstack((rq_min, lq_min)), order, "ver", t)
    )
    result["max34h"] = _flatten_feature(
        cooc(np.vstack((uq_max, dq_max)), order, "hor", t)
        + cooc(np.hstack((rq_max, lq_max)), order, "ver", t)
    )

    result["spam14h"] = _flatten_feature(cooc(rq, order, "hor", t) + cooc(uq, order, "ver", t))
    result["spam14v"] = _flatten_feature(cooc(rq, order, "ver", t) + cooc(uq, order, "hor", t))
    result["min22v"] = _flatten_feature(cooc(rlq_min, order, "ver", t) + cooc(udq_min, order, "hor", t))
    result["max22v"] = _flatten_feature(cooc(rlq_max, order, "ver", t) + cooc(udq_max, order, "hor", t))

    ruq_min = np.minimum(rq, uq)
    rdq_min = np.minimum(rq, dq)
    luq_min = np.minimum(lq, uq)
    ldq_min = np.minimum(lq, dq)
    ruq_max = np.maximum(rq, uq)
    rdq_max = np.maximum(rq, dq)
    luq_max = np.maximum(lq, uq)
    ldq_max = np.maximum(lq, dq)

    result["min24"] = _flatten_feature(
        cooc(np.vstack((ruq_min, rdq_min, luq_min, ldq_min)), order, "hor", t)
        + cooc(np.hstack((ruq_min, rdq_min, luq_min, ldq_min)), order, "ver", t)
    )
    result["max24"] = _flatten_feature(
        cooc(np.vstack((ruq_max, rdq_max, luq_max, ldq_max)), order, "hor", t)
        + cooc(np.hstack((ruq_max, rdq_max, luq_max, ldq_max)), order, "ver", t)
    )

    result["min34v"] = _flatten_feature(
        cooc(np.hstack((uq_min, dq_min)), order, "ver", t)
        + cooc(np.vstack((rq_min, lq_min)), order, "hor", t)
    )
    result["max34v"] = _flatten_feature(
        cooc(np.hstack((uq_max, dq_max)), order, "ver", t)
        + cooc(np.vstack((rq_max, lq_max)), order, "hor", t)
    )

    r_min = np.minimum(ruq_min, ldq_min)
    r_max = np.maximum(ruq_max, ldq_max)
    result["min41"] = _flatten_feature(cooc(r_min, order, "hor", t) + cooc(r_min, order, "ver", t))
    result["max41"] = _flatten_feature(cooc(r_max, order, "hor", t) + cooc(r_max, order, "ver", t))

    ruq_min2 = np.minimum(ruq_min, ruq)
    rdq_min2 = np.minimum(rdq_min, rdq)
    luq_min2 = np.minimum(luq_min, luq)
    ldq_min2 = np.minimum(ldq_min, ldq)
    ruq_max2 = np.maximum(ruq_max, ruq)
    rdq_max2 = np.maximum(rdq_max, rdq)
    luq_max2 = np.maximum(luq_max, luq)
    ldq_max2 = np.maximum(ldq_max, ldq)

    result["min34"] = _flatten_feature(
        cooc(np.vstack((ruq_min2, rdq_min2, luq_min2, ldq_min2)), order, "hor", t)
        + cooc(np.hstack((ruq_min2, rdq_min2, luq_min2, ldq_min2)), order, "ver", t)
    )
    result["max34"] = _flatten_feature(
        cooc(np.vstack((ruq_max2, rdq_max2, luq_max2, ldq_max2)), order, "hor", t)
        + cooc(np.hstack((ruq_max2, rdq_max2, luq_max2, ldq_max2)), order, "ver", t)
    )

    ruq_min3 = np.minimum(ruq_min2, luq)
    rdq_min3 = np.minimum(rdq_min2, ruq)
    ldq_min3 = np.minimum(ldq_min2, rdq)
    luq_min3 = np.minimum(luq_min2, ldq)
    ruq_min4 = np.minimum(ruq_min2, rdq)
    rdq_min4 = np.minimum(rdq_min2, ldq)
    ldq_min4 = np.minimum(ldq_min2, luq)
    luq_min4 = np.minimum(luq_min2, ruq)
    result["min48h"] = _flatten_feature(
        cooc(np.vstack((ruq_min3, ldq_min3, rdq_min4, luq_min4)), order, "hor", t)
        + cooc(np.hstack((rdq_min3, luq_min3, ruq_min4, ldq_min4)), order, "ver", t)
    )
    result["min48v"] = _flatten_feature(
        cooc(np.hstack((ruq_min3, ldq_min3, rdq_min4, luq_min4)), order, "ver", t)
        + cooc(np.vstack((rdq_min3, luq_min3, ruq_min4, ldq_min4)), order, "hor", t)
    )

    ruq_max3 = np.maximum(ruq_max2, luq)
    rdq_max3 = np.maximum(rdq_max2, ruq)
    ldq_max3 = np.maximum(ldq_max2, rdq)
    luq_max3 = np.maximum(luq_max2, ldq)
    ruq_max4 = np.maximum(ruq_max2, rdq)
    rdq_max4 = np.maximum(rdq_max2, ldq)
    ldq_max4 = np.maximum(ldq_max2, luq)
    luq_max4 = np.maximum(luq_max2, ruq)
    result["max48h"] = _flatten_feature(
        cooc(np.vstack((ruq_max3, ldq_max3, rdq_max4, luq_max4)), order, "hor", t)
        + cooc(np.hstack((rdq_max3, luq_max3, ruq_max4, ldq_max4)), order, "ver", t)
    )
    result["max48v"] = _flatten_feature(
        cooc(np.hstack((ruq_max3, ldq_max3, rdq_max4, luq_max4)), order, "ver", t)
        + cooc(np.vstack((rdq_max3, luq_max3, ruq_max4, ldq_max4)), order, "hor", t)
    )

    ruq_min5 = np.minimum(ruq_min3, rdq)
    rdq_min5 = np.minimum(rdq_min3, ldq)
    ldq_min5 = np.minimum(ldq_min3, luq)
    luq_min5 = np.minimum(luq_min3, ruq)
    ruq_max5 = np.maximum(ruq_max3, rdq)
    rdq_max5 = np.maximum(rdq_max3, ldq)
    ldq_max5 = np.maximum(ldq_max3, luq)
    luq_max5 = np.maximum(luq_max3, ruq)
    result["min54"] = _flatten_feature(
        cooc(np.vstack((ruq_min5, ldq_min5, rdq_min5, luq_min5)), order, "hor", t)
        + cooc(np.hstack((rdq_min5, luq_min5, ruq_min5, ldq_min5)), order, "ver", t)
    )
    result["max54"] = _flatten_feature(
        cooc(np.vstack((ruq_max5, ldq_max5, rdq_max5, luq_max5)), order, "hor", t)
        + cooc(np.hstack((rdq_max5, luq_max5, ruq_max5, ldq_max5)), order, "ver", t)
    )

    return result


def all3x3(x: Array, q: float) -> dict[str, Array]:
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
        cooc(np.hstack((yu, yb)), order, "ver", t) + cooc(np.vstack((yl, yr)), order, "hor", t)
    )
    result["spam14h"] = _flatten_feature(
        cooc(np.vstack((yu, yb)), order, "hor", t) + cooc(np.hstack((yl, yr)), order, "ver", t)
    )

    dmin1 = np.minimum(yu, yl)
    dmin2 = np.minimum(yb, yr)
    dmin3 = np.minimum(yu, yr)
    dmin4 = np.minimum(yb, yl)
    result["min24"] = _flatten_feature(
        cooc(np.hstack((dmin1, dmin2, dmin3, dmin4)), order, "ver", t)
        + cooc(np.vstack((dmin1, dmin2, dmin3, dmin4)), order, "hor", t)
    )

    dmax1 = np.maximum(yu, yl)
    dmax2 = np.maximum(yb, yr)
    dmax3 = np.maximum(yu, yr)
    dmax4 = np.maximum(yb, yl)
    result["max24"] = _flatten_feature(
        cooc(np.hstack((dmax1, dmax2, dmax3, dmax4)), order, "ver", t)
        + cooc(np.vstack((dmax1, dmax2, dmax3, dmax4)), order, "hor", t)
    )

    ueq_min = np.minimum(yu, yb)
    req_min = np.minimum(yr, yl)
    result["min22h"] = _flatten_feature(cooc(ueq_min, order, "hor", t) + cooc(req_min, order, "ver", t))
    result["min22v"] = _flatten_feature(cooc(ueq_min, order, "ver", t) + cooc(req_min, order, "hor", t))

    ueq_max = np.maximum(yu, yb)
    req_max = np.maximum(yr, yl)
    result["max22h"] = _flatten_feature(cooc(ueq_max, order, "hor", t) + cooc(req_max, order, "ver", t))
    result["max22v"] = _flatten_feature(cooc(ueq_max, order, "ver", t) + cooc(req_max, order, "hor", t))

    dmin5 = np.minimum(dmin1, dmin2)
    dmax5 = np.maximum(dmax1, dmax2)
    result["min41"] = _flatten_feature(cooc(dmin5, order, "ver", t) + cooc(dmin5, order, "hor", t))
    result["max41"] = _flatten_feature(cooc(dmax5, order, "ver", t) + cooc(dmax5, order, "hor", t))

    return result


def all5x5(x: Array, q: float) -> dict[str, Array]:
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
        cooc(np.hstack((yu, yb)), order, "ver", t) + cooc(np.vstack((yl, yr)), order, "hor", t)
    )
    result["spam14h"] = _flatten_feature(
        cooc(np.vstack((yu, yb)), order, "hor", t) + cooc(np.hstack((yl, yr)), order, "ver", t)
    )

    dmin1 = np.minimum(yu, yl)
    dmin2 = np.minimum(yb, yr)
    dmin3 = np.minimum(yu, yr)
    dmin4 = np.minimum(yb, yl)
    result["min24"] = _flatten_feature(
        cooc(np.hstack((dmin1, dmin2, dmin3, dmin4)), order, "ver", t)
        + cooc(np.vstack((dmin1, dmin2, dmin3, dmin4)), order, "hor", t)
    )

    dmax1 = np.maximum(yu, yl)
    dmax2 = np.maximum(yb, yr)
    dmax3 = np.maximum(yu, yr)
    dmax4 = np.maximum(yb, yl)
    result["max24"] = _flatten_feature(
        cooc(np.hstack((dmax1, dmax2, dmax3, dmax4)), order, "ver", t)
        + cooc(np.vstack((dmax1, dmax2, dmax3, dmax4)), order, "hor", t)
    )

    ueq_min = np.minimum(yu, yb)
    req_min = np.minimum(yr, yl)
    result["min22h"] = _flatten_feature(cooc(ueq_min, order, "hor", t) + cooc(req_min, order, "ver", t))
    result["min22v"] = _flatten_feature(cooc(ueq_min, order, "ver", t) + cooc(req_min, order, "hor", t))

    ueq_max = np.maximum(yu, yb)
    req_max = np.maximum(yr, yl)
    result["max22h"] = _flatten_feature(cooc(ueq_max, order, "hor", t) + cooc(req_max, order, "ver", t))
    result["max22v"] = _flatten_feature(cooc(ueq_max, order, "ver", t) + cooc(req_max, order, "hor", t))

    dmin5 = np.minimum(dmin1, dmin2)
    dmax5 = np.maximum(dmax1, dmax2)
    result["min41"] = _flatten_feature(cooc(dmin5, order, "ver", t) + cooc(dmin5, order, "hor", t))
    result["max41"] = _flatten_feature(cooc(dmax5, order, "ver", t) + cooc(dmax5, order, "hor", t))

    return result


def cooc(d: Array, order: int, kind: str, t: int) -> Array:
    bins = 2 * t + 1
    arrays = _cooc_arrays(d, order, kind)
    counts = np.zeros((bins,) * order, dtype=np.float64)
    indices = tuple((part.ravel(order="F").astype(np.int16) + t) for part in arrays)
    np.add.at(counts, indices, 1)
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def _cooc_arrays(d: Array, order: int, kind: str) -> tuple[Array, ...]:
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


def quant(x: Array, q: float | Array, t: int) -> Array:
    if np.isscalar(q):
        step = float(q)
        if step <= 0:
            raise ValueError("Quantization step must be positive.")
        return trunc(_matlab_round(x / step), t).astype(np.int16)

    q = np.rint(np.asarray(q)).astype(np.int16)
    if np.any(np.diff(q) <= 0):
        raise ValueError("Quantization vector must be strictly increasing.")
    if np.min(q) < 0:
        raise ValueError("Quantization vector must be non-negative.")

    t = int(q[-1])
    values = np.zeros(2 * t + 1, dtype=np.int16)
    y = trunc(x, t).astype(np.int16) + t

    if q[0] == 0:
        values[t] = 0
        z = 1
        index = t + 1
        for current, previous in zip(q[1:], q[:-1]):
            span = int(current - previous)
            values[index:index + span] = z
            index += span
            z += 1
        values[:t] = -values[:t:-1]
    else:
        start = t - int(q[0])
        stop = t + int(q[0]) + 1
        values[start:stop] = 0
        z = 1
        index = t + 1 + int(q[0])
        for current, previous in zip(q[1:], q[:-1]):
            span = int(current - previous)
            values[index:index + span] = z
            index += span
            z += 1
        values[: t - int(q[0])] = -values[: t + 1 : -1]

    return values[y]


def trunc(x: Array, t: int) -> Array:
    return np.clip(x, -t, t)


def _matlab_round(x: Array) -> Array:
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def symfea(f: Array, t: int, order: int, kind: str) -> Array:
    dim, columns = f.shape
    bins = 2 * t + 1
    cell_count = bins ** order

    if kind != "mnmx" or dim != 2 * cell_count:
        raise ValueError("Only the minmax SRM symmetrization used by SRM is supported.")

    if order == 3:
        reduced = bins**3 - t * bins**2
    elif order == 4:
        reduced = bins**4 - 2 * t * (t + 1) * bins**2
    else:
        raise ValueError("Unsupported order for minmax symmetrization.")

    output = np.zeros((reduced, columns), dtype=np.float64)
    for column in range(columns):
        cube_min = f[:cell_count, column].reshape((bins,) * order, order="F")
        cube_max = f[cell_count:, column].reshape((bins,) * order, order="F")
        signsym = cube_min + np.flip(cube_max)
        output[:, column] = symm_dir(signsym, t, order)
    return output


def symm_dir(a: Array, t: int, order: int) -> Array:
    bins = 2 * t + 1
    done = np.zeros_like(a, dtype=bool)
    if order == 3:
        reduced = bins**3 - t * bins**2
    elif order == 4:
        reduced = bins**4 - 2 * t * (t + 1) * bins**2
    elif order == 5:
        reduced = bins**5 - 2 * t * (t + 1) * bins**3
    else:
        raise ValueError("Unsupported order for directional symmetrization.")

    output = np.zeros(reduced, dtype=np.float64)
    index = 0

    if order == 3:
        for i in range(-t, t + 1):
            for j in range(-t, t + 1):
                for k in range(-t, t + 1):
                    p1 = (i + t, j + t, k + t)
                    p2 = (k + t, j + t, i + t)
                    if k != i:
                        if not done[p1]:
                            output[index] = a[p1] + a[p2]
                            done[p1] = True
                            done[p2] = True
                            index += 1
                    else:
                        output[index] = a[p1]
                        done[p1] = True
                        index += 1
    elif order == 4:
        for i in range(-t, t + 1):
            for j in range(-t, t + 1):
                for k in range(-t, t + 1):
                    for n in range(-t, t + 1):
                        p1 = (i + t, j + t, k + t, n + t)
                        p2 = (n + t, k + t, j + t, i + t)
                        if (i != n) or (j != k):
                            if not done[p1]:
                                output[index] = a[p1] + a[p2]
                                done[p1] = True
                                done[p2] = True
                                index += 1
                        else:
                            output[index] = a[p1]
                            done[p1] = True
                            index += 1
    else:
        for i in range(-t, t + 1):
            for j in range(-t, t + 1):
                for k in range(-t, t + 1):
                    for l in range(-t, t + 1):
                        for n in range(-t, t + 1):
                            p1 = (i + t, j + t, k + t, l + t, n + t)
                            p2 = (n + t, l + t, k + t, j + t, i + t)
                            if (i != n) or (j != l):
                                if not done[p1]:
                                    output[index] = a[p1] + a[p2]
                                    done[p1] = True
                                    done[p2] = True
                                    index += 1
                            else:
                                output[index] = a[p1]
                                done[p1] = True
                                index += 1

    return output


def symm1(f: Array, t: int, order: int) -> Array:
    dim, columns = f.shape
    bins = 2 * t + 1
    cell_count = bins**order
    if dim != cell_count:
        raise ValueError("Feature dimension is incompatible with T and order.")

    if order == 1:
        reduced = t + 1
    elif order == 2:
        reduced = (t + 1) ** 2
    elif order == 3:
        reduced = 1 + 3 * t + 4 * t**2 + 2 * t**3
    elif order == 4:
        reduced = bins**2 + 4 * t**2 * (t + 1) ** 2
    elif order == 5:
        reduced = ((bins**2 + 1) * (bins**3 + 1)) // 4
    else:
        raise ValueError("Unsupported order for symmetry marginalization.")

    output = np.zeros((reduced, columns), dtype=np.float64)
    for column in range(columns):
        cube = f[:, column].reshape((bins,) * order, order="F")
        output[:, column] = symm(cube, t, order)
    return output


def symm(a: Array, t: int, order: int) -> Array:
    bins = 2 * t + 1
    index = 1

    if order == 1:
        if a.size != bins:
            raise ValueError("Array size does not match order 1.")
        output = np.zeros(t + 1, dtype=np.float64)
        output[0] = a[t]
        output[1:] = a[:t] + a[t + 1 :]
        return output

    if order == 2:
        if a.size != bins**2:
            raise ValueError("Array size does not match order 2.")
        done = np.zeros_like(a, dtype=bool)
        output = np.zeros((t + 1) ** 2, dtype=np.float64)
        output[0] = a[t, t]
        for i in range(-t, t + 1):
            for j in range(-t, t + 1):
                p1 = (i + t, j + t)
                p2 = (t - i, t - j)
                if not done[p1] and (abs(i) + abs(j) != 0):
                    output[index] = a[p1] + a[p2]
                    done[p1] = True
                    done[p2] = True
                    p3 = (j + t, i + t)
                    p4 = (t - j, t - i)
                    if i != j and not done[p3]:
                        output[index] += a[p3] + a[p4]
                        done[p3] = True
                        done[p4] = True
                    index += 1
        return output

    if order == 3:
        if a.size != bins**3:
            raise ValueError("Array size does not match order 3.")
        done = np.zeros_like(a, dtype=bool)
        output = np.zeros(1 + 3 * t + 4 * t**2 + 2 * t**3, dtype=np.float64)
        output[0] = a[t, t, t]
        for i in range(-t, t + 1):
            for j in range(-t, t + 1):
                for k in range(-t, t + 1):
                    p1 = (i + t, j + t, k + t)
                    p2 = (t - i, t - j, t - k)
                    if not done[p1] and (abs(i) + abs(j) + abs(k) != 0):
                        output[index] = a[p1] + a[p2]
                        done[p1] = True
                        done[p2] = True
                        p3 = (k + t, j + t, i + t)
                        p4 = (t - k, t - j, t - i)
                        if i != k and not done[p3]:
                            output[index] += a[p3] + a[p4]
                            done[p3] = True
                            done[p4] = True
                        index += 1
        return output

    if order == 4:
        if a.size != bins**4:
            raise ValueError("Array size does not match order 4.")
        done = np.zeros_like(a, dtype=bool)
        output = np.zeros(bins**2 + 4 * t**2 * (t + 1) ** 2, dtype=np.float64)
        output[0] = a[t, t, t, t]
        for i in range(-t, t + 1):
            for j in range(-t, t + 1):
                for k in range(-t, t + 1):
                    for n in range(-t, t + 1):
                        p1 = (i + t, j + t, k + t, n + t)
                        p2 = (t - i, t - j, t - k, t - n)
                        if not done[p1] and (abs(i) + abs(j) + abs(k) + abs(n) != 0):
                            output[index] = a[p1] + a[p2]
                            done[p1] = True
                            done[p2] = True
                            p3 = (n + t, k + t, j + t, i + t)
                            p4 = (t - n, t - k, t - j, t - i)
                            if ((i != n) or (j != k)) and not done[p3]:
                                output[index] += a[p3] + a[p4]
                                done[p3] = True
                                done[p4] = True
                            index += 1
        return output

    if order == 5:
        if a.size != bins**5:
            raise ValueError("Array size does not match order 5.")
        done = np.zeros_like(a, dtype=bool)
        output = np.zeros(((bins**2 + 1) * (bins**3 + 1)) // 4, dtype=np.float64)
        output[0] = a[t, t, t, t, t]
        for i in range(-t, t + 1):
            for j in range(-t, t + 1):
                for k in range(-t, t + 1):
                    for l in range(-t, t + 1):
                        for n in range(-t, t + 1):
                            p1 = (i + t, j + t, k + t, l + t, n + t)
                            p2 = (t - i, t - j, t - k, t - l, t - n)
                            if not done[p1] and (abs(i) + abs(j) + abs(k) + abs(l) + abs(n) != 0):
                                output[index] = a[p1] + a[p2]
                                done[p1] = True
                                done[p2] = True
                                p3 = (n + t, l + t, k + t, j + t, i + t)
                                p4 = (t - n, t - l, t - k, t - j, t - i)
                                if ((i != n) or (j != l)) and not done[p3]:
                                    output[index] += a[p3] + a[p4]
                                    done[p3] = True
                                    done[p4] = True
                                index += 1
        return output

    raise ValueError("Order of co-occurrence is not in {1,2,3,4,5}.")


def residual(x: Array, order: int, kind: str) -> Array:
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
        return np.hstack((du, db))

    if kind == "edge-v":
        dl = 2 * x[i, border - 1 : n - border - 1] + 2 * x[border - 1 : m - border - 1, j] + 2 * x[border + 1 : m - border + 1, j] - x[border - 1 : m - border - 1, border - 1 : n - border - 1] - x[border + 1 : m - border + 1, border - 1 : n - border - 1] - 4 * x[i, j]
        dr = 2 * x[i, border + 1 : n - border + 1] + 2 * x[border - 1 : m - border - 1, j] + 2 * x[border + 1 : m - border + 1, j] - x[border - 1 : m - border - 1, border + 1 : n - border + 1] - x[border + 1 : m - border + 1, border + 1 : n - border + 1] - 4 * x[i, j]
        return np.hstack((dl, dr))

    if kind == "edge-m":
        dlu = 2 * x[i, border - 1 : n - border - 1] + 2 * x[border - 1 : m - border - 1, j] - x[border - 1 : m - border - 1, border - 1 : n - border - 1] - x[border + 1 : m - border + 1, border - 1 : n - border - 1] - x[border - 1 : m - border - 1, border + 1 : n - border + 1] - x[i, j]
        drb = 2 * x[i, border + 1 : n - border + 1] + 2 * x[border + 1 : m - border + 1, j] - x[border + 1 : m - border + 1, border + 1 : n - border + 1] - x[border + 1 : m - border + 1, border - 1 : n - border - 1] - x[border - 1 : m - border - 1, border + 1 : n - border + 1] - x[i, j]
        return np.hstack((dlu, drb))

    if kind == "edge-d":
        dru = 2 * x[border - 1 : m - border - 1, j] + 2 * x[i, border + 1 : n - border + 1] - x[border - 1 : m - border - 1, border + 1 : n - border + 1] - x[border - 1 : m - border - 1, border - 1 : n - border - 1] - x[border + 1 : m - border + 1, border + 1 : n - border + 1] - x[i, j]
        dlb = 2 * x[i, border - 1 : n - border - 1] + 2 * x[border + 1 : m - border + 1, j] - x[border + 1 : m - border + 1, border - 1 : n - border - 1] - x[border + 1 : m - border + 1, border + 1 : n - border + 1] - x[border - 1 : m - border - 1, border - 1 : n - border - 1] - x[i, j]
        return np.hstack((dru, dlb))

    if kind == "KV":
        result = 8 * x[border - 1 : m - border - 1, j] + 8 * x[border + 1 : m - border + 1, j] + 8 * x[i, border - 1 : n - border - 1] + 8 * x[i, border + 1 : n - border + 1]
        result = result - 6 * x[border - 1 : m - border - 1, border + 1 : n - border + 1] - 6 * x[border - 1 : m - border - 1, border - 1 : n - border - 1] - 6 * x[border + 1 : m - border + 1, border - 1 : n - border - 1] - 6 * x[border + 1 : m - border + 1, border + 1 : n - border + 1]
        result = result - 2 * x[border - 2 : m - border - 2, j] - 2 * x[border + 2 : m - border + 2, j] - 2 * x[i, border + 2 : n - border + 2] - 2 * x[i, border - 2 : n - border - 2]
        result = result + 2 * x[border - 1 : m - border - 1, border - 2 : n - border - 2] + 2 * x[border - 2 : m - border - 2, border - 1 : n - border - 1] + 2 * x[border - 2 : m - border - 2, border + 1 : n - border + 1] + 2 * x[border - 1 : m - border - 1, border + 2 : n - border + 2] + 2 * x[border + 1 : m - border + 1, border + 2 : n - border + 2] + 2 * x[border + 2 : m - border + 2, border + 1 : n - border + 1] + 2 * x[border + 2 : m - border + 2, border - 1 : n - border - 1] + 2 * x[border + 1 : m - border + 1, border - 2 : n - border - 2]
        result = result - x[border - 2 : m - border - 2, border - 2 : n - border - 2] - x[border - 2 : m - border - 2, border + 2 : n - border + 2] - x[border + 2 : m - border + 2, border - 2 : n - border - 2] - x[border + 2 : m - border + 2, border + 2 : n - border + 2] - 12 * x[i, j]
        return result

    raise ValueError(f"Unsupported residual kind: {kind}")


def _flatten_feature(value: Array) -> Array:
    return np.asarray(value, dtype=np.float64).reshape(-1, order="F")


def _q_suffix(q: float) -> str:
    return f"{q:g}".replace(".", "")
