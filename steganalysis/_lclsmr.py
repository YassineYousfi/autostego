from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.sparse.linalg import lsmr as scipy_lsmr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted


@dataclass(slots=True)
class LCLSMRResults:
    pfa: np.ndarray
    pmd: np.ndarray
    pe: float
    selected_fold: int
    selected_tolerance: float
    projection: np.ndarray
    validation_error: np.ndarray
    validation_min_error: np.ndarray


def _matlab_round(value: float) -> int:
    return int(np.floor(value + 0.5))


def _validate_feature_matrices(
    trn_cover: np.ndarray,
    trn_stego: np.ndarray,
    tst_cover: np.ndarray,
    tst_stego: np.ndarray,
) -> None:
    if trn_cover.ndim != 2 or trn_stego.ndim != 2 or tst_cover.ndim != 2 or tst_stego.ndim != 2:
        raise ValueError("All feature matrices must be 2D.")
    if trn_cover.shape != trn_stego.shape:
        raise ValueError("Training cover/stego feature matrices must have the same shape.")
    if tst_cover.shape != tst_stego.shape:
        raise ValueError("Testing cover/stego feature matrices must have the same shape.")
    if trn_cover.shape[1] != tst_cover.shape[1]:
        raise ValueError("Training and testing features must share the same feature dimension.")


def _greater_counts(sorted_cover: np.ndarray, stego: np.ndarray) -> np.ndarray:
    sorted_cover = np.asarray(sorted_cover, dtype=np.float64).reshape(-1)
    sorted_stego = np.sort(np.asarray(stego, dtype=np.float64).reshape(-1))
    less_equal_counts = np.searchsorted(sorted_stego, sorted_cover, side="right")
    return sorted_stego.size - less_equal_counts


def _compute_error_curve(proj_cover: np.ndarray, proj_stego: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    sorted_cover = np.sort(np.asarray(proj_cover, dtype=np.float64).reshape(-1))
    stego = np.asarray(proj_stego, dtype=np.float64).reshape(-1)
    if sorted_cover.size == 0 or stego.size == 0:
        raise ValueError("Cover and stego projections must be non-empty.")

    greater_counts = _greater_counts(sorted_cover, stego)
    pfa = 1.0 - (np.arange(1, sorted_cover.size + 1, dtype=np.float64) / sorted_cover.size)
    pmd = 1.0 - (greater_counts.astype(np.float64) / stego.size)
    pe = float(np.min((pfa + pmd) / 2.0))
    return pfa, pmd, pe


def _compute_training_threshold(proj_cover: np.ndarray, proj_stego: np.ndarray) -> float:
    sorted_cover = np.sort(np.asarray(proj_cover, dtype=np.float64).reshape(-1))
    stego = np.asarray(proj_stego, dtype=np.float64).reshape(-1)
    greater_counts = _greater_counts(sorted_cover, stego)
    pfa = 1.0 - (np.arange(1, sorted_cover.size + 1, dtype=np.float64) / sorted_cover.size)
    pmd = 1.0 - (greater_counts.astype(np.float64) / stego.size)
    pe_curve = (pfa + pmd) / 2.0
    return float(sorted_cover[int(np.argmin(pe_curve))])


def _compute_error_matrix(proj_cover: np.ndarray, proj_stego: np.ndarray) -> np.ndarray:
    proj_cover = np.asarray(proj_cover, dtype=np.float64)
    proj_stego = np.asarray(proj_stego, dtype=np.float64)
    if proj_cover.ndim == 1:
        proj_cover = proj_cover[:, None]
    if proj_stego.ndim == 1:
        proj_stego = proj_stego[:, None]

    n_samples, n_tolerances = proj_cover.shape
    sorted_cover = np.sort(proj_cover, axis=0)
    pfa = 1.0 - (np.arange(1, n_samples + 1, dtype=np.float64)[:, None] / n_samples)
    pe = np.empty(n_tolerances, dtype=np.float64)
    for column in range(n_tolerances):
        greater_counts = _greater_counts(sorted_cover[:, column], proj_stego[:, column])
        pmd = 1.0 - (greater_counts.astype(np.float64) / proj_stego.shape[0])
        pe[column] = np.min((pfa[:, 0] + pmd) / 2.0)
    return pe


def _lsmr_path(
    a: np.ndarray,
    b: np.ndarray,
    lambda_: float,
    atol: np.ndarray,
    btol: np.ndarray,
    conlim: float = 1e8,
    itnlim: int = 30000,
    local_size: float = np.inf,
) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    b = np.ascontiguousarray(np.asarray(b, dtype=np.float64).reshape(-1))
    if a.ndim != 2:
        raise ValueError("A must be a 2D array.")
    if b.ndim != 1 or b.shape[0] != a.shape[0]:
        raise ValueError("b must be a 1D array with length equal to A.shape[0].")

    atol_all = np.ascontiguousarray(np.sort(np.asarray(atol, dtype=np.float64))[::-1])
    btol_all = np.ascontiguousarray(np.sort(np.asarray(btol, dtype=np.float64))[::-1])
    if atol_all.shape != btol_all.shape:
        raise ValueError("atol and btol must have the same shape.")

    m, n = a.shape
    u = b.copy()
    beta = np.linalg.norm(u)
    if beta > 0:
        u = u / beta

    v = a.T @ u
    alpha = np.linalg.norm(v)
    if alpha > 0:
        v = v / alpha

    norm_ar = alpha * beta
    x_all = np.zeros((n, atol_all.size), dtype=np.float64)
    if norm_ar == 0:
        return x_all

    local_ortho = local_size > 0
    local_pointer = 0
    local_queue_full = False
    if local_ortho:
        effective_local_size = min(n if np.isinf(local_size) else int(local_size), min(m, n))
        local_v = np.zeros((n, effective_local_size), dtype=np.float64)
    else:
        effective_local_size = 0
        local_v = np.empty((n, 0), dtype=np.float64)

    def local_v_enqueue(vector: np.ndarray) -> None:
        nonlocal local_pointer, local_queue_full
        if local_pointer < effective_local_size:
            local_pointer += 1
        else:
            local_pointer = 1
            local_queue_full = True
        local_v[:, local_pointer - 1] = vector

    def local_v_ortho(vector: np.ndarray) -> np.ndarray:
        if not local_ortho:
            return vector
        limit = effective_local_size if local_queue_full else local_pointer
        output = vector
        for column in range(limit):
            cached = np.ascontiguousarray(local_v[:, column])
            output = output - np.dot(output, cached) * cached
        return output

    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1.0
    rhobar = 1.0
    cbar = 1.0
    sbar = 0.0
    h = v.copy()
    hbar = np.zeros(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    betadd = beta
    betad = 0.0
    rhodold = 1.0
    tautildeold = 0.0
    thetatilde = 0.0
    zeta = 0.0
    d = 0.0
    norm_a2 = alpha**2
    maxrbar = 0.0
    minrbar = 1e100
    normb = beta
    istop = np.zeros(atol_all.size, dtype=np.int64)
    ctol = 0.0 if conlim <= 0 else 1.0 / conlim
    index_tol = 0
    atol_current = atol_all[index_tol]
    btol_current = btol_all[index_tol]

    while itn < itnlim:
        itn += 1

        u = a @ v - alpha * u
        beta = np.linalg.norm(u)

        if beta > 0:
            u = u / beta
            if local_ortho:
                local_v_enqueue(v)
            v = a.T @ u - beta * v
            if local_ortho:
                v = local_v_ortho(v)
            alpha = np.linalg.norm(v)
            if alpha > 0:
                v = v / alpha

        alphahat = np.hypot(alphabar, lambda_)
        chat = alphabar / alphahat
        shat = lambda_ / alphahat

        rhoold = rho
        rho = np.hypot(alphahat, beta)
        c = alphahat / rho
        s = beta / rho
        thetanew = s * alpha
        alphabar = c * alpha

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        rhobar = np.hypot(cbar * rho, thetanew)
        cbar = cbar * rho / rhobar
        sbar = thetanew / rhobar
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        hbar = h - (thetabar * rho / (rhoold * rhobarold)) * hbar
        x = x + (zeta / (rho * rhobar)) * hbar
        h = v - (thetanew / rho) * h

        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute

        thetatildeold = thetatilde
        rhotildeold = np.hypot(rhodold, thetabar)
        ctildeold = rhodold / rhotildeold
        stildeold = thetabar / rhotildeold
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck**2
        normr = np.sqrt(d + (betad - taud) ** 2 + betadd**2)

        norm_a2 = norm_a2 + beta**2
        norm_a = np.sqrt(norm_a2)
        norm_a2 = norm_a2 + alpha**2

        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        cond_a = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        norm_ar = abs(zetabar)
        normx = np.linalg.norm(x)
        test1 = normr / normb
        test2 = norm_ar / (norm_a * normr)
        test3 = 1.0 / cond_a
        t1 = test1 / (1.0 + norm_a * normx / normb)
        rtol = btol_current + atol_current * norm_a * normx / normb

        if itn >= itnlim:
            istop[index_tol] = 7
        if 1.0 + test3 <= 1.0:
            istop[index_tol] = 6
        if 1.0 + test2 <= 1.0:
            istop[index_tol] = 5
        if 1.0 + t1 <= 1.0:
            istop[index_tol] = 4

        if test3 <= ctol:
            istop[index_tol] = 3
        if test2 <= atol_current:
            istop[index_tol] = 2
        if test1 <= rtol:
            istop[index_tol] = 1

        if istop[index_tol] > 0:
            x_all[:, atol_all.size - index_tol - 1] = x
            index_tol += 1
            if index_tol >= atol_all.size:
                break
            atol_current = atol_all[index_tol]
            btol_current = btol_all[index_tol]

    return x_all


def _lclsmr(
    trn_cover: np.ndarray,
    trn_stego: np.ndarray,
    tst_cover: np.ndarray,
    tst_stego: np.ndarray,
    verbose: bool = False,
    num_folds: int = 5,
    lambda_: float = 1e-8,
    tolerance: np.ndarray | None = None,
    random_state: int | np.random.Generator | None = None,
    permutation: np.ndarray | None = None,
) -> LCLSMRResults:
    del verbose
    trn_cover = np.asarray(trn_cover, dtype=np.float64)
    trn_stego = np.asarray(trn_stego, dtype=np.float64)
    tst_cover = np.asarray(tst_cover, dtype=np.float64)
    tst_stego = np.asarray(tst_stego, dtype=np.float64)
    _validate_feature_matrices(trn_cover, trn_stego, tst_cover, tst_stego)

    if tolerance is None:
        tolerance = (2.0 ** (-np.arange(0, 21, dtype=np.float64))) * 1e-6
    tolerance = np.asarray(tolerance, dtype=np.float64)

    num_trn = trn_cover.shape[0]
    num_fold_samples = _matlab_round(num_trn / num_folds)
    if num_fold_samples <= 0 or num_fold_samples >= num_trn:
        raise ValueError("Training set is too small for the requested number of folds.")

    fold_index = np.zeros(num_trn, dtype=bool)
    fold_index[:num_fold_samples] = True
    if permutation is not None:
        random_integers = np.asarray(permutation, dtype=np.int64).reshape(-1)
        if random_integers.shape[0] != num_trn:
            raise ValueError("permutation must have one entry per training pair.")
        if set(random_integers.tolist()) != set(range(num_trn)):
            raise ValueError("permutation must contain each training index exactly once.")
    elif isinstance(random_state, np.random.Generator):
        random_integers = random_state.permutation(num_trn)
    elif random_state is None:
        random_integers = np.random.permutation(num_trn)
    else:
        random_integers = np.random.default_rng(random_state).permutation(num_trn)

    validation_error = np.zeros((num_folds, tolerance.size), dtype=np.float64)
    validation_min_error = np.zeros(num_folds, dtype=np.float64)
    validation_min_index = np.zeros(num_folds, dtype=np.int64)
    best_fold_index = -1
    best_tol_index = -1
    best_projection: np.ndarray | None = None
    best_error = np.inf

    for fold in range(num_folds):
        cv_learning_ind = random_integers[~fold_index]
        cv_validation_ind = random_integers[fold_index]
        fold_index = np.roll(fold_index, num_fold_samples)

        cv_learning_set = np.vstack((trn_cover[cv_learning_ind], trn_stego[cv_learning_ind]))
        learning_size = cv_learning_ind.size
        y_learning = np.concatenate((-np.ones(learning_size, dtype=np.float64), np.ones(learning_size, dtype=np.float64)))
        weights = _lsmr_path(
            cv_learning_set,
            y_learning,
            lambda_=lambda_,
            atol=tolerance,
            btol=tolerance,
            local_size=np.inf,
        )

        proj_validation_cover = trn_cover[cv_validation_ind] @ weights
        proj_validation_stego = trn_stego[cv_validation_ind] @ weights
        validation_error[fold] = _compute_error_matrix(proj_validation_cover, proj_validation_stego)
        validation_min_index[fold] = int(np.argmin(validation_error[fold]))
        validation_min_error[fold] = float(validation_error[fold, validation_min_index[fold]])
        if validation_min_error[fold] < best_error:
            best_error = validation_min_error[fold]
            best_fold_index = fold
            best_tol_index = validation_min_index[fold]
            best_projection = weights[:, best_tol_index].copy()

    if best_projection is None:
        raise RuntimeError("Failed to select an LCLSMR projection.")

    proj_testing_cover = tst_cover @ best_projection
    proj_testing_stego = tst_stego @ best_projection
    pfa, pmd, pe = _compute_error_curve(proj_testing_cover, proj_testing_stego)
    return LCLSMRResults(
        pfa=pfa,
        pmd=pmd,
        pe=pe,
        selected_fold=best_fold_index + 1,
        selected_tolerance=float(tolerance[best_tol_index]),
        projection=best_projection,
        validation_error=validation_error,
        validation_min_error=validation_min_error,
    )


def _lclsmr_scipy_cv(
    trn_cover: np.ndarray,
    trn_stego: np.ndarray,
    tst_cover: np.ndarray,
    tst_stego: np.ndarray,
    *,
    lambda_: float = 1e-8,
    tolerance_grid: np.ndarray | None = None,
    num_folds: int = 3,
    random_state: int | np.random.Generator | None = None,
    conlim: float = 1e8,
    maxiter: int = 30000,
) -> LCLSMRResults:
    trn_cover = np.asarray(trn_cover, dtype=np.float64)
    trn_stego = np.asarray(trn_stego, dtype=np.float64)
    tst_cover = np.asarray(tst_cover, dtype=np.float64)
    tst_stego = np.asarray(tst_stego, dtype=np.float64)
    _validate_feature_matrices(trn_cover, trn_stego, tst_cover, tst_stego)

    if tolerance_grid is None:
        tolerance_grid = np.array([1e-4, 3e-5, 1e-5, 3e-6, 1e-6], dtype=np.float64)
    tolerance_grid = np.asarray(tolerance_grid, dtype=np.float64).reshape(-1)
    if tolerance_grid.size == 0:
        raise ValueError("tolerance_grid must contain at least one value.")

    num_trn = trn_cover.shape[0]
    num_fold_samples = _matlab_round(num_trn / num_folds)
    if num_fold_samples <= 0 or num_fold_samples >= num_trn:
        raise ValueError("Training set is too small for the requested number of folds.")

    fold_index = np.zeros(num_trn, dtype=bool)
    fold_index[:num_fold_samples] = True
    if isinstance(random_state, np.random.Generator):
        random_integers = random_state.permutation(num_trn)
    elif random_state is None:
        random_integers = np.random.permutation(num_trn)
    else:
        random_integers = np.random.default_rng(random_state).permutation(num_trn)

    validation_error = np.zeros((num_folds, tolerance_grid.size), dtype=np.float64)
    validation_min_error = np.zeros(num_folds, dtype=np.float64)
    validation_min_index = np.zeros(num_folds, dtype=np.int64)

    for fold in range(num_folds):
        cv_learning_ind = random_integers[~fold_index]
        cv_validation_ind = random_integers[fold_index]
        fold_index = np.roll(fold_index, num_fold_samples)

        learning_set = np.vstack((trn_cover[cv_learning_ind], trn_stego[cv_learning_ind]))
        y_learning = np.concatenate(
            (-np.ones(cv_learning_ind.size, dtype=np.float64), np.ones(cv_learning_ind.size, dtype=np.float64))
        )

        for tolerance_index, tolerance in enumerate(tolerance_grid):
            projection = scipy_lsmr(
                learning_set,
                y_learning,
                damp=lambda_,
                atol=float(tolerance),
                btol=float(tolerance),
                conlim=float(conlim),
                maxiter=int(maxiter),
            )[0]
            proj_validation_cover = trn_cover[cv_validation_ind] @ projection
            proj_validation_stego = trn_stego[cv_validation_ind] @ projection
            validation_error[fold, tolerance_index] = _compute_error_curve(proj_validation_cover, proj_validation_stego)[2]

        validation_min_index[fold] = int(np.argmin(validation_error[fold]))
        validation_min_error[fold] = float(validation_error[fold, validation_min_index[fold]])

    mean_error = validation_error.mean(axis=0)
    best_tolerance = float(tolerance_grid[int(np.argmin(mean_error))])

    learning_set = np.vstack((trn_cover, trn_stego))
    y_learning = np.concatenate(
        (-np.ones(trn_cover.shape[0], dtype=np.float64), np.ones(trn_stego.shape[0], dtype=np.float64))
    )
    projection = scipy_lsmr(
        learning_set,
        y_learning,
        damp=lambda_,
        atol=best_tolerance,
        btol=best_tolerance,
        conlim=float(conlim),
        maxiter=int(maxiter),
    )[0]

    proj_testing_cover = tst_cover @ projection
    proj_testing_stego = tst_stego @ projection
    pfa, pmd, pe = _compute_error_curve(proj_testing_cover, proj_testing_stego)
    return LCLSMRResults(
        pfa=pfa,
        pmd=pmd,
        pe=pe,
        selected_fold=int(np.argmin(validation_min_error)) + 1,
        selected_tolerance=best_tolerance,
        projection=projection,
        validation_error=validation_error,
        validation_min_error=validation_min_error,
    )


class LCLSMRClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        *,
        lambda_: float = 1e-8,
        random_state: int | None = 1337,
        cv_tolerance_grid: np.ndarray | None = None,
        cv_num_folds: int = 3,
        cv_maxiter: int = 30000,
    ) -> None:
        self.lambda_ = lambda_
        self.random_state = random_state
        self.cv_tolerance_grid = cv_tolerance_grid
        self.cv_num_folds = cv_num_folds
        self.cv_maxiter = cv_maxiter

    def fit(self, x: np.ndarray, y: np.ndarray, pair_ids: np.ndarray | None = None) -> "LCLSMRClassifier":
        x = check_array(x, ensure_2d=True, dtype=np.float64)
        y = np.asarray(y)
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("y must be a 1D array with the same number of rows as X.")

        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("LCLSMRClassifier supports binary classification only.")
        self.classes_ = classes

        class0_mask = y == classes[0]
        class1_mask = y == classes[1]
        x_cover = x[class0_mask]
        x_stego = x[class1_mask]
        if x_cover.shape[0] != x_stego.shape[0]:
            raise ValueError("Both classes must contain the same number of paired samples.")

        if pair_ids is not None:
            pair_ids = np.asarray(pair_ids)
            if pair_ids.shape[0] != x.shape[0]:
                raise ValueError("pair_ids must align with X.")
            cover_pair_ids = pair_ids[class0_mask]
            stego_pair_ids = pair_ids[class1_mask]
            if set(np.asarray(cover_pair_ids).tolist()) != set(np.asarray(stego_pair_ids).tolist()):
                raise ValueError("Both classes must contain the same pair_ids.")
            order_cover = np.argsort(cover_pair_ids, kind="mergesort")
            order_stego = np.argsort(stego_pair_ids, kind="mergesort")
            x_cover = x_cover[order_cover]
            x_stego = x_stego[order_stego]

        results = _lclsmr_scipy_cv(
            x_cover,
            x_stego,
            x_cover,
            x_stego,
            lambda_=self.lambda_,
            tolerance_grid=self.cv_tolerance_grid,
            num_folds=self.cv_num_folds,
            random_state=self.random_state,
            maxiter=self.cv_maxiter,
        )
        self.results_ = results
        self.coef_ = results.projection.reshape(1, -1)
        self.intercept_ = np.array([0.0], dtype=np.float64)
        self.n_features_in_ = x.shape[1]
        self.selected_fold_ = results.selected_fold
        self.selected_tolerance_ = results.selected_tolerance
        self.threshold_ = _compute_training_threshold(x_cover @ results.projection, x_stego @ results.projection)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("coef_", "threshold_", "classes_"))
        x = check_array(x, ensure_2d=True, dtype=np.float64)
        return x @ self.coef_.ravel() - self.threshold_

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        indices = (scores > 0).astype(np.int64)
        return self.classes_[indices]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        positive = expit(scores)
        return np.column_stack((1.0 - positive, positive))

    def predict_log_proba(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(x)
        return np.log(np.clip(probabilities, np.finfo(np.float64).tiny, 1.0))


__all__ = ["LCLSMRClassifier", "LCLSMRResults", "_lclsmr"]
