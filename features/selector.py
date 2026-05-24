import numpy as np
import hashlib
import os
import tempfile
from pathlib import Path
from sklearn.linear_model import MultiTaskLassoCV
from config import Config
from utils.logger import Logger

class FeatureSelector:
    ALPHAS = np.logspace(-4, 1, 15)
    NONZERO_TOL = 1e-6

    @staticmethod
    def _cache_dir():
        cache_dir = Path(Config.FEATURE_SELECTOR_CACHE_DIR)
        if cache_dir.is_absolute():
            return cache_dir
        return Path(__file__).resolve().parents[1] / cache_dir

    @staticmethod
    def _make_cache_key(X_train, Y_train):
        hasher = hashlib.blake2b(digest_size=16)
        for arr in (X_train, Y_train):
            arr = np.ascontiguousarray(arr)
            hasher.update(str(arr.shape).encode("utf-8"))
            hasher.update(str(arr.dtype).encode("utf-8"))
            hasher.update(arr.tobytes())
        hasher.update(str(Config.INNER_SPLITS).encode("utf-8"))
        hasher.update(str(Config.RANDOM_STATE).encode("utf-8"))
        hasher.update(FeatureSelector.ALPHAS.tobytes())
        hasher.update(str(FeatureSelector.NONZERO_TOL).encode("utf-8"))
        return hasher.hexdigest()

    @staticmethod
    def _rank_shared_indices(coef_abs, importance_all):
        nonzero_any = np.any(coef_abs > FeatureSelector.NONZERO_TOL, axis=0)
        nonzero_idx = np.where(nonzero_any)[0]

        if len(nonzero_idx) == 0:
            return np.argsort(importance_all)[::-1]

        sorted_nonzero = np.argsort(importance_all[nonzero_idx])[::-1]
        return nonzero_idx[sorted_nonzero]

    @staticmethod
    def _load_cached_ranking(cache_path):
        if not (Config.FEATURE_SELECTOR_CACHE_ENABLED and cache_path.exists()):
            return None
        try:
            cached = np.load(cache_path, allow_pickle=False)
            return cached["ranked_idx"], cached["importance_all"]
        except Exception:
            return None

    @staticmethod
    def _save_cached_ranking(cache_path, ranked_idx, importance_all):
        if not Config.FEATURE_SELECTOR_CACHE_ENABLED:
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=cache_path.parent,
                prefix=cache_path.stem + "_",
                suffix=".npz",
                delete=False
            ) as tmp:
                tmp_name = tmp.name
            np.savez_compressed(
                tmp_name,
                ranked_idx=np.asarray(ranked_idx, dtype=int),
                importance_all=np.asarray(importance_all, dtype=float)
            )
            os.replace(tmp_name, cache_path)
        except Exception:
            if tmp_name and os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass

    @staticmethod
    def select_shared_bands_multitask(X_train, Y_train, max_features=None):
        if max_features is None:
            max_features = Config.SHARED_MAX_FEATURES

        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        cache_key = FeatureSelector._make_cache_key(X_train, Y_train)
        cache_path = FeatureSelector._cache_dir() / f"multitask_lasso_{cache_key}.npz"

        cached = FeatureSelector._load_cached_ranking(cache_path)
        if cached is not None:
            ranked_idx, importance_all = cached
            Logger.log(f"[FeatureSelector] cache hit: {cache_path.name}")
            return np.sort(ranked_idx[:max_features]), importance_all

        mtl = MultiTaskLassoCV(
            alphas=FeatureSelector.ALPHAS,
            cv=Config.INNER_SPLITS,
            random_state=Config.RANDOM_STATE,
            n_jobs=Config.SKLEARN_N_JOBS,
            max_iter=10000
        )
        Logger.log(f"[FeatureSelector] cache miss: fitting MultiTaskLassoCV ({cache_path.name})")
        mtl.fit(X_train, Y_train)

        coef_abs = np.abs(mtl.coef_)
        importance_all = coef_abs.sum(axis=0)
        ranked_idx = FeatureSelector._rank_shared_indices(coef_abs, importance_all)
        FeatureSelector._save_cached_ranking(cache_path, ranked_idx, importance_all)

        shared_idx = ranked_idx[:max_features]

        return np.sort(shared_idx), importance_all
