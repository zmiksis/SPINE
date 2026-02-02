import warnings
from threadpoolctl import threadpool_limits, threadpool_info
import multiprocessing as mp

_requested_limit = None
_applied = False

def limit_to_one_thread():
    global _requested_limit
    _requested_limit = 1

def set_num_threads(n: int):
    global _requested_limit
    _requested_limit = n

def apply_thread_limits_if_needed():
    global _requested_limit, _applied

    if _applied or _requested_limit is None:
        return

    threadpool_limits(limits=_requested_limit)
    _applied = True

    if mp.current_process().name == "MainProcess":
        warnings.warn(f"[pyCalSim] Limiting BLAS threads to {_requested_limit}")

def detect_oversubscription(verbose=True) -> bool:
    info = threadpool_info()
    for lib in info:
        if lib.get("num_threads", 1) > 1:
            if verbose:
                warnings.warn(
                    f"[pyCalSim] Detected BLAS using {lib['num_threads']} threads "
                    "with multiprocessing. Consider `limit_to_one_thread()`."
                )
            return True
    return False
