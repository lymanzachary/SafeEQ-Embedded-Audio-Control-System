import numpy as np
import time
import math
from multiprocessing import shared_memory, resource_tracker
import threading

# ==============================
# SHARED MEMORY NAMES
# ==============================
SPL_SHM_NAME   = "Jetson_SPL_value"   # READ ONLY
BRAKE_SHM_NAME = "Jetson_Brake_dB"    # OWNER / WRITER

# ==============================
# SHM SIZES
# ==============================
SPL_SHM_SIZE   = 8   # float64
BRAKE_SHM_SIZE = 8   # float64

# ==============================
# LIMITER PARAMETERS
# ==============================
SPL_limit_dB = 70.0
BrakeMax_dB  = -25.0

attack_slow   = 1.0
attack_medium = 8.0
attack_fast   = 35.0
release_rate  = 6.0

alpha_up   = 0.35
alpha_down = 0.15

HYST_dB = 0.5

# ==============================
# INTERNAL STATE
# ==============================
_running = False
_thread  = None

spl_shm     = None
spl_value   = None
brake_shm   = None
brake_value = None

# ==============================
# HELPERS
# ==============================
def _is_finite(x: float) -> bool:
    return math.isfinite(x)

def _attach_spl():
    shm = shared_memory.SharedMemory(name=SPL_SHM_NAME)
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    arr = np.ndarray((1,), dtype=np.float64, buffer=shm.buf)
    return shm, arr

def _create_brake_shm_owner():
    try:
        shared_memory.SharedMemory(name=BRAKE_SHM_NAME).unlink()
    except FileNotFoundError:
        pass

    shm = shared_memory.SharedMemory(
        name=BRAKE_SHM_NAME,
        create=True,
        size=BRAKE_SHM_SIZE
    )

    # cosmetic; safe in master-managed lifecycle
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass

    arr = np.ndarray((1,), dtype=np.float64, buffer=shm.buf)
    arr[0] = 0.0
    return shm, arr

# ==============================
# WORKER LOOP
# ==============================
def _volume_control_loop():
    global _running
    global spl_shm, spl_value, brake_shm, brake_value

    # ---- wait for SPL ----
    while _running:
        try:
            spl_shm, spl_value = _attach_spl()
            break
        except FileNotFoundError:
            time.sleep(0.05)

    if not _running:
        return

    # ---- create brake SHM ----
    brake_shm, brake_value = _create_brake_shm_owner()

    Brake_dB = 0.0
    SPL_ctrl = None
    engaged  = False

    last_t = time.perf_counter()

    try:
        while _running:
            now_t = time.perf_counter()
            dt = now_t - last_t
            if dt <= 0:
                dt = 1e-3
            last_t = now_t

            SPL_now = float(spl_value[0])

            if not _is_finite(SPL_now):
                time.sleep(0.01)
                continue

            # Asymmetric smoothing
            if SPL_ctrl is None or not _is_finite(SPL_ctrl):
                SPL_ctrl = SPL_now
            else:
                a = alpha_up if SPL_now > SPL_ctrl else alpha_down
                SPL_ctrl = a * SPL_now + (1.0 - a) * SPL_ctrl

            # Hysteresis
            if not engaged:
                if SPL_ctrl > (SPL_limit_dB + HYST_dB):
                    engaged = True
            else:
                if SPL_ctrl < (SPL_limit_dB - HYST_dB):
                    engaged = False

            # Brake integration
            if engaged:
                excess = SPL_ctrl - SPL_limit_dB
                if excess <= 4.0:
                    rate = attack_slow
                elif excess <= 8.0:
                    rate = attack_medium
                else:
                    rate = attack_fast
                Brake_dB -= rate * dt
            else:
                Brake_dB += release_rate * dt

            # Clamp
            if Brake_dB > 0.0:
                Brake_dB = 0.0
            elif Brake_dB < BrakeMax_dB:
                Brake_dB = BrakeMax_dB

            # Publish brake
            brake_value[0] = Brake_dB

            time.sleep(0.001)

    finally:
        try:
            if spl_shm is not None:
                spl_shm.close()
        except Exception:
            pass

        try:
            if brake_shm is not None:
                brake_shm.close()
        except Exception:
            pass
        try:
            if brake_shm is not None:
                brake_shm.unlink()
        except FileNotFoundError:
            pass

# ==============================
# PUBLIC API (MASTER CALLS THESE)
# ==============================
def start_volume_control():
    global _running, _thread

    if _running:
        return

    _running = True
    _thread = threading.Thread(
        target=_volume_control_loop,
        daemon=True
    )
    _thread.start()

def stop_volume_control():
    global _running, _thread

    _running = False
    if _thread is not None:
        _thread.join(timeout=1.0)
        _thread = None
