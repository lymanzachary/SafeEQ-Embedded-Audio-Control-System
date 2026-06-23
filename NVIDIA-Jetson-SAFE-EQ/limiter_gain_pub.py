#!/usr/bin/env python3
import time, math, signal
import numpy as np
from multiprocessing import shared_memory, resource_tracker

SPL_SHM_NAME = "Jetson_SPL_value"      # existing producer
GAIN_SHM_NAME = "Jetson_Limiter_gain"  # NEW publisher

SPL_LIMIT_DB = 80.0
BRAKE_MIN_DB = -25.0   # max attenuation
HYST_DB = 0.5

attack_slow   = 1.0
attack_medium = 8.0
attack_fast   = 35.0
release_rate  = 6.0

alpha_up   = 0.35
alpha_down = 0.15

PRINT_HZ = 5

_running = True
def _stop(*_):
    global _running
    _running = False
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

def _attach_ro(name: str):
    shm = shared_memory.SharedMemory(name=name)
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    arr = np.ndarray((1,), dtype=np.float64, buffer=shm.buf)
    return shm, arr

def _create_rw_float(name: str, initial: float):
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=8)
        created = True
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name, create=False)
        created = False

    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass

    arr = np.ndarray((1,), dtype=np.float64, buffer=shm.buf)
    if created:
        arr[0] = float(initial)
    return shm, arr

def db_to_lin(db):
    return 10.0 ** (db / 20.0)

def main():
    print("[GAIN PUB] starting...")

    # attach SPL (wait until exists)
    spl_shm = spl_arr = None
    while _running:
        try:
            spl_shm, spl_arr = _attach_ro(SPL_SHM_NAME)
            break
        except FileNotFoundError:
            print("[GAIN PUB] waiting for Jetson_SPL_value...")
            time.sleep(0.1)

    if not _running:
        return

    gain_shm, gain_arr = _create_rw_float(GAIN_SHM_NAME, 1.0)
    print("[GAIN PUB] attached SPL; publishing Jetson_Limiter_gain")

    Brake_dB = 0.0
    SPL_ctrl = None
    engaged = False

    last_t = time.perf_counter()
    next_print = last_t

    try:
        while _running:
            now = time.perf_counter()
            dt = now - last_t
            if dt <= 0:
                dt = 1e-3
            last_t = now

            SPL_now = float(spl_arr[0])
            if not math.isfinite(SPL_now):
                time.sleep(0.01)
                continue

            # smooth SPL
            if SPL_ctrl is None or not math.isfinite(SPL_ctrl):
                SPL_ctrl = SPL_now
            else:
                a = alpha_up if SPL_now > SPL_ctrl else alpha_down
                SPL_ctrl = a * SPL_now + (1.0 - a) * SPL_ctrl

            # hysteresis engage/release
            if not engaged:
                if SPL_ctrl > (SPL_LIMIT_DB + HYST_DB):
                    engaged = True
            else:
                if SPL_ctrl < (SPL_LIMIT_DB - HYST_DB):
                    engaged = False

            # attack/release brake in dB
            if engaged:
                excess = SPL_ctrl - SPL_LIMIT_DB
                if excess <= 4.0:
                    rate = attack_slow
                elif excess <= 8.0:
                    rate = attack_medium
                else:
                    rate = attack_fast
                Brake_dB -= rate * dt
            else:
                Brake_dB += release_rate * dt

            if Brake_dB > 0.0:
                Brake_dB = 0.0
            elif Brake_dB < BRAKE_MIN_DB:
                Brake_dB = BRAKE_MIN_DB

            gain = db_to_lin(Brake_dB)  # 0.056..1.0
            gain_arr[0] = float(gain)

            if now >= next_print:
                print(f"[GAIN PUB] SPL={SPL_now:5.1f} ctrl={SPL_ctrl:5.1f} brake={Brake_dB:6.2f}dB gain={gain:0.3f} {'ENG' if engaged else 'REL'}")
                next_print = now + 1.0/PRINT_HZ

            time.sleep(0.001)

    finally:
        try: spl_shm.close()
        except Exception: pass
        try: gain_shm.close()
        except Exception: pass
        print("[GAIN PUB] stopped")

if __name__ == "__main__":
    main()
