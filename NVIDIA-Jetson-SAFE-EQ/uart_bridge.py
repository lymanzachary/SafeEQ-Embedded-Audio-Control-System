#!/usr/bin/env python3
import os
import time
import math
import re
import serial
import numpy as np
from multiprocessing import shared_memory, resource_tracker

# =========================
# UART CONFIG
# =========================
PORT = os.environ.get("JETSON_UART_PORT", "/dev/ttyTHS1")
BAUD = int(os.environ.get("JETSON_UART_BAUD", "115200"))
HZ   = float(os.environ.get("JETSON_UART_HZ", "20"))

# =========================
# SHARED MEMORY NAMES
# =========================
SPL_SHM_NAME      = "Jetson_SPL_value"
VOL_SHM_NAME      = "Jetson_user_volume"
THRESH_SHM_NAME   = "Jetson_safety_threshold"
LIM_GAIN_SHM_NAME = "Jetson_Limiter_gain"   # optional: published by a separate script
VOL_DB_SHM_NAME   = "Jetson_volume_db"      # Pi's volume_db setting
MAX_DB_SHM_NAME   = "Jetson_max_db"         # Pi's max_db setting

# =========================
# DEFAULTS
# =========================
DEFAULT_VOLUME = 1.0
DEFAULT_THRESH_DB = 80.0
DEFAULT_VOL_DB = -12.0
DEFAULT_MAX_DB = -6.0
BRAKE_MAX_DB = -25.0


def _unregister_resource(shm):
    # Prevent python from trying to “clean it up” on exit (consumer must not unlink)
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass


def attach_or_create(name, default):
    """Attach float64(1) shm; create if missing; ensure finite default."""
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError:
        shm = shared_memory.SharedMemory(name=name, create=True, size=8)

    _unregister_resource(shm)

    arr = np.ndarray((1,), dtype=np.float64, buffer=shm.buf)
    if not math.isfinite(float(arr[0])) or float(arr[0]) == 0.0:
        arr[0] = float(default)
    return shm, arr


def attach_if_exists(name):
    """Attach float64(1) shm if it exists; otherwise (None, None)."""
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return None, None

    _unregister_resource(shm)
    arr = np.ndarray((1,), dtype=np.float64, buffer=shm.buf)
    return shm, arr


def spl_to_gain(spl_db, thresh_db):
    """Fallback SPL limiter: returns linear gain in [10^(BRAKE_MAX_DB/20), 1]."""
    if not math.isfinite(spl_db):
        return 1.0
    if spl_db <= thresh_db:
        return 1.0

    over = spl_db - thresh_db
    brake_db = -min(over, abs(BRAKE_MAX_DB))
    return 10.0 ** (brake_db / 20.0)


def main():
    # Attach shared memory
    spl_shm, spl = attach_or_create(SPL_SHM_NAME, 0.0)
    vol_shm, vol = attach_or_create(VOL_SHM_NAME, DEFAULT_VOLUME)
    thr_shm, thr = attach_or_create(THRESH_SHM_NAME, DEFAULT_THRESH_DB)
    vol_db_shm, vol_db = attach_or_create(VOL_DB_SHM_NAME, DEFAULT_VOL_DB)
    max_db_shm, max_db = attach_or_create(MAX_DB_SHM_NAME, DEFAULT_MAX_DB)

    # Optional limiter gain published by a separate process
    lim_shm, lim = attach_if_exists(LIM_GAIN_SHM_NAME)

    ser = serial.Serial(PORT, BAUD, timeout=0.05)
    print(f"[UART BRIDGE] Running on {PORT} @ {BAUD} | HZ={HZ:g}")
    if lim is not None:
        print("[UART BRIDGE] Using Jetson_Limiter_gain when available")
    else:
        print("[UART BRIDGE] Jetson_Limiter_gain not found; using fallback SPL->gain")

    dt = 1.0 / HZ if HZ > 0 else 0.05
    last_gain = None
    last_spl = None

    try:
        while True:
            # ---------------- RX: Pi -> Jetson ----------------
            line = ser.readline()
            if line:
                msg = line.decode("utf-8", "ignore").strip()

                # Handle Pi's combined status message: 
                # "PI_OK RX_AGE=... GAIN=... VOL_DB=... MAX_DB=..."
                if msg.startswith("PI_OK"):
                    import re
                    
                    # Extract VOL_DB value
                    vol_db_match = re.search(r'VOL_DB=(-?\d+(?:\.\d+)?)', msg)
                    if vol_db_match:
                        try:
                            vol_db[0] = float(vol_db_match.group(1))
                        except Exception:
                            pass
                    
                    # Extract MAX_DB value
                    max_db_match = re.search(r'MAX_DB=(-?\d+(?:\.\d+)?)', msg)
                    if max_db_match:
                        try:
                            max_db[0] = float(max_db_match.group(1))
                        except Exception:
                            pass
                    
                    # Extract GAIN value
                    gain_match = re.search(r'GAIN=(-?\d+(?:\.\d+)?)', msg)
                    if gain_match:
                        try:
                            vol[0] = float(gain_match.group(1))
                        except Exception:
                            pass

                elif msg.startswith("VOL="):
                    try:
                        vol[0] = float(msg.split("=", 1)[1])
                    except Exception:
                        pass

                elif msg.startswith("THRESH="):
                    try:
                        thr[0] = float(msg.split("=", 1)[1])
                    except Exception:
                        pass

                elif msg.startswith("VOL_DB="):
                    try:
                        vol_db[0] = float(msg.split("=", 1)[1])
                    except Exception:
                        pass

                elif msg.startswith("MAX_DB="):
                    try:
                        max_db[0] = float(msg.split("=", 1)[1])
                    except Exception:
                        pass

            # ---------------- TX: Jetson -> Pi ----------------
            v = float(vol[0])
            if not math.isfinite(v) or v <= 0.0:
                v = DEFAULT_VOLUME

            # Prefer published limiter gain if present + finite
            if lim is not None:
                lg = float(lim[0])
            else:
                lg = float("nan")

            if math.isfinite(lg) and lg > 0.0:
                gain = lg * v
            else:
                gain = spl_to_gain(float(spl[0]), float(thr[0])) * v

            # Keep gain sane
            if not math.isfinite(gain):
                gain = 1.0
            if gain < 0.0:
                gain = 0.0
            elif gain > 2.0:
                gain = 2.0

            # Get current SPL
            spl_val = float(spl[0]) if spl is not None else 0.0
            
            # Send if gain OR SPL changed significantly
            if (last_gain is None or abs(gain - last_gain) > 0.001 or
                last_spl is None or abs(spl_val - last_spl) > 0.5):
               ser.write(f"GAIN={gain:.4f} SPL={spl_val:.1f}\n".encode("ascii"))
               last_gain = gain
               last_spl = spl_val

            time.sleep(dt)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            ser.close()
        except Exception:
            pass

        try:
            spl_shm.close()
        except Exception:
            pass
        try:
            vol_shm.close()
        except Exception:
            pass
        try:
            thr_shm.close()
        except Exception:
            pass
        try:
            vol_db_shm.close()
        except Exception:
            pass
        try:
            max_db_shm.close()
        except Exception:
            pass
        try:
            if lim_shm is not None:
                lim_shm.close()
        except Exception:
            pass

        print("[UART BRIDGE] Stopped")


if __name__ == "__main__":
    main()
