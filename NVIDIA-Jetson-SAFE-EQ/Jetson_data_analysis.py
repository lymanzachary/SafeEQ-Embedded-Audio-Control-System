import time
import os
from multiprocessing import shared_memory, resource_tracker
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================
# CONFIG
# ==============================
SPL_SHM_NAME = "Jetson_SPL_value"
SPL_SHM_SIZE = 8  # float64
VOL_DB_SHM_NAME = "Jetson_volume_db"
MAX_DB_SHM_NAME = "Jetson_max_db"

ANALYSIS_INTERVAL_SEC = 30
ANALYSIS_WINDOW_SEC = 10
SAMPLE_RATE_HZ = 20

# Fallback if Pi settings not available
DEFAULT_SPL_LIMIT_DB = 85.0
DEFAULT_VOL_DB = -12.0
DEFAULT_MAX_DB = -6.0

OUTPUT_DIR = "spl_graphs"
MAX_GRAPHS = 20

# ==============================
# SETUP
# ==============================
os.makedirs(OUTPUT_DIR, exist_ok=True)


def attach_spl_shm():
    shm = shared_memory.SharedMemory(name=SPL_SHM_NAME)

    # Reader must NOT let resource_tracker clean this up
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass

    spl_array = np.ndarray((1,), dtype=np.float64, buffer=shm.buf)
    return shm, spl_array


def attach_volume_shm():
    """Attach to volume_db and max_db from Pi. Returns (shm_list, vol_db, max_db) or ([], None, None) if unavailable."""
    shms = []
    vol_db = None
    max_db = None

    try:
        vol_db_shm = shared_memory.SharedMemory(name=VOL_DB_SHM_NAME, create=False)
        try:
            resource_tracker.unregister(vol_db_shm._name, "shared_memory")
        except Exception:
            pass
        vol_db = np.ndarray((1,), dtype=np.float64, buffer=vol_db_shm.buf)
        shms.append(vol_db_shm)
    except FileNotFoundError:
        pass

    try:
        max_db_shm = shared_memory.SharedMemory(name=MAX_DB_SHM_NAME, create=False)
        try:
            resource_tracker.unregister(max_db_shm._name, "shared_memory")
        except Exception:
            pass
        max_db = np.ndarray((1,), dtype=np.float64, buffer=max_db_shm.buf)
        shms.append(max_db_shm)
    except FileNotFoundError:
        pass

    return shms, vol_db, max_db


def get_pi_volume_limits():
    """Get current volume limits from Pi via shared memory. Returns (vol_db, max_db)."""
    _, vol_db, max_db = attach_volume_shm()
    
    vol_db_val = float(vol_db[0]) if vol_db is not None else DEFAULT_VOL_DB
    max_db_val = float(max_db[0]) if max_db is not None else DEFAULT_MAX_DB
    
    return vol_db_val, max_db_val


def prune_old_graphs():
    files = sorted(
        [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")],
        key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f))
    )

    while len(files) > MAX_GRAPHS:
        oldest = files.pop(0)
        os.remove(os.path.join(OUTPUT_DIR, oldest))


def collect_spl_data(spl_value):
    samples = int(ANALYSIS_WINDOW_SEC * SAMPLE_RATE_HZ)
    interval = 1.0 / SAMPLE_RATE_HZ

    spl_data = np.empty(samples, dtype=np.float32)
    time_axis = np.linspace(0, ANALYSIS_WINDOW_SEC, samples)

    for i in range(samples):
        spl_data[i] = spl_value[0]
        time.sleep(interval)

    return time_axis, spl_data


def save_graph(t, spl, vol_db=None, max_db=None):
    """Save SPL plot with Pi volume settings. If vol_db/max_db are None, uses defaults."""
    if vol_db is None:
        vol_db = DEFAULT_VOL_DB
    if max_db is None:
        max_db = DEFAULT_MAX_DB
    
    # Use max_db as the SPL limit for control line
    spl_limit = max_db

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"spl_{timestamp}.png"
    path = os.path.join(OUTPUT_DIR, filename)

    plt.figure(figsize=(10, 6))
    plt.plot(t, spl, label="Jetson_SPL_value", linewidth=2)
    plt.axhline(
        spl_limit,
        linestyle="dotted",
        linewidth=2,
        label=f"Max SPL ({spl_limit:.1f} dB)"
    )
    plt.axhline(
        vol_db,
        linestyle="dashed",
        linewidth=1.5,
        label=f"Volume dB ({vol_db:.1f} dB)"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("dBSPL")
    plt.title(f"SPL Validation (Volume: {vol_db:.1f}dB, Limit: {spl_limit:.1f}dB)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    prune_old_graphs()


def main():
    print("[SPL ANALYSIS] Starting headless analysis loop")

    shm, spl_value = attach_spl_shm()

    try:
        while True:
            loop_start = time.time()

            # Get current Pi volume settings
            vol_db, max_db = get_pi_volume_limits()

            t, spl = collect_spl_data(spl_value)
            save_graph(t, spl, vol_db=vol_db, max_db=max_db)

            print(f"[SPL ANALYSIS] Graph saved. Vol={vol_db:.1f}dB, Max={max_db:.1f}dB")

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, ANALYSIS_INTERVAL_SEC - elapsed)
            time.sleep(sleep_time)

    finally:
        shm.close()


if __name__ == "__main__":
    main()
