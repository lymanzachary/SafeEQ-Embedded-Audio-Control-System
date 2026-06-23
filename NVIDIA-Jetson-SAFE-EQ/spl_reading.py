import numpy as np
import struct
import time
import math
from multiprocessing import shared_memory, resource_tracker
import threading

# ==============================
# SPL SHARED MEMORY (OWNER)
# ==============================
SPL_SHM_NAME = "Jetson_SPL_value"
SPL_SHM_SIZE = 8  # float64

spl_shm = None
spl_value = None

# ==============================
# INTERNAL CONTROL
# ==============================
_running = False
_thread = None

# ===== must match your writer =====
AUDIO_SHM_NAME = "Jetson_audio_ring"
HEADER_FMT = "<qBxxxiiqq"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
HEADER_PADDED_SIZE = 64
DTYPE = np.int16

# ===== reader settings =====
BLOCK_FRAMES = 480
EPS = 1e-12
ALPHA = 0.2  # smoothing 0..1

# ===== ICS-43434 SPL conversion =====
SENS_DBFS_PEAK_AT_94 = -26.0
PEAK_TO_RMS_SINE_DB = 3.0103
SENS_DBFS_RMS_AT_94 = SENS_DBFS_PEAK_AT_94 - PEAK_TO_RMS_SINE_DB
AIRPODS_OFFSET_DB = -25.0


def unpack_header(buf):
    return struct.unpack(HEADER_FMT, buf)


def _create_spl_shm_owner():
    """Create SPL SHM exactly like ring buffer style: unlink old, create new, owner controls unlink."""
    global spl_shm, spl_value

    # Remove stale SHM if left behind
    try:
        shared_memory.SharedMemory(name=SPL_SHM_NAME).unlink()
    except FileNotFoundError:
        pass

    spl_shm = shared_memory.SharedMemory(name=SPL_SHM_NAME, create=True, size=SPL_SHM_SIZE)

    # IMPORTANT: prevent resource_tracker from trying to auto-clean this later
    # Owner will unlink explicitly in stop_volume_reader()
    try:
        resource_tracker.unregister(spl_shm._name, "shared_memory")
    except Exception:
        pass

    spl_value = np.ndarray((1,), dtype=np.float64, buffer=spl_shm.buf)
    spl_value[0] = -np.inf  # safe init


def _volume_loop():
    global _running

    # ---- attach audio ring buffer (READ ONLY) ----
    shm = shared_memory.SharedMemory(name=AUDIO_SHM_NAME)

    # Reader side: do NOT let resource_tracker manage the ring buffer
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass

    hdr = unpack_header(shm.buf[:HEADER_SIZE])
    write_index, filled, sample_rate, channels, total_frames, ts = hdr

    audio_buf = np.ndarray(
        (total_frames, channels),
        dtype=DTYPE,
        buffer=shm.buf[HEADER_PADDED_SIZE:]
    )

    # Preallocate
    block_i16 = np.empty((BLOCK_FRAMES, channels), dtype=np.int16)
    block_f32 = np.empty((BLOCK_FRAMES, channels), dtype=np.float32)
    mono = np.empty((BLOCK_FRAMES,), dtype=np.float32)

    smoothed_dbfs = None

    try:
        while _running:
            write_index, filled, _, _, _, _ = unpack_header(shm.buf[:HEADER_SIZE])

            if filled == 0:
                time.sleep(0.01)
                continue

            start = (write_index - BLOCK_FRAMES) % total_frames

            if start + BLOCK_FRAMES <= total_frames:
                block_i16[:, :] = audio_buf[start:start + BLOCK_FRAMES, :]
            else:
                first = total_frames - start
                block_i16[:first, :] = audio_buf[start:, :]
                block_i16[first:, :] = audio_buf[:BLOCK_FRAMES - first, :]

            block_f32[:, :] = block_i16[:, :].astype(np.float32) / 32768.0
            block_f32[:, :] -= np.mean(block_f32[:, :], axis=0, keepdims=True)
            mono[:] = np.mean(block_f32[:, :], axis=1)

            rms = math.sqrt(float(np.mean(mono * mono)))
            dbfs_rms = 20.0 * math.log10(rms + EPS)

            if smoothed_dbfs is None:
                smoothed_dbfs = dbfs_rms
            else:
                smoothed_dbfs = (1.0 - ALPHA) * smoothed_dbfs + ALPHA * dbfs_rms

            spl = 94.0 + (smoothed_dbfs - SENS_DBFS_RMS_AT_94 + AIRPODS_OFFSET_DB)

            # WRITE TO SPL SHM
            spl_value[0] = spl

            # run fast; your limiter dt is separate
            time.sleep(0.01)

    finally:
        shm.close()


def start_volume_reader():
    global _running, _thread

    if _running:
        return

    # Create SPL SHM once per master run (owner)
    _create_spl_shm_owner()

    _running = True
    _thread = threading.Thread(target=_volume_loop, daemon=True)
    _thread.start()


def stop_volume_reader():
    global _running, _thread, spl_shm, spl_value

    _running = False
    if _thread is not None:
        _thread.join(timeout=1.0)
        _thread = None

    # Owner cleanup: close + unlink (exactly like ring buffer)
    if spl_shm is not None:
        try:
            spl_shm.close()
        except Exception:
            pass
        try:
            spl_shm.unlink()
        except FileNotFoundError:
            pass
        spl_shm = None
        spl_value = None
