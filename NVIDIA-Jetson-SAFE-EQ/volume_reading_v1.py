import numpy as np
import struct
import time
import math
import signal
from multiprocessing import shared_memory, resource_tracker

current_spl_db = None 

# ===== must match your writer =====
SHM_NAME = "Jetson_audio_ring"
HEADER_FMT = "<qBxxxiiqq"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
HEADER_PADDED_SIZE = 64
DTYPE = np.int16

# ===== reader settings =====
BLOCK_FRAMES = 480
EPS = 1e-12
ALPHA = 0.2  # smoothing 0..1

# ===== ICS-43434 SPL conversion =====
# Datasheet: Sensitivity = -26 dBFS @ 94 dB SPL (sine, PEAK dBFS)
SENS_DBFS_PEAK_AT_94 = -26.0
PEAK_TO_RMS_SINE_DB = 3.0103
SENS_DBFS_RMS_AT_94 = SENS_DBFS_PEAK_AT_94 - PEAK_TO_RMS_SINE_DB  # -29.0103
AIRPODS_OFFSET_DB = -25.0

_running = True

def _stop(*_):
    global _running
    _running = False

signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

def unpack_header(buf):
    return struct.unpack(HEADER_FMT, buf)

# ---- attach (READ ONLY) ----
shm = shared_memory.SharedMemory(name=SHM_NAME)

# IMPORTANT: prevent resource_tracker warnings in a reader process
# (reader should NOT unlink; unregister stops Python trying to clean it up)
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

# Preallocate (NO per-loop allocs)
block_i16 = np.empty((BLOCK_FRAMES, channels), dtype=np.int16)
block_f32 = np.empty((BLOCK_FRAMES, channels), dtype=np.float32)
mono = np.empty((BLOCK_FRAMES,), dtype=np.float32)

print(f"Attached: sample_rate={sample_rate} channels={channels} total_frames={total_frames}")

smoothed_dbfs = None  # initialize on first real block

try:
    while _running:
        write_index, filled, _, _, _, _ = unpack_header(shm.buf[:HEADER_SIZE])

        if filled == 0:
            time.sleep(0.05)
            continue

        start = (write_index - BLOCK_FRAMES) % total_frames

        # Copy newest block (wrap-safe, no vstack)
        if start + BLOCK_FRAMES <= total_frames:
            block_i16[:, :] = audio_buf[start:start + BLOCK_FRAMES, :]
        else:
            first = total_frames - start
            block_i16[:first, :] = audio_buf[start:, :]
            block_i16[first:, :] = audio_buf[:BLOCK_FRAMES - first, :]

        # int16 -> float32 [-1,1)
        block_f32[:, :] = block_i16[:, :].astype(np.float32) / 32768.0

        # DC remove per channel
        block_f32[:, :] -= np.mean(block_f32[:, :], axis=0, keepdims=True)

        # downmix to mono
        mono[:] = np.mean(block_f32[:, :], axis=1)

        # RMS -> dBFS(RMS)
        rms = math.sqrt(float(np.mean(mono * mono)))
        dbfs_rms = 20.0 * math.log10(rms + EPS)

        # Smooth
        if smoothed_dbfs is None:
            smoothed_dbfs = dbfs_rms
        else:
            smoothed_dbfs = (1.0 - ALPHA) * smoothed_dbfs + ALPHA * dbfs_rms

        # Convert dBFS(RMS) -> dB SPL using datasheet sensitivity
        spl = 94.0 + (smoothed_dbfs - SENS_DBFS_RMS_AT_94 + AIRPODS_OFFSET_DB)

        current_spl_db = spl
        #print(f"SPL={current_spl_db:7.2f} dB")

        time.sleep(0.05)

finally:
    shm.close()
