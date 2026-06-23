import numpy as np
import time, struct
import sounddevice as sd
from multiprocessing import shared_memory

# ================= CONSTANTS (UNCHANGED) =================
SAMPLE_RATE = 48_000
CHANNELS = 2
BLOCK_FRAMES = 480
DTYPE_IN = "int16"
DTYPE_SHM = np.int16
BUFFER_SECONDS = 10.0
SHM_NAME = "Jetson_audio_ring"
DEVICE_NAME_HINT = "ESP"

# ================= HEADER (UNCHANGED) ====================
HEADER_FMT = '<qBxxxiiqq'
HEADER_SIZE = struct.calcsize(HEADER_FMT)
HEADER_PADDED_SIZE = 64

def pack_header(write_index, filled, sample_rate, channels, total_frames, timestamp_ns):
    return struct.pack(
        HEADER_FMT,
        write_index,
        filled,
        sample_rate,
        channels,
        total_frames,
        timestamp_ns
    )

def unpack_header(buf: bytes):
    return struct.unpack(HEADER_FMT, buf)

# ================= GLOBAL STATE (UNCHANGED) ===============
shm = None
audio_buf = None
stream = None

total_frames = int(BUFFER_SECONDS * SAMPLE_RATE)

_widx = 0
_filled = 0

# ================= WRITE + CALLBACK (UNCHANGED) ===========
def write_block(block: np.ndarray):
    global _widx, _filled

    frames = block.shape[0]
    end = _widx + frames

    if end <= total_frames:
        audio_buf[_widx:end, :] = block
    else:
        split = total_frames - _widx
        audio_buf[_widx:, :] = block[:split, :]
        audio_buf[:(end % total_frames), :] = block[split:, :]

    _widx = end % total_frames
    if (_filled == 0) and (_widx == 0):
        _filled = 1

    shm.buf[:HEADER_SIZE] = pack_header(
        write_index=_widx,
        filled=_filled,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        total_frames=total_frames,
        timestamp_ns=int(time.time_ns())
    )

def audio_callback(indata, frames, time_info, status):
    if indata.shape[1] != CHANNELS:
        return
    write_block(indata)

def pick_input_device_index(name_hint: str) -> int:
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0 and name_hint.lower() in d["name"].lower():
            return i
    raise RuntimeError(f"No input device matched '{name_hint}'")

# ================= START / STOP (THIS IS THE ONLY CHANGE) ==========

def start_ringbuffer():
    global shm, audio_buf, stream, _widx, _filled

    _widx = 0
    _filled = 0

    audio_bytes = total_frames * CHANNELS * np.dtype(DTYPE_SHM).itemsize
    total_bytes = HEADER_PADDED_SIZE + audio_bytes

    try:
        shared_memory.SharedMemory(name=SHM_NAME).unlink()
    except FileNotFoundError:
        pass

    shm = shared_memory.SharedMemory(
        name=SHM_NAME,
        create=True,
        size=total_bytes
    )

    audio_buf = np.ndarray(
        (total_frames, CHANNELS),
        dtype=DTYPE_SHM,
        buffer=shm.buf[HEADER_PADDED_SIZE:]
    )

    shm.buf[:HEADER_SIZE] = pack_header(
        write_index=0,
        filled=0,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        total_frames=total_frames,
        timestamp_ns=int(time.time_ns())
    )

    DEVICE_INDEX = pick_input_device_index(DEVICE_NAME_HINT)

    stream = sd.InputStream(
        device=DEVICE_INDEX,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_FRAMES,
        dtype=DTYPE_IN,
        callback=audio_callback,
    )

    stream.start()

def stop_ringbuffer():
    global stream, shm

    if stream:
        stream.stop()
        stream.close()
        stream = None

    if shm:
        # prevent resource_tracker double-unlink warnings
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass

        try:
            shm.close()
        except Exception:
            pass

        try:
            shm.unlink()
        except FileNotFoundError:
            pass

        shm = None

