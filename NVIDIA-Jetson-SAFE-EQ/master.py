import time
import signal
import sys
import subprocess
import os

import audio_ringbuffer_v1 as rb
import spl_reading as spl

_running = True
_uart_proc = None


def _stop_proc(p, name: str, timeout_s: float = 3.0):
    if p is None:
        return
    try:
        if p.poll() is None:
            print(f"Master: stopping {name}...")
            p.terminate()
            try:
                p.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                print(f"Master: {name} didn't exit, killing...")
                p.kill()
    except Exception:
        pass


def _shutdown(signum=None, frame=None):
    global _running, _uart_proc
    _running = False

    # Stop UART bridge first (it owns the serial port)
    _stop_proc(_uart_proc, "UART bridge")

    # Stop your existing services
    try:
        spl.stop_volume_reader()
        print("Master: shutdown volume reader.")
    except Exception:
        pass

    try:
        rb.stop_ringbuffer()
        print("Master: shutdown ring buffer.")
    except Exception:
        pass

    sys.exit(0)


def main():
    global _uart_proc

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start existing pipeline
    rb.start_ringbuffer()
    print("Master: ring buffer started.")
    spl.start_volume_reader()
    print("Master: volume reading started.")

    # Start UART bridge
    base_dir = os.path.dirname(os.path.abspath(__file__))
    uart_bridge = os.path.join(base_dir, "uart_bridge.py")

    try:
        if os.path.exists(uart_bridge):
            print(f"Master: starting UART bridge ({uart_bridge})")
            _uart_proc = subprocess.Popen([sys.executable, uart_bridge])
        else:
            print("Master: uart_bridge.py not found (skipping UART).")
    except Exception as e:
        print(f"Master: failed to start UART bridge: {e}")

    # Keep alive
    while _running:
        time.sleep(1)


if __name__ == "__main__":
    main()
