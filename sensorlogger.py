#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║           SensorLogger v1.0 — Multi-Channel DAQ              ║
║  Channels: Mic RMS · Cam Motion · Keystrokes · CPU · Net I/O ║
║  Pattern : Ring Buffer · 100ms sample · Anomaly Z-score      ║
╚══════════════════════════════════════════════════════════════╝
"""

import csv
import math
import sys
import threading
import time
import collections
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import pyaudio
import cv2
from pynput import keyboard as kb_listener
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
SAMPLE_RATE_HZ   = 10          # samples per second  (100 ms interval)
RING_BUFFER_SIZE = 200         # total samples retained per channel
ANOMALY_WINDOW   = 20          # recent samples used to build baseline
ANOMALY_Z_THRESH = 2.8         # z-score required to flag anomaly
SPARK_WIDTH      = 35          # sparkline character width in dashboard
DASHBOARD_FPS    = 4           # rich Live refresh rate

AUDIO_RATE  = 44100
AUDIO_CHUNK = 4096
CAM_INDEX   = 0                # try 1 if built-in cam is index 0

CSV_PATH = Path("sensor_log.csv")

# ══════════════════════════════════════════════════════════════
#  RING BUFFER
# ══════════════════════════════════════════════════════════════
class RingBuffer:
    """Thread-safe fixed-size ring buffer backed by collections.deque."""

    def __init__(self, maxlen: int):
        self._buf  = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, value: float) -> None:
        with self._lock:
            self._buf.append(value)

    def snapshot(self) -> list:
        with self._lock:
            return list(self._buf)

    def tail(self, n: int) -> list:
        with self._lock:
            d = list(self._buf)
            return d[-n:] if len(d) >= n else d

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


# ══════════════════════════════════════════════════════════════
#  SHARED STATE
# ══════════════════════════════════════════════════════════════
class SensorHub:
    """
    Central data store.  Background threads write live_* values;
    the main loop reads them, pushes to ring buffers, writes CSV.
    """

    # ── Channel definitions: (attr, label, color, vmin, vmax, unit) ──
    CHANNELS = [
        ("mic_rms",    "🎙  Mic RMS",        "cyan",    0,    32767, "amp"),
        ("cam_motion", "📷  Cam Motion",      "magenta", 0,    5.0,   "px/f"),
        ("key_cps",    "⌨  Keys/sec",        "yellow",  0,    15.0,  "cps"),
        ("cpu_pct",    "💻  CPU %",           "green",   0,    100.0, "%"),
        ("net_kbps",   "🌐  Net KB/s",        "blue",    0,    2000,  "KB/s"),
    ]

    def __init__(self):
        # Live values written by sensor threads
        self.live_mic  = 0.0
        self.live_cam  = 0.0
        self.live_key  = 0.0
        self.live_cpu  = 0.0
        self.live_net  = 0.0
        self.battery   = None   # None on desktops

        # Ring buffers (one per channel)
        self.mic_rms    = RingBuffer(RING_BUFFER_SIZE)
        self.cam_motion = RingBuffer(RING_BUFFER_SIZE)
        self.key_cps    = RingBuffer(RING_BUFFER_SIZE)
        self.cpu_pct    = RingBuffer(RING_BUFFER_SIZE)
        self.net_kbps   = RingBuffer(RING_BUFFER_SIZE)
        self.timestamps = RingBuffer(RING_BUFFER_SIZE)

        # Anomaly event log (newest last, max 8 shown)
        self.anomalies  = collections.deque(maxlen=8)

        # Keystroke timing
        self._key_times: collections.deque = collections.deque(maxlen=40)

        # Network baseline
        self._net_prev_time:  float = 0.0
        self._net_prev_bytes: int   = 0

    # ── helpers ───────────────────────────────────────────────
    def record_key(self):
        self._key_times.append(time.monotonic())

    def compute_key_cps(self, window_sec: float = 5.0) -> float:
        now    = time.monotonic()
        recent = [t for t in self._key_times if now - t < window_sec]
        return len(recent) / window_sec

    def compute_net_kbps(self) -> float:
        ctr   = psutil.net_io_counters()
        total = ctr.bytes_sent + ctr.bytes_recv
        now   = time.monotonic()

        if self._net_prev_bytes == 0:
            self._net_prev_bytes = total
            self._net_prev_time  = now
            return 0.0

        dt    = max(now - self._net_prev_time, 1e-6)
        rate  = (total - self._net_prev_bytes) / dt / 1024   # KB/s
        self._net_prev_bytes = total
        self._net_prev_time  = now
        return max(rate, 0.0)

    def buffer_for(self, attr: str) -> RingBuffer:
        return getattr(self, attr)


hub = SensorHub()


# ══════════════════════════════════════════════════════════════
#  SENSOR THREADS
# ══════════════════════════════════════════════════════════════

def _thread_mic():
    """Continuously capture mic audio and update hub.live_mic (RMS)."""
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK,
        )
        while True:
            try:
                raw     = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                hub.live_mic = math.sqrt(float(np.mean(samples ** 2)))
            except OSError:
                hub.live_mic = 0.0
    except Exception as exc:
        print(f"[Mic] init failed → {exc}", file=sys.stderr)
        hub.live_mic = 0.0
    finally:
        pa.terminate()


def _thread_cam():
    """Use Farneback optical flow to measure frame-to-frame motion magnitude."""
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[Cam] VideoCapture({CAM_INDEX}) failed — motion channel = 0", file=sys.stderr)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    ok, frame = cap.read()
    if not ok:
        cap.release()
        return
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
        hub.live_cam = mag
        prev_gray    = gray
        time.sleep(0.04)   # ~25 fps for flow; main loop samples at 10 Hz

    cap.release()


def _start_keyboard_listener():
    def on_press(key):
        hub.record_key()

    listener = kb_listener.Listener(on_press=on_press, daemon=True)
    listener.start()


# ══════════════════════════════════════════════════════════════
#  ANOMALY DETECTION  (z-score on recent window)
# ══════════════════════════════════════════════════════════════

def _check_anomaly(label: str, buf: RingBuffer, current: float) -> bool:
    recent = buf.tail(ANOMALY_WINDOW)
    if len(recent) < 8:
        return False
    mean = float(np.mean(recent))
    std  = float(np.std(recent))
    if std < 1e-6:
        return False
    z = abs(current - mean) / std
    if z > ANOMALY_Z_THRESH:
        ts = datetime.now().strftime("%H:%M:%S")
        hub.anomalies.append(
            f"[{ts}] [bold red]⚠[/]  {label}  "
            f"val=[yellow]{current:.2f}[/]  z=[red]{z:.1f}[/]"
        )
        return True
    return False


# ══════════════════════════════════════════════════════════════
#  SPARKLINE
# ══════════════════════════════════════════════════════════════
_BLOCKS = "▁▂▃▄▅▆▇█"

def _spark(values: list, width: int = SPARK_WIDTH) -> str:
    if not values:
        return "·" * width
    recent = values[-width:]
    lo, hi = min(recent), max(recent)
    span   = hi - lo or 1.0
    return "".join(_BLOCKS[int((v - lo) / span * 7)] for v in recent).rjust(width)


# ══════════════════════════════════════════════════════════════
#  DASHBOARD RENDERER
# ══════════════════════════════════════════════════════════════

def _render_dashboard() -> Group:
    now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S.%f")[:-3]

    # ── main channel table ────────────────────────────────────
    tbl = Table(
        title=f"[bold white]SensorLogger v1.0[/]  ·  [dim]{now_str}[/]",
        box=box.ROUNDED,
        header_style="bold white on dark_blue",
        show_lines=True,
        expand=True,
        min_width=90,
    )
    tbl.add_column("Channel",      style="bold", width=18, no_wrap=True)
    tbl.add_column("Value",        justify="right", width=11)
    tbl.add_column("Unit",         width=6,  justify="center")
    tbl.add_column(f"Sparkline (last {SPARK_WIDTH})", width=SPARK_WIDTH + 2)
    tbl.add_column("Bar  0 ────── max", width=22)
    tbl.add_column("Avg", justify="right", width=8)
    tbl.add_column("σ",   justify="right", width=8)

    anomaly_flags: dict[str, bool] = {}

    live_vals = {
        "mic_rms":    hub.live_mic,
        "cam_motion": hub.live_cam,
        "key_cps":    hub.live_key,
        "cpu_pct":    hub.live_cpu,
        "net_kbps":   hub.live_net,
    }

    for attr, label, color, vmin, vmax, unit in SensorHub.CHANNELS:
        buf     = hub.buffer_for(attr)
        current = live_vals[attr]
        history = buf.snapshot()
        anom    = _check_anomaly(label, buf, current)
        anomaly_flags[attr] = anom

        spark  = _spark(history)
        vals   = history or [0.0]
        avg    = float(np.mean(vals))
        sigma  = float(np.std(vals))

        # Scaled bar
        ratio   = min(1.0, max(0.0, (current - vmin) / (vmax - vmin + 1e-9)))
        bar_len = int(ratio * 18)
        bar_str = f"[{color}]{'█' * bar_len}[/][dim]{'░' * (18 - bar_len)}[/]"

        anom_style = "on dark_red" if anom else ""
        anom_icon  = " [blink][red]⚠[/][/]" if anom else ""

        tbl.add_row(
            f"[{color}]{label}[/]{anom_icon}",
            f"[bold {color}]{current:>9.2f}[/]",
            f"[dim]{unit}[/]",
            f"[{color}]{spark}[/]",
            bar_str,
            f"[dim]{avg:>6.1f}[/]",
            f"[dim]{sigma:>6.1f}[/]",
            style=anom_style,
        )

    # ── system info strip ─────────────────────────────────────
    batt   = psutil.sensors_battery()
    if batt:
        icon      = "🔋" if batt.power_plugged else "🪫"
        batt_text = f"{icon} {batt.percent:.0f}%"
    else:
        # Fallback: use CPU temp if available
        try:
            temps = psutil.sensors_temperatures()
            key   = next(iter(temps))
            t     = temps[key][0].current
            batt_text = f"🌡 {t:.0f}°C (no battery)"
        except Exception:
            batt_text = "N/A (desktop)"

    fills = "  ".join(
        f"[dim]{attr}=[/][cyan]{len(hub.buffer_for(attr))}[/]"
        for attr, *_ in SensorHub.CHANNELS
    )

    info = Panel(
        f"[bold]Battery/Temp:[/] {batt_text}   "
        f"[bold]Sample rate:[/] [cyan]{SAMPLE_RATE_HZ} Hz[/] ({1000//SAMPLE_RATE_HZ} ms)   "
        f"[bold]Ring buffer:[/] [cyan]{RING_BUFFER_SIZE}[/] samples/ch   "
        f"[bold]Anomaly z-thresh:[/] [red]{ANOMALY_Z_THRESH}[/]\n"
        f"[bold]Buffer fill:[/] {fills}\n"
        f"[bold]CSV:[/] [dim]{CSV_PATH.resolve()}[/]   "
        "[bold]Ctrl-C[/] to quit",
        title="[bold green]System[/]",
        border_style="green",
        padding=(0, 1),
    )

    # ── anomaly log ───────────────────────────────────────────
    anom_lines = (
        "\n".join(reversed(hub.anomalies))
        if hub.anomalies
        else "[dim]No anomalies detected yet — system nominal[/]"
    )
    anom_panel = Panel(
        anom_lines,
        title="[bold red]Anomaly Log[/] [dim](newest top)[/]",
        border_style="red",
        padding=(0, 1),
    )

    return Group(tbl, info, anom_panel)


# ══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════

def main():
    console = Console()
    console.print(
        "\n[bold cyan]SensorLogger v1.0[/] — starting sensor threads...\n"
        "[dim]Mic · Camera optical flow · Keyboard · CPU · Network I/O[/]\n"
    )

    # Spin up background sensor threads
    threading.Thread(target=_thread_mic, daemon=True, name="mic").start()
    threading.Thread(target=_thread_cam, daemon=True, name="cam").start()
    _start_keyboard_listener()

    time.sleep(1.2)   # allow threads to initialise

    # CSV setup
    csv_file   = CSV_PATH.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp_iso", "mic_rms", "cam_motion_px",
        "key_cps", "cpu_pct", "net_kbps",
    ])

    interval = 1.0 / SAMPLE_RATE_HZ
    console.print(f"[green]✓[/] Logging to [cyan]{CSV_PATH}[/] at [cyan]{SAMPLE_RATE_HZ} Hz[/]\n")

    try:
        with Live(
            _render_dashboard(),
            console=console,
            refresh_per_second=DASHBOARD_FPS,
            screen=False,
            vertical_overflow="visible",
        ) as live:
            while True:
                t0 = time.monotonic()

                # ── sample all five channels ──────────────────
                mic_val = hub.live_mic
                cam_val = hub.live_cam
                key_val = hub.compute_key_cps()
                cpu_val = psutil.cpu_percent(interval=None)
                net_val = hub.compute_net_kbps()

                # Update live display values
                hub.live_key = key_val
                hub.live_cpu = cpu_val
                hub.live_net = net_val

                ts = datetime.now().isoformat(timespec="milliseconds")

                # ── push to ring buffers ──────────────────────
                hub.timestamps.push(ts)
                hub.mic_rms.push(mic_val)
                hub.cam_motion.push(cam_val)
                hub.key_cps.push(key_val)
                hub.cpu_pct.push(cpu_val)
                hub.net_kbps.push(net_val)

                # ── write CSV ─────────────────────────────────
                csv_writer.writerow([
                    ts,
                    f"{mic_val:.2f}",
                    f"{cam_val:.4f}",
                    f"{key_val:.3f}",
                    f"{cpu_val:.1f}",
                    f"{net_val:.2f}",
                ])
                csv_file.flush()

                # ── refresh dashboard ─────────────────────────
                live.update(_render_dashboard())

                # ── sleep remainder of 100 ms slot ───────────
                elapsed = time.monotonic() - t0
                time.sleep(max(0.0, interval - elapsed))

    except KeyboardInterrupt:
        console.print(
            f"\n[bold yellow]SensorLogger stopped.[/]  "
            f"Data saved → [cyan]{CSV_PATH.resolve()}[/]\n"
            f"Samples logged: [cyan]{len(hub.timestamps)}[/]"
        )
    finally:
        csv_file.close()


if __name__ == "__main__":
    main()