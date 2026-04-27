# SensorLogger v1.0
### Multi-Channel Software DAQ for Your Laptop

Treats your laptop as a **5-channel data acquisition board** — same architectural patterns as a hardware NI-DAQ card: shared timestamps, configurable sample rate, ring buffer storage, and multi-channel synchronous sampling.

---

## Channels

| # | Channel | Source | Unit | Notes |
|---|---------|--------|------|-------|
| 1 | 🎙 **Mic RMS** | PyAudio (44.1 kHz stream) | amplitude | Root-mean-square of 4096-sample frames |
| 2 | 📷 **Cam Motion** | OpenCV Farneback optical flow | px/frame | Mean flow-vector magnitude, 320×240 |
| 3 | ⌨ **Keys/sec** | pynput global listener | cps | 5-second sliding window |
| 4 | 💻 **CPU %** | psutil `cpu_percent` | % | Non-blocking, sampled every 100 ms |
| 5 | 🌐 **Net KB/s** | psutil `net_io_counters` | KB/s | Combined TX+RX delta — **the added 4th+ channel** |

### Why Network I/O as the 4th channel?
Network activity correlates with background processes invisible to CPU metrics (cloud syncs, streaming, update daemons). It surfaces anomalies that pure CPU monitoring misses — complementary to the other four channels.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    SensorHub                        │
│                                                     │
│  Mic thread ──► live_mic ─────────┐                 │
│  Cam thread ──► live_cam ─────────┤                 │
│  KB listener ──► _key_times ──────┤                 │
│                                   ▼                 │
│  Main loop (10 Hz / 100 ms) ──► RingBuffer ×5      │
│               │                   │                 │
│               ├──► CSV write      └──► Rich Live   │
│               └──► Anomaly z-score check            │
└─────────────────────────────────────────────────────┘
```

### Ring Buffer
`collections.deque(maxlen=N)` wrapped in a `threading.Lock`.  
Default: **200 samples** per channel (~20 seconds of history at 10 Hz).

### Sample Rate
`SAMPLE_RATE_HZ = 10` → 100 ms intervals.  
The main loop measures its own execution time and sleeps the remainder to stay on schedule.

### Anomaly Detection
Z-score over a **20-sample sliding window**. Threshold: **z > 2.8**.  
Each trigger appends to a fixed-length anomaly deque displayed in the dashboard.

---

## Installation

```bash
# 1. Create venv (recommended)
python3 -m venv .venv && source .venv/bin/activate

# 2. Install deps
pip install -r requirements.txt

# macOS — PyAudio also needs portaudio:
brew install portaudio && pip install pyaudio

# Ubuntu/Debian:
sudo apt install portaudio19-dev python3-dev && pip install pyaudio
```

### Accessibility permission (macOS)
`pynput` needs **Accessibility** permission in  
*System Settings → Privacy & Security → Accessibility*  
Add your terminal application.

---

## Running

```bash
python sensor_logger.py
```

Press **Ctrl-C** to stop.  Timestamped CSV is saved to `sensor_log.csv`.

---

## Configuration (top of `sensor_logger.py`)

| Constant | Default | Effect |
|----------|---------|--------|
| `SAMPLE_RATE_HZ` | `10` | Samples per second (100 ms) |
| `RING_BUFFER_SIZE` | `200` | Samples retained per channel |
| `ANOMALY_WINDOW` | `20` | Baseline window for z-score |
| `ANOMALY_Z_THRESH` | `2.8` | Spike sensitivity (lower = more sensitive) |
| `SPARK_WIDTH` | `35` | Sparkline width in characters |
| `CAM_INDEX` | `0` | OpenCV VideoCapture index |

---

## Tuning the Anomaly Threshold

The z-score threshold controls sensitivity:

| `ANOMALY_Z_THRESH` | Behaviour |
|-------------------|-----------|
| **2.0** | Very sensitive — triggers on normal variation |
| **2.8** | Balanced — flags genuine spikes |
| **3.5** | Conservative — only extreme outliers |

For noisy microphones or busy networks lower it to 2.0–2.5.  
For stable environments raise it to 3.0+ to reduce false positives.

---

## CSV Output Format

```
timestamp_iso,mic_rms,cam_motion_px,key_cps,cpu_pct,net_kbps
2025-03-15T14:22:01.100,312.44,0.0831,0.600,12.3,45.21
```

---

## v2.0 Bridge

The ring-buffer + shared-timestamp + configurable-sample-rate pattern maps directly onto a hardware DAQ build:

| Software (v1.0) | Hardware (v2.0) |
|-----------------|-----------------|
| `RingBuffer` deque | SD card sector ring |
| `SAMPLE_RATE_HZ` | DAQ timer interrupt frequency |
| Sensor threads | ADC/GPIO ISR handlers |
| psutil | Temperature / voltage sensors |
| Rich Live dashboard | OLED / serial plotter |

---

## Common Fixes

| Issue | Fix |
|-------|-----|
| Rich display flickers | Already uses `Live` context manager with `refresh_per_second=4` |
| Camera channel stays 0 | Try `CAM_INDEX = 1`; check `cv2.VideoCapture(1).isOpened()` |
| Battery shows N/A | Normal on desktops — falls back to CPU temperature via `psutil.sensors_temperatures()` |
| PyAudio import error | Install `portaudio` system library first (see Installation) |
| Keyboard listener needs permission | Grant Accessibility in macOS System Settings |
