"""
live_dashboard.py
-----------------
Browser-based LIVE dashboard for DOT CV telemetry.

Receives the same UDP packet format as live_pose_gui.py and serves a web UI.

Run:
    python live_dashboard.py

Open in browser:
    http://127.0.0.1:8050

Packet format (must match main_withROI.py sender):
    < d   i    6f     6f     6f     f
      |   |    |      |      |      |
     t  frame pose1  pose2  chosen  yaw

Each pose block is:
    [nx, ny, nz, cx_mm, cy_mm, cz_mm]
"""

import time
import socket
import struct
import threading
import argparse
from collections import deque
from math import atan2, degrees, sqrt

from flask import Flask, jsonify, Response


# -----------------------------
# UDP packet format
# -----------------------------
GUI_PKT_FMT = "<di" + "6f" + "6f" + "6f" + "f"
GUI_PKT_SIZE = struct.calcsize(GUI_PKT_FMT)

POSES = ["Pose1", "Pose2", "Chosen"]


def yaw_abs_deg(nx, nz):
    ang = degrees(atan2(nx, nz)) + 180.0
    return ang % 360.0


class TelemetryStore:
    def __init__(self, max_points=3000):
        self.lock = threading.Lock()
        self.max_points = max_points

        self.frames = deque(maxlen=max_points)

        self.data = {
            "Pose1": {"Rz": deque(maxlen=max_points), "Rx": deque(maxlen=max_points), "Yaw": deque(maxlen=max_points)},
            "Pose2": {"Rz": deque(maxlen=max_points), "Rx": deque(maxlen=max_points), "Yaw": deque(maxlen=max_points)},
            "Chosen": {"Rz": deque(maxlen=max_points), "Rx": deque(maxlen=max_points), "Yaw": deque(maxlen=max_points)},
        }

        self.latest_pose = {"Pose1": None, "Pose2": None, "Chosen": None}
        self.last_rx_wall = None
        self.first_rx_wall = None
        self.last_t_s = None
        self.last_frame = None
        self.packet_count = 0
        self.packet_times = deque(maxlen=120)

    def clear(self):
        with self.lock:
            self.frames.clear()
            for p in POSES:
                for sig in ["Rz", "Rx", "Yaw"]:
                    self.data[p][sig].clear()
            self.latest_pose = {"Pose1": None, "Pose2": None, "Chosen": None}
            self.last_rx_wall = None
            self.first_rx_wall = None
            self.last_t_s = None
            self.last_frame = None
            self.packet_count = 0
            self.packet_times.clear()

    def push_packet(self, t_s, frame_idx, pose1, pose2, chosen, yaw_from_sender):
        now = time.time()
        with self.lock:
            if self.first_rx_wall is None:
                self.first_rx_wall = now
            self.last_rx_wall = now
            self.last_t_s = float(t_s)
            self.last_frame = int(frame_idx)
            self.packet_count += 1
            self.packet_times.append(now)

            self.latest_pose["Pose1"] = pose1
            self.latest_pose["Pose2"] = pose2
            self.latest_pose["Chosen"] = chosen

            def push_pose(name, block, yaw_deg):
                nx, ny, nz, cx, cy, cz = block
                self.data[name]["Rz"].append(float(cz))
                self.data[name]["Rx"].append(float(cx))
                self.data[name]["Yaw"].append(float(yaw_deg))

            def yaw_from_block(block):
                nx, ny, nz, cx, cy, cz = block
                return yaw_abs_deg(float(nx), float(nz))

            yaw1 = yaw_from_block(pose1)
            yaw2 = yaw_from_block(pose2)
            yawc = float(yaw_from_sender)

            self.frames.append(int(frame_idx))
            push_pose("Pose1", pose1, yaw1)
            push_pose("Pose2", pose2, yaw2)
            push_pose("Chosen", chosen, yawc)

    def _rate_hz(self):
        if len(self.packet_times) >= 2:
            dt = self.packet_times[-1] - self.packet_times[0]
            if dt > 1e-6:
                return (len(self.packet_times) - 1) / dt
        return 0.0

    def snapshot(self):
        with self.lock:
            age_ms = None
            if self.last_rx_wall is not None:
                age_ms = (time.time() - self.last_rx_wall) * 1000.0

            duration_s = None
            if self.first_rx_wall is not None:
                duration_s = time.time() - self.first_rx_wall

            d1 = None
            d2 = None
            chosen_txt = None

            if self.latest_pose["Pose1"] and self.latest_pose["Pose2"] and self.latest_pose["Chosen"]:
                def triplet(block):
                    nx, ny, nz, cx, cy, cz = block
                    yaw = yaw_abs_deg(float(nx), float(nz))
                    return [float(cx), float(cz), float(yaw)]

                p1 = triplet(self.latest_pose["Pose1"])
                p2 = triplet(self.latest_pose["Pose2"])
                ch = triplet(self.latest_pose["Chosen"])

                d1 = sqrt((p1[0] - ch[0])**2 + (p1[1] - ch[1])**2 + (p1[2] - ch[2])**2)
                d2 = sqrt((p2[0] - ch[0])**2 + (p2[1] - ch[1])**2 + (p2[2] - ch[2])**2)
                chosen_txt = "Pose1" if d1 <= d2 else "Pose2"

            return {
                "packet_count": self.packet_count,
                "rate_hz": self._rate_hz(),
                "age_ms": age_ms,
                "duration_s": duration_s,
                "last_t_s": self.last_t_s,
                "last_frame": self.last_frame,
                "latest_pose": self.latest_pose,
                "frames": list(self.frames),
                "data": {
                    p: {sig: list(self.data[p][sig]) for sig in ["Rz", "Rx", "Yaw"]}
                    for p in POSES
                },
                "d1": d1,
                "d2": d2,
                "chosen_txt": chosen_txt,
            }


def udp_listener(store: TelemetryStore, udp_host: str, udp_port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_host, udp_port))
    sock.settimeout(0.5)

    print(f"[UDP] Listening on {udp_host}:{udp_port}")

    while True:
        try:
            pkt, _addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except Exception as e:
            print("[UDP] Listener error:", e)
            continue

        if len(pkt) < GUI_PKT_SIZE:
            continue

        try:
            unpacked = struct.unpack(GUI_PKT_FMT, pkt[:GUI_PKT_SIZE])
        except Exception:
            continue

        t_s = unpacked[0]
        frame_idx = unpacked[1]

        off = 2
        pose1 = unpacked[off:off+6]
        off += 6
        pose2 = unpacked[off:off+6]
        off += 6
        chosen = unpacked[off:off+6]
        off += 6
        yaw_from_sender = unpacked[off]

        store.push_packet(t_s, frame_idx, pose1, pose2, chosen, yaw_from_sender)


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>DOT LIVE Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1"></script>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 18px;
      background: #fafafa;
      color: #222;
    }

    .container {
      max-width: 1100px;
      margin: 0 auto;
    }

    .topbar {
      display: flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }

    .info { font-weight: 600; }
    .key { font-weight: 600; }

    .row {
      display: flex;
      gap: 18px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }

    .card {
      background: white;
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 12px 14px;
      min-width: 180px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    .mono {
      font-family: Consolas, monospace;
      white-space: pre-line;
    }

    .chartbox {
      background: white;
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 12px;
      margin: 0 auto 14px auto;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      max-width: 900px;
    }

    canvas {
      width: 100% !important;
      height: 320px !important;
    }

    select, button {
      padding: 6px 10px;
      border-radius: 8px;
      border: 1px solid #bbb;
      background: white;
    }

    .small {
      font-size: 13px;
      color: #444;
    }
  </style>
</head>
<body>
<div class="container">

  <div class="topbar">
    <div class="info" id="info">Waiting for telemetry...</div>
    <label>Window:
      <select id="windowSel">
        <option value="5">Last 5s</option>
        <option value="10" selected>Last 10s</option>
        <option value="20">Last 20s</option>
        <option value="full">Full</option>
      </select>
    </label>
    <button onclick="clearData()">Clear</button>
    <div class="key">Blue=Pose1 | Orange=Pose2 | Green(thick)=Chosen</div>
  </div>

  <div class="row small">
    <div id="frameLbl">Frame: -</div>
    <div id="timeLbl">t = - s</div>
  </div>

  <div class="row">
    <div class="card">
      <div><b>Pose1</b></div>
      <div class="mono" id="pose1">Tx: -&#10;Ty: -&#10;Tz: -&#10;Rx: -&#10;Ry: -&#10;Rz: -</div>
    </div>
    <div class="card">
      <div><b>Pose2</b></div>
      <div class="mono" id="pose2">Tx: -&#10;Ty: -&#10;Tz: -&#10;Rx: -&#10;Ry: -&#10;Rz: -</div>
    </div>
    <div class="card">
      <div><b>Chosen</b></div>
      <div class="mono" id="chosen">Tx: -&#10;Ty: -&#10;Tz: -&#10;Rx: -&#10;Ry: -&#10;Rz: -</div>
    </div>
    <div class="card" style="min-width: 300px;">
      <div><b>Chosen</b></div>
      <div class="small" id="choiceLbl">-</div>
    </div>
  </div>

  <div class="chartbox"><canvas id="chartRz"></canvas></div>
  <div class="chartbox"><canvas id="chartRx"></canvas></div>
  <div class="chartbox"><canvas id="chartYaw"></canvas></div>

</div>

<script>
const COLORS = {
  Pose1: 'rgb(80,170,255)',
  Pose2: 'rgb(255,170,80)',
  Chosen: 'rgb(120,255,120)'
};

function makeChart(canvasId, yLabel) {
  return new Chart(document.getElementById(canvasId), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Pose1 (Blue)', data: [], borderColor: COLORS.Pose1, borderWidth: 2, pointRadius: 0, tension: 0 },
        { label: 'Pose2 (Orange)', data: [], borderColor: COLORS.Pose2, borderWidth: 2, pointRadius: 0, tension: 0 },
        { label: 'Chosen (Green)', data: [], borderColor: COLORS.Chosen, borderWidth: 3, pointRadius: 0, tension: 0 }
      ]
    },
    options: {
      responsive: true,
      animation: false,
      plugins: {
        legend: { display: true },
        annotation: {
          annotations: {
            cursor: {
              type: 'line',
              xMin: 0,
              xMax: 0,
              borderColor: 'rgb(180,180,180)',
              borderWidth: 2
            }
          }
        }
      },
      scales: {
        x: { title: { display: true, text: 'Frame' } },
        y: { title: { display: true, text: yLabel } }
      }
    }
  });
}

const chartRz = makeChart('chartRz', 'Depth Translation (mm)');
const chartRx = makeChart('chartRx', 'Lateral Translation (mm)');
const chartYaw = makeChart('chartYaw', 'Absolute Yaw (deg) = atan2(Tx, Tz)+180');

function fmtPose(block) {
  if (!block) return "Tx: -\nTy: -\nTz: -\nRx: -\nRy: -\nRz: -";
  const [nx, ny, nz, cx, cy, cz] = block;
  return `Tx: ${nx.toFixed(3)}\nTy: ${ny.toFixed(3)}\nTz: ${nz.toFixed(3)}\nRx: ${cx.toFixed(3)}\nRy: ${cy.toFixed(3)}\nRz: ${cz.toFixed(3)}`;
}

function sliceWindow(state) {
  const sel = document.getElementById('windowSel').value;
  const frames = state.frames || [];
  const n = frames.length;
  if (n === 0) return { idx0: 0, idx1: 0 };

  if (sel === 'full') return { idx0: 0, idx1: n };

  const secs = parseFloat(sel);
  const rate = state.rate_hz || 0;
  let pts = 20;
  if (rate > 0.1) pts = Math.max(20, Math.floor(rate * secs));
  pts = Math.min(pts, n);

  return { idx0: n - pts, idx1: n };
}

function updateCharts(state) {
  const { idx0, idx1 } = sliceWindow(state);
  const x = (state.frames || []).slice(idx0, idx1);
  const cur = x.length ? x[x.length - 1] : 0;

  function setChart(chart, sig) {
    chart.data.labels = x;
    chart.data.datasets[0].data = (state.data.Pose1[sig] || []).slice(idx0, idx1);
    chart.data.datasets[1].data = (state.data.Pose2[sig] || []).slice(idx0, idx1);
    chart.data.datasets[2].data = (state.data.Chosen[sig] || []).slice(idx0, idx1);
    chart.options.plugins.annotation.annotations.cursor.xMin = cur;
    chart.options.plugins.annotation.annotations.cursor.xMax = cur;
    chart.update();
  }

  setChart(chartRz, 'Rz');
  setChart(chartRx, 'Rx');
  setChart(chartYaw, 'Yaw');
}

async function poll() {
  try {
    const res = await fetch('/api/state');
    const state = await res.json();

    const age = state.age_ms == null ? '-' : `${Math.round(state.age_ms)} ms`;
    const dur = state.duration_s == null ? '-' : `${state.duration_s.toFixed(1)}s`;
    const rate = state.rate_hz == null ? '-' : `${state.rate_hz.toFixed(1)} Hz`;

    document.getElementById('info').textContent =
      `Live | Packets: ${state.packet_count} | Duration: ${dur} | Rate≈${rate} | Last age: ${age} | Window: ${document.getElementById('windowSel').selectedOptions[0].text}`;

    document.getElementById('frameLbl').textContent =
      state.last_frame == null ? 'Frame: -' : `Frame: ${state.last_frame}`;

    document.getElementById('timeLbl').textContent =
      state.last_t_s == null ? 't = - s' : `t = ${state.last_t_s.toFixed(3)} s`;

    document.getElementById('pose1').textContent = fmtPose(state.latest_pose.Pose1);
    document.getElementById('pose2').textContent = fmtPose(state.latest_pose.Pose2);
    document.getElementById('chosen').textContent = fmtPose(state.latest_pose.Chosen);

    if (state.d1 == null || state.d2 == null) {
      document.getElementById('choiceLbl').innerHTML = '-';
    } else {
      document.getElementById('choiceLbl').innerHTML =
        `<b>Chosen:</b> ${state.chosen_txt} (d1=${state.d1.toFixed(4)}, d2=${state.d2.toFixed(4)})<br>` +
        `<span class="small">d1 = ||[Pose1_Rx, Pose1_Rz, yaw1] - [Chosen_Rx, Chosen_Rz, yawC]||<br>` +
        `d2 = ||[Pose2_Rx, Pose2_Rz, yaw2] - [Chosen_Rx, Chosen_Rz, yawC]||</span>`;
    }

    updateCharts(state);
  } catch (err) {
    document.getElementById('info').textContent = 'Error fetching dashboard state.';
  }
}

async function clearData() {
  await fetch('/api/clear', { method: 'POST' });
}

setInterval(poll, 300);
poll();
</script>
</body>
</html>
"""


def create_app(store: TelemetryStore):
    app = Flask(__name__)

    @app.get("/")
    def index():
        return Response(HTML, mimetype="text/html")

    @app.get("/api/state")
    def api_state():
        return jsonify(store.snapshot())

    @app.post("/api/clear")
    def api_clear():
        store.clear()
        return jsonify({"ok": True})

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--udp-host", default="0.0.0.0", help="UDP bind host for telemetry receiver.")
    parser.add_argument("--udp-port", type=int, default=50006, help="UDP port for telemetry receiver.")
    parser.add_argument("--web-host", default="127.0.0.1", help="Web dashboard bind host.")
    parser.add_argument("--web-port", type=int, default=8050, help="Web dashboard port.")
    args = parser.parse_args()

    store = TelemetryStore(max_points=3000)

    th = threading.Thread(target=udp_listener, args=(store, args.udp_host, args.udp_port), daemon=True)
    th.start()

    app = create_app(store)

    print(f"[WEB] Open: http://{args.web_host}:{args.web_port}")
    app.run(host=args.web_host, port=args.web_port, debug=False, threaded=True)


if __name__ == "__main__":
    main()