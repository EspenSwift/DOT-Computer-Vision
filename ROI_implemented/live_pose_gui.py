"""
live_pose_gui.py
----------------
LIVE pose visualization GUI for the DOT Capstone CV pipeline.

INPUT:
- UDP telemetry packets sent from main_withROI.py

This is the LIVE counterpart to csv_recorded_gui.py:
- Same look/feel: Pose1, Pose2, Chosen panels
- Same plot styling: Blue=Pose1, Orange=Pose2, Green(thick)=Chosen
- Same 3 main plots:
    1) Rz : Depth translation (mm)   [cz_mm]
    2) Rx : Lateral translation (mm) [cx_mm]
    3) Yaw_abs_deg = atan2(Tx, Tz)+180, derived from normal [nx, nz]
- Shows current frame and elapsed time
- Shows run duration (since first packet), packet rate, and last packet age
- Vertical cursor line at current frame on all plots
- NEW: Plot window dropdown (Last 5s / 10s / 20s / Full)

Packet format expected (little-endian):
    < d   i    6f     6f     6f     f
      |   |    |      |      |      |
     t  frame pose1  pose2  chosen  yaw

Where each pose block is:
    [nx, ny, nz, cx_mm, cy_mm, cz_mm]

NOTE:
- In your "mismatch" convention:
    Tx,Ty,Tz = normal components (unitless)
    Rx,Ry,Rz = translations/distances (mm)
- We display the panels in that convention:
    Tx..Tz = nx..nz, Rx..Rz = cx..cz
"""

import sys
import time
import socket
import struct
from collections import deque

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QComboBox
)
import pyqtgraph as pg


# -----------------------------
# UDP packet format (must match sender)
# -----------------------------
GUI_PKT_FMT = "<di" + "6f" + "6f" + "6f" + "f"
GUI_PKT_SIZE = struct.calcsize(GUI_PKT_FMT)

DEFAULT_BIND_HOST = "0.0.0.0"
DEFAULT_BIND_PORT = 50006

POSES = ["Pose1", "Pose2", "Chosen"]  # LIVE uses "Chosen" (instead of "True")


def yaw_abs_deg(nx, nz):
    """Yaw in degrees from normal components: atan2(nx, nz)+180 in [0,360)."""
    ang = np.degrees(np.arctan2(nx, nz)) + 180.0
    return float(ang % 360.0)


class LivePosePlayer(QMainWindow):
    def __init__(self, bind_host=DEFAULT_BIND_HOST, bind_port=DEFAULT_BIND_PORT):
        super().__init__()
        self.setWindowTitle("LIVE LAR Pose Viewer (Depth / Lateral / Yaw)")

        # -----------------------
        # UDP socket (non-blocking)
        # -----------------------
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((bind_host, bind_port))
        except Exception as e:
            QMessageBox.critical(self, "UDP Bind Error", f"Failed to bind UDP {bind_host}:{bind_port}\n\n{e}")
            raise
        self.sock.setblocking(False)

        # -----------------------
        # Live state / timing
        # -----------------------
        self.last_rx_wall = None     # wall-clock time of last packet
        self.first_rx_wall = None    # wall-clock time of first packet (for duration)
        self.last_t_s = None         # telemetry time (seconds since start) from packet
        self.last_frame = None       # current frame idx from packet
        self.packet_count = 0
        self.packet_times = deque(maxlen=120)  # for rate estimate (last ~12s at 10 Hz)

        # -----------------------
        # Plot window options (NEW)
        # -----------------------
        # None = Full buffer (max_points) shown
        self.window_seconds = 10.0
        self.min_window_points = 20  # fallback when rate estimate isn't ready yet

        # -----------------------
        # Data buffers (ring buffers)
        # Store per pose: arrays for each plotted signal
        # -----------------------
        self.max_points = 3000  # give headroom; window dropdown controls what is displayed
        self.frames = deque(maxlen=self.max_points)

        self.data = {
            "Pose1": {"Rz": deque(maxlen=self.max_points),
                      "Rx": deque(maxlen=self.max_points),
                      "Yaw": deque(maxlen=self.max_points)},
            "Pose2": {"Rz": deque(maxlen=self.max_points),
                      "Rx": deque(maxlen=self.max_points),
                      "Yaw": deque(maxlen=self.max_points)},
            "Chosen": {"Rz": deque(maxlen=self.max_points),
                       "Rx": deque(maxlen=self.max_points),
                       "Yaw": deque(maxlen=self.max_points)},
        }

        # Latest pose blocks for panels (nx,ny,nz,cx,cy,cz)
        self.latest_pose = {"Pose1": None, "Pose2": None, "Chosen": None}

        # -----------------------
        # UI timer (poll UDP + redraw)
        # -----------------------
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll_udp)
        self.timer.start(30)  # UI refresh rate (~33 Hz). Telemetry usually 5–10 Hz.

        # -----------------------
        # Build UI (match csv_recorded_gui.py layout)
        # -----------------------
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)

        # --- Top row: status/info + window dropdown + color key + clear ---
        top = QHBoxLayout()

        self.lbl_info = QLabel("Waiting for telemetry...  (no packets yet)")
        self.lbl_info.setMinimumWidth(620)

        self.window_combo = QComboBox()
        self.window_combo.addItems(["Last 5s", "Last 10s", "Last 20s", "Full"])
        self.window_combo.setCurrentText("Last 10s")
        self.window_combo.currentTextChanged.connect(self._set_window)

        self.lbl_key = QLabel("Blue=Pose1 | Orange=Pose2 | Green(thick)=Chosen")
        self.lbl_key.setStyleSheet("font-weight: 600;")

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._clear_buffers)

        top.addWidget(self.lbl_info)
        top.addSpacing(10)
        top.addWidget(QLabel("Window:"))
        top.addWidget(self.window_combo)
        top.addStretch(1)
        top.addWidget(self.lbl_key)
        top.addSpacing(10)
        top.addWidget(self.btn_clear)
        main.addLayout(top)

        # --- Second row: Frame + telemetry time ---
        row2 = QHBoxLayout()
        self.lbl_frame = QLabel("Frame: -")
        self.lbl_time = QLabel("t = - s")
        row2.addWidget(self.lbl_frame)
        row2.addSpacing(12)
        row2.addWidget(self.lbl_time)
        row2.addStretch(1)
        main.addLayout(row2)

        # --- Pose panels: Pose1 / Pose2 / Chosen ---
        pose_row = QHBoxLayout()
        self.pose_labels = {}

        for p in POSES:
            box = QVBoxLayout()
            title = QLabel(f"<b>{p}</b>")
            box.addWidget(title)

            # show as Tx..Tz and Rx..Rz to match your “mismatch convention”
            lbl = QLabel("Tx: -\nTy: -\nTz: -\nRx: -\nRy: -\nRz: -")
            lbl.setStyleSheet("font-family: Consolas, monospace;")
            box.addWidget(lbl)

            self.pose_labels[p] = lbl
            pose_row.addLayout(box)

        self.lbl_choice = QLabel(
            "<b>Chosen:</b> -<br>"
            "<span style='font-size:11px;'>"
            "d1 = ||[Pose1_Rx, Pose1_Rz, yaw1] - [Chosen_Rx, Chosen_Rz, yawC]||<br>"
            "d2 = ||[Pose2_Rx, Pose2_Rz, yaw2] - [Chosen_Rx, Chosen_Rz, yawC]||"
            "</span>"
        )
        pose_row.addWidget(self.lbl_choice)
        main.addLayout(pose_row)

        # --- Plots area (3 plots) ---
        self.plot_widget = pg.GraphicsLayoutWidget()
        main.addWidget(self.plot_widget, 1)

        self.plots = {}
        self.curves = {}
        self.cursors = {}

        self._setup_plots()

    # ------------------------------------------------------------
    # Plot window dropdown handler (NEW)
    # ------------------------------------------------------------
    def _set_window(self, text: str):
        if text == "Last 5s":
            self.window_seconds = 5.0
        elif text == "Last 10s":
            self.window_seconds = 10.0
        elif text == "Last 20s":
            self.window_seconds = 20.0
        elif text == "Full":
            self.window_seconds = None
        else:
            self.window_seconds = 10.0

        # Redraw immediately with the new window selection
        self._redraw()

    # ------------------------------------------------------------
    # Plot setup (same style as recorded GUI)
    # ------------------------------------------------------------
    def _setup_plots(self):
        self.plot_widget.clear()
        self.plots.clear()
        self.curves.clear()
        self.cursors.clear()

        pose_styles = {
            "Pose1": pg.mkPen(color=(80, 170, 255), width=2),   # blue
            "Pose2": pg.mkPen(color=(255, 170, 80), width=2),   # orange
            "Chosen": pg.mkPen(color=(120, 255, 120), width=3), # green thick
        }

        y_labels = {
            "Rz": "Depth Translation (mm)",
            "Rx": "Lateral Translation (mm)",
            "Yaw": "Absolute Yaw (deg) = atan2(Tx, Tz)+180",
        }

        signals = ["Rz", "Rx", "Yaw"]

        for r, sig in enumerate(signals):
            p = self.plot_widget.addPlot(row=r, col=0)
            p.showGrid(x=True, y=True)
            p.setLabel("left", y_labels[sig])
            p.setLabel("bottom", "Frame")
            self.plots[sig] = p

            self.curves[sig] = {}
            for pose_name in POSES:
                legend_name = {
                    "Pose1": "Pose1 (Blue)",
                    "Pose2": "Pose2 (Orange)",
                    "Chosen": "Chosen (Green)"
                }[pose_name]
                self.curves[sig][pose_name] = p.plot([], [], pen=pose_styles[pose_name], name=legend_name)

            cursor_pen = pg.mkPen(color=(180, 180, 180), width=2, style=Qt.DashLine)
            vline = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=cursor_pen)
            p.addItem(vline)
            self.cursors[sig] = vline

        # Legend only on first plot
        self.plots["Rz"].addLegend()

    # ------------------------------------------------------------
    # Buffer control
    # ------------------------------------------------------------
    def _clear_buffers(self):
        self.frames.clear()
        for p in POSES:
            for sig in ["Rz", "Rx", "Yaw"]:
                self.data[p][sig].clear()
        self.latest_pose = {"Pose1": None, "Pose2": None, "Chosen": None}
        self.last_frame = None
        self.last_t_s = None
        self.packet_count = 0
        self.packet_times.clear()
        self.first_rx_wall = None
        self.last_rx_wall = None
        self._redraw()

    # ------------------------------------------------------------
    # UDP poll loop
    # ------------------------------------------------------------
    def _poll_udp(self):
        got_any = False

        while True:
            try:
                pkt, _addr = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            except Exception:
                break

            if len(pkt) < GUI_PKT_SIZE:
                continue

            try:
                unpacked = struct.unpack(GUI_PKT_FMT, pkt[:GUI_PKT_SIZE])
            except Exception:
                continue

            got_any = True
            self.packet_count += 1
            now_wall = time.time()
            if self.first_rx_wall is None:
                self.first_rx_wall = now_wall
            self.last_rx_wall = now_wall
            self.packet_times.append(now_wall)

            # Parse packet fields
            t_s = float(unpacked[0])
            frame_idx = int(unpacked[1])

            off = 2
            pose1 = unpacked[off:off+6]; off += 6
            pose2 = unpacked[off:off+6]; off += 6
            chosen = unpacked[off:off+6]; off += 6
            yaw_from_sender = float(unpacked[off])

            self.last_t_s = t_s
            self.last_frame = frame_idx

            # Save latest pose blocks for panels
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
            yawc = yaw_from_sender

            self.frames.append(frame_idx)
            push_pose("Pose1", pose1, yaw1)
            push_pose("Pose2", pose2, yaw2)
            push_pose("Chosen", chosen, yawc)

        # Update UI elements and plots
        self._update_labels()
        if got_any:
            self._redraw()

    # ------------------------------------------------------------
    # UI updates
    # ------------------------------------------------------------
    def _update_labels(self):
        # Frame/time labels
        if self.last_frame is None:
            self.lbl_frame.setText("Frame: -")
        else:
            self.lbl_frame.setText(f"Frame: {self.last_frame}")

        if self.last_t_s is None:
            self.lbl_time.setText("t = - s")
        else:
            self.lbl_time.setText(f"t = {self.last_t_s:.3f} s")

        # Top info label: duration, pkt rate, last age
        if self.last_rx_wall is None:
            self.lbl_info.setText("Waiting for telemetry...  (no packets yet)")
        else:
            age = time.time() - self.last_rx_wall
            dur = 0.0 if self.first_rx_wall is None else (time.time() - self.first_rx_wall)

            rate = 0.0
            if len(self.packet_times) >= 2:
                dt = self.packet_times[-1] - self.packet_times[0]
                if dt > 1e-6:
                    rate = (len(self.packet_times) - 1) / dt

            window_txt = self.window_combo.currentText()
            self.lbl_info.setText(
                f"Live | Packets: {self.packet_count} | Duration: {dur:.1f}s | "
                f"Rate≈{rate:.1f} Hz | Last age: {age*1000:.0f} ms | Window: {window_txt}"
            )

        # Pose panels
        for p in POSES:
            block = self.latest_pose[p]
            if block is None:
                self.pose_labels[p].setText("Tx: -\nTy: -\nTz: -\nRx: -\nRy: -\nRz: -")
                continue

            nx, ny, nz, cx, cy, cz = block
            self.pose_labels[p].setText(
                f"Tx: {nx: .3f}\nTy: {ny: .3f}\nTz: {nz: .3f}\n"
                f"Rx: {cx: .3f}\nRy: {cy: .3f}\nRz: {cz: .3f}"
            )

        # d1/d2: compare Pose1 vs Pose2 distance to Chosen in [Rx, Rz, yaw]
        if self.latest_pose["Pose1"] is None or self.latest_pose["Pose2"] is None or self.latest_pose["Chosen"] is None:
            self.lbl_choice.setText("<b>Chosen:</b> -")
            return

        def triplet(block):
            nx, ny, nz, cx, cy, cz = block
            yaw = yaw_abs_deg(float(nx), float(nz))
            return np.array([float(cx), float(cz), float(yaw)], dtype=float)

        p1 = triplet(self.latest_pose["Pose1"])
        p2 = triplet(self.latest_pose["Pose2"])
        ch = triplet(self.latest_pose["Chosen"])

        d1 = np.linalg.norm(p1 - ch)
        d2 = np.linalg.norm(p2 - ch)
        chosen_txt = "Pose1" if d1 <= d2 else "Pose2"

        self.lbl_choice.setText(
            f"<b>Chosen:</b> {chosen_txt} (d1={d1:.4g}, d2={d2:.4g})<br>"
            "<span style='font-size:11px;'>"
            "d1 = ||[Pose1_Rx, Pose1_Rz, yaw1] - [Chosen_Rx, Chosen_Rz, yawC]||<br>"
            "d2 = ||[Pose2_Rx, Pose2_Rz, yaw2] - [Chosen_Rx, Chosen_Rz, yawC]||"
            "</span>"
        )

    # ------------------------------------------------------------
    # Plot redraw (UPDATED: windowed view)
    # ------------------------------------------------------------
    def _estimate_rate_hz(self) -> float:
        if len(self.packet_times) >= 2:
            dt = self.packet_times[-1] - self.packet_times[0]
            if dt > 1e-6:
                return (len(self.packet_times) - 1) / dt
        return 0.0

    def _redraw(self):
        if not self.frames:
            for sig in ["Rz", "Rx", "Yaw"]:
                for p in POSES:
                    self.curves[sig][p].setData([], [])
                self.cursors[sig].setPos(0)
            return

        # Decide how many points to show based on window_seconds (NEW)
        if self.window_seconds is None:
            window_points = len(self.frames)
        else:
            rate = self._estimate_rate_hz()
            if rate > 0.1:
                window_points = int(max(self.min_window_points, self.window_seconds * rate))
            else:
                window_points = self.min_window_points

            window_points = min(window_points, len(self.frames))

        # Slice most recent window
        frames_list = list(self.frames)
        x = np.array(frames_list[-window_points:], dtype=float)
        cur = x[-1]

        for sig in ["Rz", "Rx", "Yaw"]:
            for p in POSES:
                y = np.array(list(self.data[p][sig])[-window_points:], dtype=float)
                self.curves[sig][p].setData(x, y)

            # Cursor at current frame
            self.cursors[sig].setPos(cur)

            # X-range to match window
            if len(x) >= 2:
                self.plots[sig].setXRange(x[0], x[-1], padding=0)

    def closeEvent(self, event):
        try:
            self.sock.close()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = LivePosePlayer()
    w.resize(1220, 920)  # match your recorded GUI size
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()