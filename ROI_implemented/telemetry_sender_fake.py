import time
import math
import socket
import struct
import argparse

# Must match live_pose_gui.py exactly
PKT_FMT = "<di" + "6f" + "6f" + "6f" + "f"
PKT_SIZE = struct.calcsize(PKT_FMT)


def yaw_abs_deg(nx, nz):
    """Absolute yaw (deg) from normal components."""
    ang = math.degrees(math.atan2(nx, nz)) + 180.0
    ang = ang % 360.0
    return float(ang)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="GUI host/IP to send to")
    parser.add_argument("--port", type=int, default=50006, help="GUI UDP port")
    parser.add_argument("--hz", type=float, default=10.0, help="Send rate (Hz)")
    parser.add_argument("--seconds", type=float, default=60.0, help="How long to run")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)

    start_wall = time.time()
    frame_idx = 0

    # Simulate “range closing” + “lateral oscillation” + “yaw sweep”
    # Units:
    # - center translation in mm (cx, cy, cz)
    # - normal components unitless (nx, ny, nz)

    while True:
        now = time.time()
        t_s = now - start_wall

        if t_s > args.seconds:
            break

        # ---- Simulated translation (mm) ----
        # cz: depth/range (e.g., start at 3000 mm, approach to 1200 mm)
        cz_mm = 3000.0 - 30.0 * t_s   # decreases over time
        cz_mm = max(cz_mm, 1200.0)

        # cx: lateral oscillation ±150 mm
        cx_mm = 150.0 * math.sin(2.0 * math.pi * 0.15 * t_s)

        # cy: small drift
        cy_mm = 25.0 * math.sin(2.0 * math.pi * 0.05 * t_s)

        # ---- Simulated normal vector (unitless) ----
        # Make nx vary so yaw changes; keep vector normalized
        yaw = 2.0 * math.pi * 0.10 * t_s  # radians
        nx = math.sin(yaw)
        nz = math.cos(yaw)
        ny = 0.05 * math.sin(2.0 * math.pi * 0.07 * t_s)

        # Normalize (just to be safe)
        norm = math.sqrt(nx*nx + ny*ny + nz*nz)
        nx, ny, nz = nx/norm, ny/norm, nz/norm

        yaw_deg = yaw_abs_deg(nx, nz)

        # ---- Pose1 & Pose2 candidates ----
        # Candidate poses are the same “shape” but add noise and slight bias
        def noisy_pose(nx, ny, nz, cx, cy, cz, noise_n=0.02, noise_c=20.0, sign=1.0):
            nnx = nx + sign * noise_n * (0.5 - math.sin(1.7 * t_s))
            nny = ny + sign * noise_n * (0.5 - math.cos(1.3 * t_s))
            nnz = nz + sign * noise_n * (0.5 - math.sin(1.1 * t_s))
            # renormalize
            nrm = math.sqrt(nnx*nnx + nny*nny + nnz*nnz)
            nnx, nny, nnz = nnx/nrm, nny/nrm, nnz/nrm

            ccx = cx + sign * noise_c * math.sin(0.9 * t_s)
            ccy = cy + sign * 0.6 * noise_c * math.cos(0.7 * t_s)
            ccz = cz + sign * 0.4 * noise_c * math.sin(0.5 * t_s)
            return (float(nnx), float(nny), float(nnz), float(ccx), float(ccy), float(ccz))

        pose1 = noisy_pose(nx, ny, nz, cx_mm, cy_mm, cz_mm, sign=+1.0)
        pose2 = noisy_pose(nx, ny, nz, cx_mm, cy_mm, cz_mm, sign=-1.0)

        # ---- Chosen pose ----
        # Simulate “chooser” picking the closer lateral magnitude candidate
        chosen = pose1 if abs(pose1[3]) <= abs(pose2[3]) else pose2

        pkt = struct.pack(
            PKT_FMT,
            float(t_s),
            int(frame_idx),
            *pose1,
            *pose2,
            *chosen,
            float(yaw_deg),
        )

        sock.sendto(pkt, addr)

        frame_idx += 1
        time.sleep(max(0.0, 1.0 / args.hz))

    sock.close()
    print(f"Sent {frame_idx} packets to {addr}.")


if __name__ == "__main__":
    main()