#!/usr/bin/env python3
"""ATE/RPE evaluation of a TUM-format trajectory against dataset groundtruth.

Usage: evaluate_ate.py <groundtruth.txt> <trajectory.txt> [--max-dt 0.02]

ATE: associates poses by timestamp, aligns with the closed-form rigid Umeyama
solution (no scale), reports translational RMSE/mean/median/max in meters.
RPE: per-frame relative pose error (translational drift per frame).
"""
import argparse
import sys

import numpy as np


def read_tum(path):
    poses = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            stamp = float(parts[0])
            tx, ty, tz, qx, qy, qz, qw = map(float, parts[1:])
            poses[stamp] = (np.array([tx, ty, tz]), np.array([qx, qy, qz, qw]))
    return poses


def quat_to_rot(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def associate(ground, estimate, max_dt):
    ground_stamps = np.array(sorted(ground))
    pairs = []
    for stamp in sorted(estimate):
        idx = np.searchsorted(ground_stamps, stamp)
        best = None
        for candidate in (idx - 1, idx):
            if 0 <= candidate < len(ground_stamps):
                dt = abs(ground_stamps[candidate] - stamp)
                if best is None or dt < best[0]:
                    best = (dt, ground_stamps[candidate])
        if best is not None and best[0] <= max_dt:
            pairs.append((best[1], stamp))
    return pairs


def umeyama_rigid(source, target):
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    cov = (target - mu_t).T @ (source - mu_s) / len(source)
    u, _, vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(u @ vt))
    fix = np.diag([1.0, 1.0, sign])
    rot = u @ fix @ vt
    trans = mu_t - rot @ mu_s
    return rot, trans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("groundtruth")
    parser.add_argument("trajectory")
    parser.add_argument("--max-dt", type=float, default=0.02)
    args = parser.parse_args()

    ground = read_tum(args.groundtruth)
    estimate = read_tum(args.trajectory)
    pairs = associate(ground, estimate, args.max_dt)
    if len(pairs) < 3:
        print(f"ERROR: only {len(pairs)} associations", file=sys.stderr)
        return 1

    gt_pos = np.array([ground[g][0] for g, _ in pairs])
    est_pos = np.array([estimate[e][0] for _, e in pairs])
    rot, trans = umeyama_rigid(est_pos, gt_pos)
    aligned = (rot @ est_pos.T).T + trans
    errors = np.linalg.norm(aligned - gt_pos, axis=1)

    print(f"pairs             {len(pairs)}")
    print(f"ate.rmse          {np.sqrt(np.mean(errors ** 2)) * 100:.3f} cm")
    print(f"ate.mean          {np.mean(errors) * 100:.3f} cm")
    print(f"ate.median        {np.median(errors) * 100:.3f} cm")
    print(f"ate.max           {np.max(errors) * 100:.3f} cm")

    # RPE: frame-to-frame translational drift on associated estimate frames.
    rel_errors = []
    for (g0, e0), (g1, e1) in zip(pairs, pairs[1:]):
        gt_delta = ground[g1][0] - ground[g0][0]
        est_delta = estimate[e1][0] - estimate[e0][0]
        # rotate estimate delta into the aligned frame
        rel_errors.append(np.linalg.norm(rot @ est_delta - gt_delta))
    rel = np.array(rel_errors)
    print(f"rpe.rmse          {np.sqrt(np.mean(rel ** 2)) * 1000:.3f} mm/frame")
    print(f"rpe.mean          {np.mean(rel) * 1000:.3f} mm/frame")
    return 0


if __name__ == "__main__":
    sys.exit(main())
