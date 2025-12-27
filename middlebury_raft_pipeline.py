#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-only pipeline: RAFT-Stereo disparity (PyTorch/CUDA) -> 3D point cloud (OpenCV Q) -> clustering (Open3D) -> metrics (CSV/JSON).

Input layout (your current folders):
  data/
    Rocks1-2views/ (optional Middlebury pair: view1.png, view5.png)
    Rocks2-2views/ (optional)
    two_view_training/
        delivery_area_1l/ (ETH3D two-view: cameras.txt, images.txt, im0.png, im1.png or images/*)
        ...
Outputs:
  outputs/
    eth3d_<scene>.disp.png  (disparity visualization)
    eth3d_<scene>.ply       (3D point cloud)
    eth3d_<scene>.csv       (cluster sizes + d_eq)
    eth3d_<scene>.summary.json (quality metrics for quick judgment)

Prereqs on Windows:
  - PyTorch with CUDA (e.g., 2.x + cu12x). Install via the official selector.  # See refs in README.
  - Clone RAFT-Stereo to third_party/ and download a checkpoint (e.g., raftstereo-eth3d.pth).
  - This script calls RAFT's demo.py with --save_numpy and then loads the .npy disparity it wrote.

References:
  - RAFT-Stereo repo & README (flags: --restore_ckpt, -l, -r, --save_numpy, corr impl notes)
  - OpenCV reprojectImageTo3D + Q matrix (from stereoRectify)
  - Open3D DBSCAN clustering (eps, min_points)
"""
import os
import sys
import math
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2 as cv
import open3d as o3d
import pandas as pd

# =========== CONFIG (edit these two lines to your absolute paths) ===========
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
OUT_ROOT = PROJECT_ROOT / "outputs"
OUT_ROOT.mkdir(exist_ok=True)

# RAFT repo & checkpoint
RAFT_REPO = PROJECT_ROOT / "third_party" / "RAFT-Stereo"  # e.g. E:/Project/BinocularVision/third_party/RAFT-Stereo
RAFT_CKPT = RAFT_REPO / "models" / "raftstereo-eth3d.pth"  # or raftstereo-middlebury.pth / iraftstereo_rvc.pth
RAFT_PYTHON_EXE = sys.executable

# RAFT inference flags
RAFT_ITERS = 32
RAFT_MIXED_PRECISION = True
RAFT_CORR_IMPL = "reg"  # "reg" is无需扩展即可用；若你编译了 sampler/ 可改为 "reg_cuda"

# Clustering & downsample
DBSCAN_EPS = 0.02
DBSCAN_MINPOINTS = 80
VOXEL_DOWNSAMPLE = 0.005

# Dataset roots
ETH3D_ROOT = DATA_ROOT / "two_view_training"  # 你的 ETH3D two-view 场景都在这个目录的子文件夹里
MIDDLEBURY_SCENES = [
    {"name": "Rocks1-2views", "folder": DATA_ROOT / "Rocks1-2views"},
    {"name": "Rocks2-2views", "folder": DATA_ROOT / "Rocks2-2views"},
]
# When using Middlebury without true calibration, Q is a "visualization scale" (non-metric).
MIDDLEBURY_FX = None
MIDDLEBURY_CX = None
MIDDLEBURY_CY = None
MIDDLEBURY_BASELINE = 0.12
# ============================================================================


def _np_vis_norm(disp: np.ndarray) -> np.ndarray:
    valid = np.isfinite(disp) & (disp > 0)
    if np.any(valid):
        mn = float(np.min(disp[valid]))
        rng = float(np.ptp(disp[valid]))
        rng = rng if rng > 1e-12 else 1.0
        return (disp - mn) / (rng + 1e-6)
    return np.zeros_like(disp, dtype=np.float32)


def raft_available() -> bool:
    return RAFT_REPO.exists() and (RAFT_REPO / "demo.py").exists() and RAFT_CKPT.exists()


def _run_raft_and_grab_npy(left_path: Path, right_path: Path) -> np.ndarray:
    """Call RAFT-Stereo's demo.py (GPU) and load the .npy disparity it writes.
       We pass POSIX paths on Windows to avoid split('/') issues in demo.py.
       We search common output spots: repo root, repo/demo_output, CWD."""
    if not raft_available():
        raise RuntimeError(f"RAFT not ready.\nrepo={RAFT_REPO}\nckpt={RAFT_CKPT}")

    lfp = left_path.as_posix()
    rfp = right_path.as_posix()
    ckpt = RAFT_CKPT.as_posix()
    demo = (RAFT_REPO / "demo.py").as_posix()

    cmd = [
        str(RAFT_PYTHON_EXE),
        demo,
        "--restore_ckpt",
        ckpt,
        f"-l={lfp}",
        f"-r={rfp}",
        "--save_numpy",
        "--corr_implementation",
        RAFT_CORR_IMPL,
        "--valid_iters",
        str(RAFT_ITERS),
    ]
    if RAFT_MIXED_PRECISION:
        cmd.append("--mixed_precision")
    print("[RAFT] running:", " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=str(RAFT_REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        print("[RAFT] stderr (tail):", e.stderr.decode("utf-8", errors="ignore")[-800:])
        raise

    # RAFT demo 通常会以父目录名作为文件名保存 .npy（官方 README 支持 --save_numpy）。
    # 我们用“父文件夹名”作为关键词在常见目录里匹配；找不到就 fallback 到“最新修改”的 .npy。
    stem_hint = left_path.parent.name  # e.g., delivery_area_1l
    search_dirs = [
        RAFT_REPO,
        RAFT_REPO / "demo_output",  # 一些派生脚本/Colab会用这个目录
        Path.cwd(),
    ]
    candidates = []
    for d in search_dirs:
        if d.exists():
            candidates += list(d.glob("*.npy"))
    if not candidates:
        raise RuntimeError("RAFT finished but no .npy disparity was found. Check stderr above.")

    # 优先选文件名匹配“父目录名”的，再按修改时间排序
    def _score(p: Path):
        s = 1 if p.stem == stem_hint else 0
        return (s, p.stat().st_mtime)

    best = sorted(candidates, key=_score, reverse=True)[0]
    disp = np.load(best).astype(np.float32)
    # --- Sanity: fix sign if almost all disparities are <=0 ---
    pos = np.count_nonzero(disp > 0)
    neg = np.count_nonzero(disp < 0)
    if pos < 100 and neg > 1000:
        # very likely sign is flipped (right-to-left)
        disp = -disp

    # --- Invalidate non-positive & NaN values for later mask ---
    disp[~np.isfinite(disp)] = 0.0

    # 可把npy拷贝到输出目录方便留痕
    try:
        np.save((OUT_ROOT / f"{stem_hint}.raft.npy").as_posix(), disp)
    except Exception:
        pass
    print(f"[RAFT] took {time.time() - t0:.2f}s, loaded: {best.name}")
    return disp


# ---------- ETH3D: read cameras.txt / images.txt to build Q ----------
def parse_eth3d_cameras(cameras_txt: Path) -> Dict[int, Dict]:
    cams = {}
    with open(cameras_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            cam_id = int(toks[0])
            model = toks[1]
            width = int(toks[2])
            height = int(toks[3])
            params = list(map(float, toks[4:]))
            cams[cam_id] = dict(model=model, width=width, height=height, params=params)
    return cams


def _quat_to_R(qw, qx, qy, qz) -> np.ndarray:
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    R = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qz), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    return R


def parse_eth3d_images(images_txt: Path) -> List[Dict]:
    lines = [
        ln.strip()
        for ln in images_txt.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    imgs = []
    i = 0
    while i < len(lines):
        toks = lines[i].split()
        if len(toks) < 10:
            i += 1
            continue
        img_id = int(toks[0])
        qw, qx, qy, qz = map(float, toks[1:5])
        tx, ty, tz = map(float, toks[5:8])
        cam_id = int(toks[8])
        name = toks[9]
        R = _quat_to_R(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)
        C = -R.T @ t
        imgs.append(dict(id=img_id, R=R, t=t, C=C, cam_id=cam_id, name=name))
        i += 2 if (i + 1 < len(lines) and len(lines[i + 1].split()) > 10) else 1
    return imgs


def build_Q_from_eth3d(cams: Dict[int, Dict], imgL: Dict, imgR: Dict) -> np.ndarray:
    camL = cams[imgL["cam_id"]]
    model = camL["model"]
    params = camL["params"]
    if model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    elif model in ("SIMPLE_PINHOLE",):
        fx, cx, cy = params[0], params[1], params[2]
        fy = fx
    else:
        fx = fy = 1.2 * camL["width"]
        cx = camL["width"] / 2.0
        cy = camL["height"] / 2.0
    baseline = float(np.linalg.norm(imgR["C"] - imgL["C"])) or 0.12
    Q = np.array(
        [[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, fx], [0, 0, -1.0 / (-baseline), 0]],
        dtype=np.float32,
    )
    return Q


def resolve_eth3d_pair(scene_dir: Path, cams: Dict[int, Dict], imgs: List[Dict]) -> Tuple[Path, Path, np.ndarray]:
    im0 = scene_dir / "im0.png"
    im1 = scene_dir / "im1.png"
    if im0.exists() and im1.exists():
        def find_by_suffix(name):
            for it in imgs:
                if it["name"].endswith(name):
                    return it
            return imgs[0]

        Lrec = find_by_suffix("im0.png")
        Rrec = find_by_suffix("im1.png")
        return im0, im1, build_Q_from_eth3d(cams, Lrec, Rrec)

    if len(imgs) >= 2:
        Lrec, Rrec = imgs[0], imgs[1]
        left = scene_dir / Lrec["name"]
        right = scene_dir / Rrec["name"]
        if not left.exists() or not right.exists():
            left = scene_dir / "images" / Lrec["name"]
            right = scene_dir / "images" / Rrec["name"]
        if not left.exists() or not right.exists():
            raise FileNotFoundError(f"ETH3D missing images: {Lrec['name']}, {Rrec['name']}")
        return left, right, build_Q_from_eth3d(cams, Lrec, Rrec)

    raise RuntimeError("images.txt has fewer than 2 images.")


def make_Q_simple(h, w, fx=None, cx=None, cy=None, baseline=None):
    fx = fx if fx is not None else 1.2 * w
    cx = cx if cx is not None else w / 2.0
    cy = cy if cy is not None else h / 2.0
    baseline = baseline if baseline is not None else 0.12
    return np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, fx], [0, 0, -1.0 / (-baseline), 0]], dtype=np.float32)


def disp_to_pcd(disp: np.ndarray, Q: np.ndarray, left_rgb_path: Path) -> o3d.geometry.PointCloud:
    points3d = cv.reprojectImageTo3D(disp, Q)  # OpenCV official path: disparity + Q -> 3D
    mask = disp > 0
    xyz = points3d[mask]
    rgb_img = cv.imread(str(left_rgb_path), cv.IMREAD_COLOR)
    if rgb_img is None:
        dn = _np_vis_norm(disp)
        rgb = np.stack([dn, dn, dn], axis=-1)[mask]
    else:
        rgb = cv.cvtColor(rgb_img, cv.COLOR_BGR2RGB)[mask] / 255.0
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def cluster_and_measure(pcd: o3d.geometry.PointCloud, out_csv: Path) -> pd.DataFrame:
    if len(pcd.points) == 0:
        return pd.DataFrame(columns=["id", "lx", "ly", "lz", "d_eq"])
    if VOXEL_DOWNSAMPLE and VOXEL_DOWNSAMPLE > 0:
        pcd = pcd.voxel_down_sample(VOXEL_DOWNSAMPLE)
    labels = np.array(pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MINPOINTS))
    rows = []
    for k in np.unique(labels[labels >= 0]):
        idx = np.where(labels == k)[0]
        pts = np.asarray(pcd.points)[idx]
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts))
        size = aabb.get_max_bound() - aabb.get_min_bound()
        d_eq = float(np.cbrt(np.prod(size)))
        rows.append(dict(id=int(k), lx=float(size[0]), ly=float(size[1]), lz=float(size[2]), d_eq=d_eq))
    rows = []
    # ...填 rows...
    df = pd.DataFrame(rows, columns=["id", "lx", "ly", "lz", "d_eq"])
    if not df.empty:
        df = df.sort_values("d_eq", ascending=False)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def summarize_quality(scene_name: str, disp: np.ndarray, pcd: o3d.geometry.PointCloud, df: pd.DataFrame, out_json: Path):
    valid = (disp > 0) & np.isfinite(disp)
    valid_ratio = float(valid.mean())
    if len(pcd.points) > 0:
        pts = np.asarray(pcd.points)
        z = pts[:, 2]
        z = z[np.isfinite(z)]
        z_stats = dict(
            z_min=float(np.min(z)) if z.size else None,
            z_max=float(np.max(z)) if z.size else None,
            z_med=float(np.median(z)) if z.size else None,
        )
    else:
        z_stats = dict(z_min=None, z_max=None, z_med=None)
    psd = None
    if df is not None and "d_eq" in df and len(df) > 0:
        q = np.quantile(df["d_eq"].values, [0.1, 0.5, 0.8])
        psd = dict(D10=float(q[0]), D50=float(q[1]), D80=float(q[2]))
    summary = dict(scene=scene_name, valid_ratio=valid_ratio, points=int(len(pcd.points)), z=z_stats, psd=psd)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------- RUNNERS ----------------
def run_middlebury(scene: dict):
    left = scene["folder"] / "view1.png"
    right = scene["folder"] / "view5.png"
    name = f"middlebury_{scene['name']}"
    if not left.exists() or not right.exists():
        print(f"[skip] {name}: missing view1.png/view5.png in {scene['folder']}")
        return
    print(f"[Middlebury] {name} (RAFT GPU)")
    disp = _run_raft_and_grab_npy(left, right)
    disp_vis = _np_vis_norm(disp)
    out_prefix = OUT_ROOT / name
    cv.imwrite(str(out_prefix.with_suffix(".disp.png")), (disp_vis * 255).astype(np.uint8))
    h, w = cv.imread(str(left), cv.IMREAD_GRAYSCALE).shape[:2]
    Q = make_Q_simple(h, w, MIDDLEBURY_FX, MIDDLEBURY_CX, MIDDLEBURY_CY, MIDDLEBURY_BASELINE)
    pcd = disp_to_pcd(disp, Q, left)
    o3d.io.write_point_cloud(str(out_prefix.with_suffix(".ply")), pcd)
    df = cluster_and_measure(pcd, out_prefix.with_suffix(".csv"))
    summarize_quality(name, disp, pcd, df, out_prefix.with_suffix(".summary.json"))
    if df is not None and len(df) > 0:
        q = np.quantile(df["d_eq"].values, [0.1, 0.5, 0.8])
        print(f"  PSD: D10={q[0]:.4f}, D50={q[1]:.4f}, D80={q[2]:.4f}")


def run_eth3d_scene(scene_dir: Path):
    cams_txt = scene_dir / "cameras.txt"
    imgs_txt = scene_dir / "images.txt"
    if not cams_txt.exists() or not imgs_txt.exists():
        print(f"[skip] ETH3D {scene_dir.name}: missing cameras.txt/images.txt")
        return
    try:
        cams = parse_eth3d_cameras(cams_txt)
        imgs = parse_eth3d_images(imgs_txt)
        left, right, Q = resolve_eth3d_pair(scene_dir, cams, imgs)
    except Exception as e:
        print(f"[ETH3D] {scene_dir.name} parse failed: {e}")
        return

    print(f"[ETH3D] {scene_dir.name} (RAFT GPU) -> {left.name} / {right.name}")
    try:
        disp = _run_raft_and_grab_npy(left, right)
    except Exception as e:
        print(f"  disparity failed: {e}")
        return

    valid = np.isfinite(disp) & (disp > 0)
    print(f"[diag] disp>0 ratio = {valid.mean():.3f}, min={np.nanmin(disp):.3f}, max={np.nanmax(disp):.3f}")

    out_prefix = OUT_ROOT / f"eth3d_{scene_dir.name}"
    try:
        disp_vis = _np_vis_norm(disp)
        cv.imwrite(str(out_prefix.with_suffix(".disp.png")), (disp_vis * 255).astype(np.uint8))
    except Exception as e:
        print(f"  save disp failed: {e}")

    # disp 是 float32，先把>0的部分归一化
    valid = np.isfinite(disp) & (disp > 0)
    dn = np.zeros_like(disp, dtype=np.float32)
    if np.any(valid):
        mn, mx = float(disp[valid].min()), float(disp[valid].max())
        dn[valid] = (disp[valid] - mn) / max(mx - mn, 1e-6)
    vis8 = (dn * 255).astype(np.uint8)

    # 用 OpenCV 上色（JET 很显眼，但更建议 viridis: 用 matplotlib 画）
    heat = cv.applyColorMap(vis8, cv.COLORMAP_JET)

    cv.imwrite(str(out_prefix.with_suffix(".disp_heat.png")), heat)

    try:
        pcd = disp_to_pcd(disp, Q, left)
        o3d.io.write_point_cloud(str(out_prefix.with_suffix(".ply")), pcd)
    except Exception as e:
        print(f"  point cloud failed: {e}")
        return

    try:
        df = cluster_and_measure(pcd, out_prefix.with_suffix(".csv"))
        summarize_quality(f"eth3d_{scene_dir.name}", disp, pcd, df, out_prefix.with_suffix(".summary.json"))
        if df is not None and len(df) > 0:
            q = np.quantile(df["d_eq"].values, [0.1, 0.5, 0.8])
            print(f"  PSD: D10={q[0]:.4f}, D50={q[1]:.4f}, D80={q[2]:.4f}")
        else:
            print("  no clusters")
    except Exception as e:
        print(f"  clustering failed: {e}")


def run_eth3d_all(root_dir: Path):
    if not root_dir.exists():
        print(f"[skip] ETH3D root missing: {root_dir}")
        return
    scenes = [p for p in root_dir.iterdir() if p.is_dir()]
    if not scenes:
        print(f"[skip] ETH3D has no subfolders under: {root_dir}")
        return
    for sd in sorted(scenes, key=lambda p: p.name):
        run_eth3d_scene(sd)


if __name__ == "__main__":
    # Optional: Middlebury pairs if present
    for sc in MIDDLEBURY_SCENES:
        run_middlebury(sc)
    # ETH3D: all subfolders
    run_eth3d_all(ETH3D_ROOT)
    print(f"Outputs saved under: {OUT_ROOT}")
