import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# ---------- 路径 ----------
velodyne_dir = Path(r"G:\view_of_delft_PUBLIC\lidar\training\velodyne")
image_dir    = Path(r"G:\view_of_delft_PUBLIC\lidar\training\image_2")
out_dir      = image_dir.parent / "overlay_intensity"
out_dir.mkdir(exist_ok=True)

# ---------- 相机内参 ----------
K = np.array([
    [1495.468642,    0.0,         961.272442],
    [   0.0,       1495.468642,   624.89592],
    [   0.0,          0.0,           1.0    ]
], dtype=np.float64)

# ---------- 外参 ----------
ext_3x4 = np.array([
    [-0.0079802, -0.9998541,  0.0151049,  0.151],
    [ 0.118497 , -0.0159445, -0.9928264, -0.461],
    [ 0.9929224, -0.0061331,  0.1186069, -0.915]
], dtype=np.float64)
R = ext_3x4[:, :3]
t = ext_3x4[:, 3].reshape(3,1)

def invert_extrinsic(R, t):
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv

def read_kitti_bin(p: Path, stride=1):
    a = np.fromfile(str(p), dtype=np.float32)
    ncols = 4 if (a.size % 4 == 0) else 3
    pts = a.reshape(-1, ncols)
    xyz = pts[:, :3].astype(np.float64)
    inten = pts[:, 3] if ncols == 4 else np.ones(len(pts))
    return xyz[::stride], inten[::stride]

def sensor_to_cam(pts, R, t):
    return (R @ pts.T + t).T

def project_no_dist(X_cam, K):
    Z = X_cam[:, 2:3]
    uv = (K @ (X_cam / np.clip(Z, 1e-6, None)).T).T
    return uv[:, :2]

def count_valid(pts_cam, W, H):
    front = pts_cam[:, 2] > 0.1
    if not np.any(front): 
        return 0
    uv = project_no_dist(pts_cam[front], K)
    u, v = uv[:, 0], uv[:, 1]
    m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return int(m.sum())

# ---------- 自动判 T/T^-1 ----------
img_files = sorted(image_dir.glob("*.jpg"))
probe = None
for ip in img_files:
    bp = velodyne_dir / f"{ip.stem}.bin"
    if bp.exists():
        probe = (ip, bp)
        break
probe_img = np.array(Image.open(probe[0]).convert("RGB"))
H_probe, W_probe = probe_img.shape[:2]
probe_pts, _ = read_kitti_bin(probe[1], stride=5)

pts_cam_T = sensor_to_cam(probe_pts, R, t)
R_inv, t_inv = invert_extrinsic(R, t)
pts_cam_Tinv = sensor_to_cam(probe_pts, R_inv, t_inv)

valid_T = count_valid(pts_cam_T, W_probe, H_probe)
valid_Tinv = count_valid(pts_cam_Tinv, W_probe, H_probe)
if valid_Tinv > valid_T:
    R_use, t_use = R_inv, t_inv
    print("使用 T^-1 (逆外参)")
else:
    R_use, t_use = R, t
    print("使用 T (原外参)")

# ---------- 批处理 ----------
for img_path in img_files:
    stem = img_path.stem
    bin_path = velodyne_dir / f"{stem}.bin"
    if not bin_path.exists():
        continue

    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]

    pts, inten = read_kitti_bin(bin_path, stride=1)
    pts_cam = sensor_to_cam(pts, R_use, t_use)
    front = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[front]
    inten = inten[front]
    if pts_cam.size == 0:
        continue

    uv = project_no_dist(pts_cam, K)
    u, v = uv[:, 0], uv[:, 1]
    m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[m], v[m]
    d = pts_cam[m, 2]
    inten = inten[m]

    # 归一化强度到 [0,1]，避免不同帧范围不一致
    inten_norm = (inten - inten.min()) / (inten.ptp() + 1e-6)

    plt.figure(figsize=(W/100, H/100), dpi=100)
    plt.imshow(img)
    # 用强度上色，深度控制大小
    plt.scatter(u, v, s=np.clip(30.0/d, 1, 5), c=inten_norm, cmap='viridis', alpha=0.9)
    plt.axis('off')
    out = out_dir / f"{stem}_overlay.jpg"
    plt.savefig(out, bbox_inches='tight', pad_inches=0)
    plt.close()

print(f"完成：结果已保存到 {out_dir}")
