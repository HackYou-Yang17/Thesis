"""
CARMA — multicell render module (PARTICLE model, confocal-matched).
Composites per-cell intensity/junction maps onto the full canvas, passes them
through a simulated confocal pipeline (PSF blur + photon/readout noise + gamma),
crops to 1:1, and writes mp4 / snapshot panels. Reads physics state from
multicell_model_ai.py and per-cell maps from cell_render.record_state; no fibre
dynamics here.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter, binary_dilation

import bundle_model.cell_particle as cm
from bundle_model.cell_particle import P
from bundle_model.cell_render import record_state
from bundle_model.multicell_particle import (
    CANVAS_W, CANVAS_H, CROP_SIZE, N_CELLS,
)


# ── CONFOCAL RENDER CONSTANTS (tuned to match the real images) ──────────────────
# PSF_SIGMA was 1.4, with the comment "~0.2 um at 0.139 um/lu". UM_PER_LATTICE is 0.362, not
# 0.139, so at that scale 1.4 lu is 0.51 um -- roughly 2.5x an actual confocal lateral PSF, and
# enough to merge neighbouring fibres into one blob at the 6 lu mesh pitch. Reduced to 0.7 lu =
# 0.25 um, which resolves individual fibres. The physically-derived value for 0.2 um is 0.55 lu;
# 0.7 is used instead because below ~0.6 the discrete lattice starts showing as pixel speckle,
# which is a rasterisation artifact rather than anything optical. Override with CARMA_PSF_SIGMA.
PSF_SIGMA = float(os.environ.get("CARMA_PSF_SIGMA", 0.7))   # confocal PSF half-width (lu)
JUNCTION_BOOST = 2.2     # crossing pixels (H x V overlap) boosted -> bright mesh-junction foci
BACKGROUND_HAZE = 0.04   # diffuse intracellular background (unassembled G-actin)
WALL_BRIGHTNESS = 0.16   # faint cell-wall outline
NOISE_SIGMA = 0.032      # photon shot + readout noise, added after blur
GAMMA = 0.80             # detector non-linearity (<1 lifts mid-tones)


def _canvas_masks(mc):
    """Build (interior, wall) canvas masks from the cells' pasted masks; cached on mc.
    Render-only: interior -> background haze, wall -> faint cell outline."""
    cached = getattr(mc, "_render_masks", None)
    if cached is not None:
        return cached
    union = np.zeros((CANVAS_H, CANVAS_W), bool)
    for m in mc._cell_masks:
        union |= m
    inner = (np.roll(union, 1, 0) & np.roll(union, -1, 0)
             & np.roll(union, 1, 1) & np.roll(union, -1, 1))
    wall = binary_dilation(union & ~inner, iterations=1)
    mc._render_masks = (union, wall)
    return mc._render_masks


# ── COMPOSITING ─────────────────────────────────────────────────────────────────

def composite_intensity(mc):
    """Composite per-cell intensity + junction maps onto the full canvas (brighter pixel wins on overlap).
    Uses cell_render.record_state per cell, with P.mesh_noise set to that cell's axis so the
    renderer's is_primary() binning matches how the fibres were grown."""
    intens = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    junct = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    rows_c = np.repeat(np.arange(P.H), P.W)
    cols_c = np.tile(np.arange(P.W), P.H)
    for i, (sim, cx, cy) in enumerate(mc.cells):
        # FIXED (was `cm.P.mesh_noise = ...`): is_primary() reads P.mesh_axis, and Params has no
        # `mesh_noise` field, so the original line set a stray attribute and left every cell binned
        # H/V against whichever axis the last physics step happened to leave in P.
        cm.P.mesh_axis = mc.axes[i]       # keep is_primary() consistent with this cell's axis
        st = record_state(sim)
        i_cell = st["intensity"]
        j_cell = st["junction"]
        CX = cols_c - mc.hcx + cx
        CY = rows_c - mc.hcy + cy
        ok = (CX >= 0) & (CX < CANVAS_W) & (CY >= 0) & (CY < CANVAS_H)
        np.maximum.at(intens, (CY[ok], CX[ok]), i_cell[rows_c[ok], cols_c[ok]])
        np.maximum.at(junct, (CY[ok], CX[ok]), j_cell[rows_c[ok], cols_c[ok]])
    return intens, junct


def render_arrays(intens, junct, interior_mask, wall_mask=None,
                  crop_size=CROP_SIZE, rng_seed=None, crop_offset=(0, 0)):
    """Confocal pipeline: intensity + walls + junction boost + haze -> PSF blur -> noise -> gamma -> 1:1 crop.

    crop_offset is (dcol, drow) in lu from the canvas centre — the crop WINDOW moves, its SIZE does
    not. The blur/noise/gamma still run on the full canvas before the cut, so an off-centre crop is
    not missing any signal that bleeds in across its edges."""
    intens = intens.copy()
    intens += BACKGROUND_HAZE * interior_mask.astype(np.float32)
    if wall_mask is not None:
        intens = np.maximum(intens, WALL_BRIGHTNESS * wall_mask.astype(np.float32))
    junct_boost = np.clip(junct * JUNCTION_BOOST, 0.0, 1.0)
    intens = np.clip(intens + junct_boost * 0.6, 0.0, 1.5)
    blurred = gaussian_filter(intens, sigma=PSF_SIGMA)
    rng = np.random.default_rng(rng_seed if rng_seed is not None else 0)
    noisy = np.clip(blurred + rng.normal(0.0, NOISE_SIGMA, blurred.shape), 0.0, 1.0)
    corrected = np.power(noisy, GAMMA)
    h, w = corrected.shape
    half = crop_size // 2
    dc, dr = crop_offset
    r0 = int(np.clip(h // 2 - half + dr, 0, h - crop_size)); r1 = r0 + crop_size
    c0 = int(np.clip(w // 2 - half + dc, 0, w - crop_size)); c1 = c0 + crop_size
    return np.clip(corrected[r0:r1, c0:c1], 0.0, 1.0)


def render_frame(mc, frame_idx):
    """Callback for the driver loop: composite + render one cropped [0,1] frame.
    Crop size comes from CROP_SIZE as before; the crop PLACEMENT comes from the model side
    (mc.crop_offset), which is where the tiling geometry lives."""
    intens, junct = composite_intensity(mc)
    interior, wall = _canvas_masks(mc)
    return render_arrays(intens, junct, interior, wall_mask=wall,
                         crop_size=CROP_SIZE, rng_seed=frame_idx * 17 + 3,
                         crop_offset=getattr(mc, "crop_offset", (0, 0)))


# ── SAVING ──────────────────────────────────────────────────────────────────────

def save_animation(frames, path, fps=12):
    """Write pre-rendered confocal frames (uint8 list) to mp4."""
    import subprocess
    import shutil
    n = len(frames)
    if n == 0:
        print("No frames to render.")
        return
    h, w = frames[0].shape
    print(f"Encoding {n} confocal frames ({w}x{h})...")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "gray",
            "-r", str(fps), "-i", "pipe:0",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "fast", str(path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for img in frames:
            proc.stdin.write(img.tobytes())
        proc.stdin.close()
        proc.wait()
    else:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        im = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=255,
                       origin="upper", interpolation="nearest", aspect="equal")

        def upd(i):
            im.set_data(frames[i])
            return (im,)
        ani = animation.FuncAnimation(fig, upd, frames=n, interval=60, blit=True)
        writer = animation.FFMpegWriter(fps=fps, bitrate=4000,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"])
        ani.save(path, writer=writer, dpi=w // 6)
        plt.close(fig)
    print(f"animation -> {path}")


def save_snapshots(frames, t_frames, path, hpf_targets=(32, 35, 36, 40, 44, 48, 52), start_hpf=32.0):
    """Panel of pre-rendered confocal frames at times closest to the given hpf targets.
    Maps step -> hpf via P.dt (particle Params must define dt)."""
    n = len(frames)
    hpf = start_hpf + np.array([t_frames[i] * P.dt / 3600.0 for i in range(n)])
    fig, axes = plt.subplots(1, len(hpf_targets),
                             figsize=(4 * len(hpf_targets), 4.2), facecolor="black")
    if len(hpf_targets) == 1:
        axes = [axes]
    for ax, target in zip(axes, hpf_targets):
        idx = int(np.argmin(np.abs(hpf - target)))
        ax.imshow(frames[idx], cmap="gray", vmin=0, vmax=255,
                  origin="upper", interpolation="nearest", aspect="equal")
        ax.set_title(f"{target} hpf\n(model {hpf[idx]:.1f})",
                     color="white", fontsize=9, fontfamily="monospace")
        ax.axis("off")
    plt.tight_layout(pad=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"snapshots -> {path}")
