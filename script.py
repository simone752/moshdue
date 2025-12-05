#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# HYPER-LUDOVICO: FAST ABSTRACT ENGINE (ENHANCED)
# -----------------------------------------------------------------------------
# Speed: ~10x faster (Processes at 240p, Upscales to 720p)
# Style: Abstract, melting P-frames, blocky glitches, color drifts, long smears.
# -----------------------------------------------------------------------------

import os
import sys
import shutil
import subprocess
import random
import argparse
import tempfile
import time
import numpy as np
import cv2

# -------------------- BASE CONFIG --------------------
# These are "sane defaults". You can override most of them via CLI.
BASE_SETTINGS = {
    "fps": 24,
    "internal_res": (426, 240),   # 240p = Blocky & Insanely Fast

    # Feedback / drag physics
    "threshold": 32,              # Higher = more areas keep the ghost trail
    "decay": 0.992,               # 0.97–0.999 (higher = longer trails)
    "zoom_drift": 1.012,          # Base zoom (melting forward movement)
    "zoom_jitter": 0.015,         # Frame-to-frame zoom randomness
    "rotate_range": 1.5,          # +/- degrees of rotation for feedback
    "smear_passes": 2,            # How many times we warp the ghost per frame

    # Block glitch parameters
    "block_prob": 0.08,           # Probability of a block-glitch event
    "block_iterations": 3,        # How many blocks to scramble when triggered
    "block_size_min": 16,
    "block_size_max": 64,

    # Color drift / jitter
    "color_jitter_base": 6,       # Base channel offset in pixels
}

# -------------------- AUDIO FX --------------------
def get_abrasive_audio_filter():
    """Generates a random, fast-rendering audio chain."""
    chains = [
        # Bitcrush
        "acrusher=bits=4:mode=log:aa=1, volume=1.5",
        # Haunted Reverb
        "aecho=0.8:0.9:500:0.3, lowpass=f=800",
        # Broken Radio
        "highpass=f=300, lowpass=f=3000, vibrato=f=10:d=0.5",
        # Deep Fried
        "treble=g=10, bass=g=10, acrusher=level_in=1:level_out=1:bits=8:mode=log:aa=1",
    ]
    return random.choice(chains)

# -------------------- FAST VISUAL FX --------------------

def fast_color_jitter(img, base_intensity=6, chaos_mult=1.0):
    """
    Drifts RGB channels using array slicing (Instant).
    Now stronger & chaos-aware.
    """
    h, w, c = img.shape
    intensity = max(1, int(base_intensity * chaos_mult))
    dx_b = random.randint(-intensity, intensity)
    dy_b = random.randint(-intensity, intensity)
    dx_r = random.randint(-intensity, intensity)
    dy_r = random.randint(-intensity, intensity)

    out = img.copy()

    # Blue channel drift
    out[:, :, 0] = np.roll(out[:, :, 0], dx_b, axis=1)
    out[:, :, 0] = np.roll(out[:, :, 0], dy_b, axis=0)

    # Red channel drift (slightly different vector)
    out[:, :, 2] = np.roll(out[:, :, 2], dx_r, axis=1)
    out[:, :, 2] = np.roll(out[:, :, 2], dy_r, axis=0)

    return out


def fast_block_shuffle(img, prob=0.08, iterations=3, size_min=16, size_max=64):
    """
    Blocky glitch: pick some rectangular blocks and shuffle them around.
    Cheap and very blocky, tuned for 240p.
    """
    if random.random() > prob:
        return img

    h, w, c = img.shape
    out = img.copy()

    for _ in range(iterations):
        bh = random.randint(size_min, size_max)
        bw = random.randint(size_min, size_max)
        y1 = random.randint(0, max(0, h - bh))
        x1 = random.randint(0, max(0, w - bw))

        y2 = random.randint(0, max(0, h - bh))
        x2 = random.randint(0, max(0, w - bw))

        block1 = out[y1:y1+bh, x1:x1+bw].copy()
        block2 = out[y2:y2+bh, x2:x2+bw].copy()
        out[y1:y1+bh, x1:x1+bw] = block2
        out[y2:y2+bh, x2:x2+bw] = block1

    return out


def apply_feedback_warp(prev_img, zoom, rotate, smear_passes=1):
    """
    The Core "Melt" Mechanic.
    Instead of flow, we just slightly zoom/rotate the previous frame.
    Now supports multiple smear passes for stronger dragging.
    """
    h, w = prev_img.shape[:2]
    center = (w // 2, h // 2)
    ghost = prev_img

    for _ in range(max(1, smear_passes)):
        M = cv2.getRotationMatrix2D(center, rotate, zoom)
        ghost = cv2.warpAffine(ghost, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return ghost

# -------------------- PROCESSOR --------------------

def process_video(
    v1_path,
    v2_path,
    out_path,
    drag_mult=1.0,
    chaos_mult=1.0,
    # Optional detailed overrides (all may be None)
    fps=None,
    internal_res=None,
    threshold=None,
    decay=None,
    zoom=None,
    zoom_jitter=None,
    rotate_range=None,
    smear_passes=None,
    block_prob=None,
    block_iterations=None,
    block_size_min=None,
    block_size_max=None,
    color_jitter_base=None,
    seed=-1,
    crf=22,
):
    # 1. RANDOMIZE SESSION / SEED
    if seed < 0:
        seed = random.randint(0, 999999)
    random.seed(seed)
    np.random.seed(seed)
    print(f">>> SEED: {seed}")

    # Build settings from base + overrides
    s = BASE_SETTINGS.copy()

    if fps is not None:
        s["fps"] = int(fps)
    if internal_res is not None:
        s["internal_res"] = internal_res
    if threshold is not None:
        s["threshold"] = float(threshold)
    if decay is not None:
        s["decay"] = float(decay)
    if zoom is not None:
        s["zoom_drift"] = float(zoom)
    if zoom_jitter is not None:
        s["zoom_jitter"] = float(zoom_jitter)
    if rotate_range is not None:
        s["rotate_range"] = float(rotate_range)
    if smear_passes is not None:
        s["smear_passes"] = int(smear_passes)
    if block_prob is not None:
        s["block_prob"] = float(block_prob)
    if block_iterations is not None:
        s["block_iterations"] = int(block_iterations)
    if block_size_min is not None:
        s["block_size_min"] = int(block_size_min)
    if block_size_max is not None:
        s["block_size_max"] = int(block_size_max)
    if color_jitter_base is not None:
        s["color_jitter_base"] = int(color_jitter_base)

    # Apply drag multiplier to threshold & decay for stronger dragging
    s['threshold'] *= float(drag_mult)          # more pixels keep the ghost
    s['decay'] = 1.0 - (1.0 - s['decay']) / max(0.1, float(drag_mult))

    # Chaos modulates zoom jitter, block glitches, and color jitter
    s['zoom_jitter'] *= float(chaos_mult)
    s['block_prob'] *= float(chaos_mult)
    s['block_iterations'] = max(1, int(s['block_iterations'] * chaos_mult))
    s['color_jitter_base'] = max(1, int(s['color_jitter_base'] * chaos_mult))

    # Randomized zoom/rotation around configured ranges
    s['zoom_drift'] = s['zoom_drift'] * (1.0 + random.uniform(-0.02, 0.03) * chaos_mult)
    rotate_base = random.uniform(-s['rotate_range'], s['rotate_range']) * chaos_mult

    # Randomize Color Mapping
    color_map = [0, 1, 2]  # BGR
    if random.random() < (0.3 * chaos_mult):
        random.shuffle(color_map)  # Psychedelic channel swap

    print(f" -> FPS: {s['fps']}")
    print(f" -> Internal Res: {s['internal_res'][0]}x{s['internal_res'][1]}")
    print(f" -> Zoom base: {s['zoom_drift']:.4f} | Rotate base: {rotate_base:.2f}")
    print(f" -> Threshold: {s['threshold']:.2f} | Decay: {s['decay']:.4f}")
    print(f" -> Smear passes: {s['smear_passes']}")
    print(f" -> Block prob: {s['block_prob']:.3f} | Block iters: {s['block_iterations']}")
    print(f" -> Color jitter base: {s['color_jitter_base']}")
    print(f" -> Channels: {color_map}")

    tmp = tempfile.mkdtemp()

    try:
        # 2. EXTRACT FRAMES (Fast BMP sequence at internal res)
        w, h = s['internal_res']
        print(" -> Extracting low-res frames...")

        cmd_v1 = [
            "ffmpeg", "-y", "-i", v1_path,
            "-vf", f"fps={s['fps']},scale={w}:{h}",
            f"{tmp}/f1_%05d.bmp"
        ]
        subprocess.run(cmd_v1, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        cmd_v2 = [
            "ffmpeg", "-y", "-i", v2_path,
            "-vf", f"fps={s['fps']},scale={w}:{h}",
            f"{tmp}/f2_%05d.bmp"
        ]
        subprocess.run(cmd_v2, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        f1_files = sorted([f for f in os.listdir(tmp) if f.startswith("f1_")])
        f2_files = sorted([f for f in os.listdir(tmp) if f.startswith("f2_")])

        if not f1_files:
            raise Exception("Input 1 failed to extract any frames.")

        total_frames = len(f1_files) + len(f2_files)
        print(f" -> Total frames to mosh: {total_frames}")

        prev_frame = None
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)

        thresh_val = s['threshold']
        decay_val = s['decay']

        # P-frame "hold" mechanic for extra dragging ribbons
        hold_frames_max = int(15 * drag_mult)  # how long a ghost can persist
        hold_counter = 0

        for i in range(total_frames):
            if i % 50 == 0:
                print(f"    Frame {i}/{total_frames}...", end='\r')

            # --- Source Logic (Linear Mix) ---
            if i < len(f1_files):
                curr = cv2.imread(os.path.join(tmp, f1_files[i]))
                if curr is None:
                    continue
                # Transition blend into second clip
                if f2_files and i > len(f1_files) * 0.8:
                    idx2 = random.randint(0, len(f2_files) - 1)
                    alt = cv2.imread(os.path.join(tmp, f2_files[idx2]))
                    if alt is not None:
                        curr = cv2.addWeighted(curr, 0.7, alt, 0.3, 0)
            else:
                idx = i - len(f1_files)
                if idx >= len(f2_files):
                    break
                curr = cv2.imread(os.path.join(tmp, f2_files[idx]))
                if curr is None:
                    continue

            # Apply Color Scramble (Psychedelic channel remap)
            curr = curr[:, :, color_map]

            if prev_frame is None:
                final = curr
                prev_frame = curr
                cv2.imwrite(f"{out_dir}/frame_{i:05d}.bmp", final)
                continue

            # --- FEEDBACK P-FRAME / DRAGGING GHOST ---
            # Slightly varying zoom/rotation per frame for living trails
            local_zoom = s['zoom_drift'] + random.uniform(-s['zoom_jitter'], s['zoom_jitter'])
            local_rotate = rotate_base + random.uniform(-s['rotate_range'], s['rotate_range']) * 0.1

            ghost = apply_feedback_warp(prev_frame, local_zoom, local_rotate, s['smear_passes'])

            # 1. Difference field
            diff = cv2.absdiff(curr, ghost)
            diff_mag = np.sum(diff, axis=2)

            # 2. Mask: where is the scene "similar" -> keep ghost trail
            mask = diff_mag < thresh_val
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

            # 3. Ghost decay blend (prevents total overexposure)
            ghost_decayed = cv2.addWeighted(ghost, decay_val, curr, 1.0 - decay_val, 0)

            # 4. Initial composite
            final = np.where(mask_3d, ghost_decayed, curr)

            # P-frame "hold": sometimes do NOT update prev_frame to stretch smears
            if hold_counter > 0:
                hold_counter -= 1
            else:
                if random.random() < 0.02 * drag_mult:
                    hold_counter = random.randint(1, max(2, hold_frames_max))
                else:
                    prev_frame = final

            # --- CHEAP CHAOS FX ---

            # RGB channel drift (chaos-scaled)
            if random.random() < (0.15 * chaos_mult):
                final = fast_color_jitter(final, s['color_jitter_base'], chaos_mult)

            # Negative flashes
            if random.random() < (0.03 * chaos_mult):
                final = cv2.bitwise_not(final)

            # Blocky glitches
            final = fast_block_shuffle(
                final,
                prob=s['block_prob'],
                iterations=s['block_iterations'],
                size_min=s['block_size_min'],
                size_max=s['block_size_max'],
            )

            # Save as BMP (fast I/O)
            cv2.imwrite(f"{out_dir}/frame_{i:05d}.bmp", final)

        print("\n -> Rendering video (upscale to 720p)...")
        temp_vid = os.path.join(tmp, "temp.mp4")

        cmd_render = [
            "ffmpeg", "-y", "-framerate", str(s['fps']),
            "-i", f"{out_dir}/frame_%05d.bmp",
            "-vf", "scale=1280:720:flags=neighbor",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            temp_vid
        ]
        subprocess.run(cmd_render, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 4. AUDIO DESTRUCTION
        print(" -> Muxing & destroying audio...")
        try:
            audio_path = os.path.join(tmp, "audio.aac")
            subprocess.run(
                ["ffmpeg", "-y", "-i", v1_path, "-vn", "-c:a", "aac", audio_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            af = get_abrasive_audio_filter()
            cmd_mux = [
                "ffmpeg", "-y",
                "-i", temp_vid,
                "-i", audio_path,
                "-af", af,
                "-c:v", "copy",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", out_path,
            ]
            subprocess.run(cmd_mux, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f" !! Audio mux failed, using silent video. Reason: {e}")
            shutil.move(temp_vid, out_path)

        print(f"DONE: {out_path}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyper-Ludovico fast abstract datamosh engine."
    )
    parser.add_argument("--v1", required=True, help="Primary input video.")
    parser.add_argument("--v2", required=True, help="Secondary input / texture video.")
    parser.add_argument("--out", default="output.mp4", help="Output video path.")
    parser.add_argument("--drag", type=float, default=1.0,
                        help="Drag multiplier (trail strength & length). Default=1.0")
    parser.add_argument("--chaos", type=float, default=1.0,
                        help="Chaos multiplier (glitches, color drifts). Default=1.0")

    # Extra manual controls (all optional)
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed (-1 = random).")
    parser.add_argument("--fps", type=int, default=None,
                        help="Override FPS (default from BASE_SETTINGS).")
    parser.add_argument("--internal-width", type=int, default=None,
                        help="Internal processing width (default 426).")
    parser.add_argument("--internal-height", type=int, default=None,
                        help="Internal processing height (default 240).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Base threshold for ghost masking.")
    parser.add_argument("--decay", type=float, default=None,
                        help="Base decay (0-1) for trails.")
    parser.add_argument("--zoom", type=float, default=None,
                        help="Base zoom drift.")
    parser.add_argument("--zoom-jitter", type=float, default=None,
                        help="Frame-to-frame zoom randomness.")
    parser.add_argument("--rotate-range", type=float, default=None,
                        help="Max rotation range for feedback.")
    parser.add_argument("--smear-passes", type=int, default=None,
                        help="Number of smear passes per frame (1-3 is good).")
    parser.add_argument("--block-prob", type=float, default=None,
                        help="Probability of block glitches.")
    parser.add_argument("--block-iters", type=int, default=None,
                        help="Number of blocks to scramble when glitch triggers.")
    parser.add_argument("--block-size-min", type=int, default=None,
                        help="Minimum block size for glitches.")
    parser.add_argument("--block-size-max", type=int, default=None,
                        help="Maximum block size for glitches.")
    parser.add_argument("--color-jitter-base", type=int, default=None,
                        help="Base intensity of color channel drift.")
    parser.add_argument("--crf", type=int, default=22,
                        help="CRF for x264 (lower = higher quality, larger file).")

    args = parser.parse_args()

    internal_res = None
    if args.internal_width is not None and args.internal_height is not None:
        internal_res = (args.internal_width, args.internal_height)

    process_video(
        args.v1,
        args.v2,
        args.out,
        drag_mult=args.drag,
        chaos_mult=args.chaos,
        fps=args.fps,
        internal_res=internal_res,
        threshold=args.threshold,
        decay=args.decay,
        zoom=args.zoom,
        zoom_jitter=args.zoom_jitter,
        rotate_range=args.rotate_range,
        smear_passes=args.smear_passes,
        block_prob=args.block_prob,
        block_iterations=args.block_iters,
        block_size_min=args.block_size_min,
        block_size_max=args.block_size_max,
        color_jitter_base=args.color_jitter_base,
        seed=args.seed,
        crf=args.crf,
    )

