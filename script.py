#!/usr/bin/env python3
# Ludovico: Extreme datamosh ++ (ultra-lag drag, psychedelic, faster)
# Usage example:
#   python ludovico_mosh.py --v1 input.mp4 --v2 image2.mp4 --out output_abrasive.mp4 --fps 24 --res 640x360 --extreme

import os
import sys
import shutil
import subprocess
import random
import argparse
import tempfile
import time
from collections import deque

import numpy as np
import cv2

# -------------------- DEFAULT CONFIG (tweakable) --------------------
SETTINGS = {
    "fps": 24,
    "internal_res": (640, 360),   # lower processing res for speed + blocky style
    "transition_frac": 0.2,
    "mosh_threshold": 12,         # lower -> more smear (when using color diff)
    "bloom_chance": 0.12,
    "sort_chance": 0.35,
    "invert_chance": 0.04,
    "stutter_chance": 0.25,
    "history_depth": 8,           # number of past processed frames to keep for aggressive dragging
    "drag_strength": 0.85,        # how strongly old pixel colors survive
    "flow_strength": 0.8,         # how much optical flow drags pixels
    "slice_pixel_sort_max_height": 48,  # max height for slice sorting (limits cost)
    "block_teleport_chance": 0.10, # randomly teleport blocks from older frames
    "extreme_mode_scale": 1.6,    # multiplier for intensity when --extreme passed
}

# -------------------- HELPERS --------------------

def run_ffmpeg(cmd):
    """Run ffmpeg command and raise on error. Prints stderr for troubleshooting."""
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc
    except subprocess.CalledProcessError as e:
        print("FFmpeg Error:", e.stderr, file=sys.stderr)
        raise

def get_audio_filter(extreme=False):
    """Return an abrasive audio filter chain (ffmpeg filter)."""
    base = "acrusher=level_in=8:level_out=18:bits=8:mode=log:aa=1, aecho=0.8:0.9:100:0.5, vibrato=f=6:d=0.5"
    if extreme:
        base += ", atempo=1.02, afftfilt=real='hypot(re,im)':imag='0.0'"
    return base

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

# -------------------- CORE EFFECTS --------------------

def pixel_difference_mask(curr, prev):
    """Return boolean mask where pixels are considered 'similar' (so old should survive)."""
    if prev is None:
        return np.zeros(curr.shape[:2], dtype=bool)
    # Sum absolute channel diff (fast)
    diff = np.sum(np.abs(curr.astype(int) - prev.astype(int)), axis=2)
    return diff < SETTINGS['mosh_threshold']

def fast_partial_pixel_sort(img_arr, chance=0.35):
    """Apply pixel sorting only to random horizontal slices for speed."""
    h, w = img_arr.shape[:2]
    out = img_arr.copy()
    if random.random() >= chance:
        return out
    # choose a few slices
    slices = random.randint(1, 3)
    for _ in range(slices):
        y = random.randint(0, max(0, h - 2))
        max_h = min(SETTINGS['slice_pixel_sort_max_height'], h - y)
        slice_h = random.randint(1, max(1, max_h))
        row_slice = out[y:y+slice_h, :, :]
        # sort by luminance across rows (vectorized)
        lum = np.dot(row_slice[...,:3], [0.299, 0.587, 0.114])
        order = np.argsort(lum, axis=1)
        # apply sorted indices along width
        try:
            sorted_slice = np.take_along_axis(row_slice, np.expand_dims(order, axis=2), axis=1)
            out[y:y+slice_h, :, :] = sorted_slice
        except Exception:
            # in case of weird shapes, skip
            pass
    return out

def crush_resolution_img(img, crush_factor=4):
    """Downscale/upscale to create blocky compression artifacts."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w//crush_factor), max(1, h//crush_factor)), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def chroma_shift(img, max_shift=6):
    """Split channels and offset to create chromatic aberration feel."""
    b, g, r = cv2.split(img)
    h, w = b.shape
    # random tiny offsets
    dx1, dy1 = random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift)
    dx2, dy2 = random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift)
    M1 = np.float32([[1, 0, dx1],[0,1,dy1]])
    M2 = np.float32([[1, 0, dx2],[0,1,dy2]])
    b_t = cv2.warpAffine(b, M1, (w,h), borderMode=cv2.BORDER_REFLECT)
    r_t = cv2.warpAffine(r, M2, (w,h), borderMode=cv2.BORDER_REFLECT)
    return cv2.merge([b_t, g, r_t])

def optical_flow_smear(curr_gray, prev_gray, prev_processed, strength=0.8):
    """
    Calculate optical flow vecs between previous and current grayscale frames and warp previous processed frame
    along the flow to create pixel dragging trails. Returns warped previous and a flow map.
    """
    h, w = curr_gray.shape
    # Farneback is fairly fast and produces dense flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # flow is dx,dy per pixel; we want to warp previous processed pixels towards the flow direction
    # build remap coordinates
    coords_x, coords_y = np.meshgrid(np.arange(w), np.arange(h))
    # scale flow
    mv_x = (coords_x + flow[...,0] * strength).astype(np.float32)
    mv_y = (coords_y + flow[...,1] * strength).astype(np.float32)
    # remap previous frame according to motion vectors (warpPrev -> curr space)
    warped_prev = cv2.remap(prev_processed, mv_x, mv_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped_prev, flow

def block_teleport_from_history(history_buf, target_frame, chance=0.1):
    """
    Randomly copy blocks from older frames in history into target_frame, simulating delayed youtube artifacts.
    history_buf: deque of older processed frames (most recent last).
    """
    if len(history_buf) < 2 or random.random() >= chance:
        return target_frame
    src_idx = random.randint(0, len(history_buf)-1)
    src = history_buf[src_idx]
    out = target_frame.copy()
    h, w = out.shape[:2]
    # random block
    bw = random.randint(16, min(128, w//2))
    bh = random.randint(16, min(128, h//2))
    sx = random.randint(0, max(0, w-bw))
    sy = random.randint(0, max(0, h-bh))
    tx = random.randint(0, max(0, w-bw))
    ty = random.randint(0, max(0, h-bh))
    out[ty:ty+bh, tx:tx+bw] = src[sy:sy+bh, sx:sx+bw]
    return out

# -------------------- MAIN PROCESS --------------------

def process(v1, v2, out_file, fps=None, internal_res=None, extreme=False):
    st = time.time()
    if fps is None:
        fps = SETTINGS['fps']
    if internal_res is None:
        internal_res = SETTINGS['internal_res']
    if extreme:
        # amplify certain settings for max chaos
        SETTINGS['mosh_threshold'] = max(1, int(SETTINGS['mosh_threshold'] * 0.6))
        SETTINGS['sort_chance'] = min(1.0, SETTINGS['sort_chance'] * SETTINGS['extreme_mode_scale'])
        SETTINGS['bloom_chance'] = min(1.0, SETTINGS['bloom_chance'] * SETTINGS['extreme_mode_scale'])
        SETTINGS['invert_chance'] = min(1.0, SETTINGS['invert_chance'] * SETTINGS['extreme_mode_scale'])
        SETTINGS['drag_strength'] = min(0.99, SETTINGS['drag_strength'] * SETTINGS['extreme_mode_scale'])
        SETTINGS['flow_strength'] = min(1.6, SETTINGS['flow_strength'] * SETTINGS['extreme_mode_scale'])

    tmpdir = tempfile.mkdtemp(prefix="ludovico_")
    prev_dir = os.getcwd()
    os.chdir(tmpdir)
    try:
        print("Working dir:", tmpdir)
        # 1) sanity check inputs
        if not os.path.exists(v1):
            raise FileNotFoundError(f"v1 not found: {v1}")
        if not os.path.exists(v2):
            raise FileNotFoundError(f"v2 not found: {v2}")

        w, h = internal_res
        print("Extracting frames (low-res) with ffmpeg...")
        # extract frames for both inputs scaled to internal_res - use fast output codec BMP sequence
        run_ffmpeg(["ffmpeg", "-y", "-i", v1, "-vf", f"fps={fps},scale={w}:{h}", "f1_%05d.bmp"])
        run_ffmpeg(["ffmpeg", "-y", "-i", v2, "-vf", f"fps={fps},scale={w}:{h}", "f2_%05d.bmp"])

        frames1 = sorted([f for f in os.listdir(".") if f.startswith("f1_")])
        frames2 = sorted([f for f in os.listdir(".") if f.startswith("f2_")])
        if not frames1:
            raise RuntimeError("No frames extracted from v1")
        total_frames = len(frames1) + max(0, len(frames2))
        print("Total frames to process:", total_frames)

        # buffers / state
        prev_processed = None
        prev_orig_gray = None
        history = deque(maxlen=SETTINGS['history_depth'])  # store previous processed

        out_frames_dir = os.path.join(tmpdir, "out_frames")
        ensure_dir(out_frames_dir)

        # process frames sequentially (preserve temporal dependency)
        for i in range(total_frames):
            # choose source frame
            if i < len(frames1):
                fname = frames1[i]
            else:
                idx = i - len(frames1)
                if idx < len(frames2):
                    fname = frames2[idx]
                else:
                    break
            orig = cv2.imread(fname, cv2.IMREAD_COLOR)
            if orig is None:
                print("Warning: failed to load", fname)
                continue

            # optionally do transition/difference with second clip near switch
            if i < len(frames1):
                left = len(frames1) - i
                trans_len = max(1, int(len(frames1) * SETTINGS['transition_frac']))
                if left < trans_len and frames2:
                    idx2 = int((1 - (left/trans_len)) * (len(frames2)-1))
                    alt = cv2.imread(frames2[idx2])
                    if alt is not None:
                        orig = cv2.absdiff(orig, alt)

            # quick pre-conversions
            curr = orig.copy()
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            # STUTTER (freeze frame effect)
            if random.random() < SETTINGS['stutter_chance'] and len(history) > 0:
                # pick a recent processed frame to freeze
                curr = history[-1].copy()
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            # MOSH: optical-flow based smear of prev_processed onto current
            if prev_processed is not None:
                warped_prev, flow = optical_flow_smear(curr_gray, prev_orig_gray, prev_processed,
                                                      strength=SETTINGS['flow_strength']* (1.5 if extreme else 1.0))
                # combine: where current and warped_prev similar -> keep warped_prev (dragged pixels)
                sim_mask = pixel_difference_mask(curr, warped_prev)
                sim3 = np.repeat(sim_mask[:, :, np.newaxis], 3, axis=2)
                # smearing: mix warped_prev and curr with drag_strength
                mixed = np.where(sim3, (warped_prev * SETTINGS['drag_strength'] + curr * (1 - SETTINGS['drag_strength'])).astype(np.uint8), curr)
            else:
                mixed = curr

            # Aggressive feedback: composite previous processed over current with slight alpha and color transform
            if prev_processed is not None and random.random() < 0.6:
                alpha = 0.08 + random.random() * 0.2
                # color-shifted previous
                pm = prev_processed.copy()
                if random.random() < 0.3:
                    pm = chroma_shift(pm, max_shift=8 if extreme else 4)
                mixed = cv2.addWeighted(mixed, 1 - alpha, pm, alpha, 0)

            # Pixel-sorting slices (fast partial)
            if random.random() < SETTINGS['sort_chance']:
                mixed = fast_partial_pixel_sort(mixed, chance=1.0)

            # Block teleport from history (strong youtube-lagging artifact)
            if random.random() < SETTINGS['block_teleport_chance'] * (1.5 if extreme else 1.0):
                mixed = block_teleport_from_history(history, mixed, chance=1.0)

            # Invert flash
            if random.random() < SETTINGS['invert_chance']:
                mixed = 255 - mixed

            # Bloom / crush occasionally
            if random.random() < SETTINGS['bloom_chance'] * (2.0 if extreme else 1.0):
                mixed = crush_resolution_img(mixed, crush_factor=random.choice([2,3,4,6]))

            # occasional chroma shift / jitter
            if random.random() < 0.25:
                mixed = chroma_shift(mixed, max_shift=(12 if extreme else 6))

            # store output
            out_name = os.path.join(out_frames_dir, f"frame_{i:05d}.bmp")
            cv2.imwrite(out_name, mixed, [int(cv2.IMWRITE_BMP)])
            # push to history
            history.append(mixed.copy())
            prev_processed = mixed.copy()
            prev_orig_gray = curr_gray.copy()

            if (i+1) % 25 == 0:
                print(f"Processed frame {i+1}/{total_frames}")

        # cleanup extracted sequences
        for f in frames1 + frames2:
            try:
                os.remove(f)
            except Exception:
                pass

        print("Encoding final video (upscale to 1280x720 with neighbor filter to keep blocks)...")
        temp_vid = os.path.join(tmpdir, "temp_vid.mp4")
        run_ffmpeg([
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", f"{out_frames_dir}/frame_%05d.bmp",
            "-vf", "scale=1280:720:flags=neighbor",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p",
            temp_vid
        ])

        # attach audio if present (and apply harsh filter)
        audio_exists = False
        try:
            run_ffmpeg(["ffmpeg", "-y", "-i", v1, "-vn", "-c:a", "aac", "temp_audio.aac"])
            audio_exists = True
        except Exception:
            audio_exists = False

        if audio_exists:
            print("Applying abrasive audio chain and muxing...")
            run_ffmpeg([
                "ffmpeg", "-y", "-i", temp_vid, "-i", "temp_audio.aac",
                "-af", get_audio_filter(extreme=extreme),
                "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", out_file
            ])
        else:
            shutil.move(temp_vid, out_file)

        print("Done. Output:", out_file)
        print("Elapsed:", time.time() - st, "seconds")

    finally:
        # cleanup completely
        os.chdir(prev_dir)
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

# -------------------- CLI --------------------

def parse_res(s):
    if 'x' in s:
        a,b = s.split('x')
        return (int(a), int(b))
    raise argparse.ArgumentTypeError("res must be WIDTHxHEIGHT, e.g. 640x360")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ludovico: extreme datamosher")
    p.add_argument("--v1", required=True, help="Primary input video file path")
    p.add_argument("--v2", required=True, help="Secondary input video or image (for transitions)")
    p.add_argument("--out", default="output_abrasive.mp4", help="Output file")
    p.add_argument("--fps", type=int, default=SETTINGS['fps'], help="FPS to process at")
    p.add_argument("--res", type=parse_res, default=SETTINGS['internal_res'], help="Internal processing resolution WxH (lower = faster)")
    p.add_argument("--extreme", action="store_true", help="Crank intensity and chaos")
    args = p.parse_args()

    # apply CLI args to settings
    SETTINGS['fps'] = args.fps
    SETTINGS['internal_res'] = args.res

    try:
        process(args.v1, args.v2, args.out, fps=args.fps, internal_res=args.res, extreme=args.extreme)
    except Exception as e:
        print("Fatal:", e)
        sys.exit(1)

