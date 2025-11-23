#!/usr/bin/env python3
"""
Datamosh / YouTube Low Data Effect (P-Frame Smearing) — fixed & amplified.

Requirements:
    pip install pillow numpy

Usage:
    python extreme_datamosh.py

Input files:
    - input.mp4   (or any video / image for the first source)
    - image2.mp4  (second source to transition into)

Output:
    - output_horrifying_mosh.mp4
"""
import os
import shutil
import subprocess
import random
import math
import numpy as np
from PIL import Image, ImageChops

# --- CONFIGURATION (cranked up for extreme surreal datamosh) ---
SETTINGS = {
    'fps': 24,
    'transition_duration': 0.20,      # fraction of v1 frames used to morph into v2
    'pixel_sort_threshold': 0.85,     # chance to apply pixel sorting to a chunk
    'mosh_threshold': 40,             # lower = more exact-preserve, higher = more smear (0-765 for full RGB sum)
    'bloom_intensity': 0.85,          # chance to apply JPEG re-encode artifacts
    'block_size': 16,                 # macroblock size to decide smearing (like codec blocks)
    'block_spread': 2,                # expand neighboring blocks by this many blocks when smearing
    'frame_dup_chance': 0.12,         # chance to duplicate previous frame (classic datamosh technique)
    'jpeg_passes': 2,                 # repeated JPEG resaves to amplify blockiness
    'pixel_sort_chance_per_frame': 0.6
}

# --- UTIL: run ffmpeg ---
def run_ffmpeg(cmd):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print("FFMPEG ERROR:", e.stderr[:400])
        raise

# --- FRAME EXTRACTION (robust) ---
def safe_extract_frames(input_path, output_folder, target_fps=24):
    os.makedirs(output_folder, exist_ok=True)
    pattern = os.path.join(output_folder, "frame_%05d.png")
    try:
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", input_path, "-vf", f"fps={target_fps}", "-vsync", "0",
            pattern
        ])
        print(f"Extracted frames from {input_path} -> {output_folder}")
        return
    except Exception:
        # fallback: if it's an image, create repeated frames
        try:
            img = Image.open(input_path).convert("RGB")
            count = target_fps * 3
            for i in range(count):
                img.save(os.path.join(output_folder, f"frame_{i:05d}.png"))
            print(f"Treated {input_path} as static image and generated frames.")
        except Exception as e:
            raise RuntimeError(f"Could not extract frames from {input_path}: {e}")

# --- AUDIO GLITCH (simple) ---
def extract_and_glitch_audio(video_path, output_audio):
    temp_raw = "temp_raw_audio.aac"
    try:
        run_ffmpeg(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video_path, "-vn", "-c:a", "aac", temp_raw])
    except Exception:
        return False
    # apply echo + vibrato + slow
    try:
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", temp_raw, "-af",
            "aecho=0.8:0.9:1000:0.3,vibrato=f=5.0:d=0.4,atempo=0.92",
            "-c:a", "aac", output_audio
        ])
        return True
    except Exception:
        return False

# --- EFFECTS HELPERS ---

def jpeg_bloom(image, intensity=0.5, passes=1):
    """Re-save image to JPEG multiple times to accentuate block artifacts and color shifts."""
    if random.random() > intensity:
        return image
    import io
    out = image.convert("RGB")
    for p in range(max(1, passes)):
        buf = io.BytesIO()
        q = random.randint(6, 18)
        out.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    return out

def channel_shift(image, max_offset=8):
    """Slightly offset color channels to create chromatic aberration and surreal look."""
    if random.random() > 0.6:
        return image
    r, g, b = image.split()
    w, h = image.size
    def shift_band(band):
        dx = random.randint(-max_offset, max_offset)
        dy = random.randint(-max_offset, max_offset)
        return ImageChops.offset(band, dx, dy)
    r2 = shift_band(r)
    g2 = shift_band(g)
    b2 = shift_band(b)
    return Image.merge("RGB", (r2, g2, b2))

def pixel_sort(image, probability=0.5):
    """Randomly sort pixel runs horizontally/vertically across several rows/cols."""
    if random.random() > probability:
        return image
    arr = np.array(image)
    h, w, c = arr.shape
    # choose random band height/width
    band_h = random.randint(1, max(1, h // 12))
    start_row = random.randint(0, max(0, h - band_h))
    end_row = start_row + band_h
    for r in range(start_row, end_row):
        row = arr[r].copy()
        lum = np.dot(row[..., :3], [0.299, 0.587, 0.114])
        # sort by luminance but in randomized direction to avoid uniform look
        idx = np.argsort(lum)
        if random.random() > 0.5:
            idx = idx[::-1]
        arr[r] = row[idx]
    return Image.fromarray(arr)

# --- P-FRAME MACROBLOCK SMEAR ---
def simulated_p_frame_mosh(cur_img, prev_img, threshold=15, block_size=16, spread=1):
    """
    Block-based smear:
    - Compare blocks between current and previous frame (sum of absolute differences).
    - If block difference < threshold -> copy the ENTIRE block from previous frame (smear).
    - Optionally expand smear to neighbor blocks (spread).
    """
    if prev_img is None:
        return cur_img
    # ensure same size
    if cur_img.size != prev_img.size:
        prev_img = prev_img.resize(cur_img.size)
    cur_arr = np.array(cur_img).astype(np.int32)
    prev_arr = np.array(prev_img).astype(np.int32)
    h, w, ch = cur_arr.shape

    bs = max(4, int(block_size))
    # pad to multiple of bs
    pad_h = (bs - (h % bs)) % bs
    pad_w = (bs - (w % bs)) % bs
    if pad_h or pad_w:
        cur_arr = np.pad(cur_arr, ((0, pad_h), (0, pad_w), (0,0)), mode='edge')
        prev_arr = np.pad(prev_arr, ((0, pad_h), (0, pad_w), (0,0)), mode='edge')
    H, W, _ = cur_arr.shape
    bh = H // bs
    bw = W // bs

    # compute block diffs
    # reshape into blocks: (bh, bs, bw, bs, ch)
    cur_blocks = cur_arr.reshape(bh, bs, bw, bs, ch)
    prev_blocks = prev_arr.reshape(bh, bs, bw, bs, ch)
    # compute sum absolute diff per block
    diff_blocks = np.sum(np.abs(cur_blocks - prev_blocks), axis=(1,3,4))  # shape (bh, bw)

    # threshold scaled to block total (max diff per pixel is 255*3)
    # allow threshold to be in same units as sum per block
    # if user used small threshold (like 40) we compare average per pixel:
    avg_diff_blocks = diff_blocks / (bs * bs)  # average diff per pixel per block (0-255*3)
    # create boolean mask of blocks to smear
    smear_block_mask = avg_diff_blocks < threshold

    # optionally spread smear to neighbors to make it more dramatic
    if spread > 0:
        new_mask = smear_block_mask.copy()
        for y in range(bh):
            for x in range(bw):
                if smear_block_mask[y, x]:
                    y0 = max(0, y - spread)
                    y1 = min(bh, y + spread + 1)
                    x0 = max(0, x - spread)
                    x1 = min(bw, x + spread + 1)
                    new_mask[y0:y1, x0:x1] = True
        smear_block_mask = new_mask

    # Build output by copying blocks from prev where mask True
    out_arr = cur_arr.copy()
    for by in range(bh):
        for bx in range(bw):
            if smear_block_mask[by, bx]:
                y0, y1 = by * bs, (by + 1) * bs
                x0, x1 = bx * bs, (bx + 1) * bs
                out_arr[y0:y1, x0:x1, :] = prev_arr[y0:y1, x0:x1, :]

    # unpad if padded
    if pad_h or pad_w:
        out_arr = out_arr[:h, :w, :]

    return Image.fromarray(out_arr.astype('uint8'))

# --- MAIN WORKFLOW ---

def main():
    v1_path = "input.mp4"
    v2_path = "image2.mp4"
    final_output = "output_horrifying_mosh.mp4"

    t_v1 = "frames_v1"
    t_v2 = "frames_v2"
    t_out = "frames_out"
    t_audio = "glitched_audio.aac"

    for d in [t_v1, t_v2, t_out]:
        if os.path.exists(d): shutil.rmtree(d)

    # 1. Extract frames (with fallback)
    safe_extract_frames(v1_path, t_v1, SETTINGS['fps'])
    safe_extract_frames(v2_path, t_v2, SETTINGS['fps'])

    has_audio = extract_and_glitch_audio(v1_path, t_audio)

    files_v1 = sorted([os.path.join(t_v1, f) for f in os.listdir(t_v1) if f.lower().endswith('.png')])
    files_v2 = sorted([os.path.join(t_v2, f) for f in os.listdir(t_v2) if f.lower().endswith('.png')])

    if not files_v1 and not files_v2:
        raise RuntimeError("No frames extracted from either input.")

    first_frame = files_v1[0] if files_v1 else files_v2[0]
    with Image.open(first_frame) as img:
        target_size = img.size

    total_estimated = len(files_v1) + len(files_v2)
    os.makedirs(t_out, exist_ok=True)
    print(f"Target resolution: {target_size} — producing ~{total_estimated} frames...")

    prev_img = None
    out_index = 0

    # Preload lists for speed (optional)
    def open_rgb(path):
        return Image.open(path).convert("RGB")

    for i in range(len(files_v1) + len(files_v2)):
        # Determine source
        if i < len(files_v1):
            src = open_rgb(files_v1[i])
            # transition into v2 near the end of v1
            frames_left = len(files_v1) - i
            transition_len = max(1, int(len(files_v1) * SETTINGS['transition_duration']))
            if frames_left <= transition_len and files_v2:
                tnorm = 1 - (frames_left / float(max(1, transition_len)))
                v2_idx = min(len(files_v2)-1, int(tnorm * (len(files_v2)-1)))
                img_v2 = open_rgb(files_v2[v2_idx]).resize(src.size)
                # use difference + additive blend to make weird morphs
                src = ImageChops.add(ImageChops.difference(src, img_v2), img_v2, scale=1.2, offset=-10)
        else:
            v2_idx = i - len(files_v1)
            if v2_idx >= len(files_v2):
                break
            src = open_rgb(files_v2[v2_idx])

        if src.size != target_size:
            src = src.resize(target_size)

        # Occasionally duplicate previous frame (makes no I-frame effect)
        if prev_img is not None and random.random() < SETTINGS['frame_dup_chance']:
            # save the previous frame again to exaggerate smear
            dup_path = os.path.join(t_out, f"frame_{out_index:05d}.png")
            prev_img.save(dup_path)
            out_index += 1
            print(f"Frame duplication at output index {out_index}", end='\r')

        # 1) P-frame macroblock smear
        src = simulated_p_frame_mosh(src, prev_img,
                                     threshold=SETTINGS['mosh_threshold'],
                                     block_size=SETTINGS['block_size'],
                                     spread=SETTINGS['block_spread'])

        # 2) Pixel sort (occasionally)
        src = pixel_sort(src, SETTINGS['pixel_sort_chance_per_frame'])

        # 3) channel chromatic shifts
        src = channel_shift(src)

        # 4) aggressive jpeg bloom (repeated)
        src = jpeg_bloom(src, intensity=SETTINGS['bloom_intensity'], passes=SETTINGS['jpeg_passes'])

        # final safety convert and save
        out_path = os.path.join(t_out, f"frame_{out_index:05d}.png")
        src.convert("RGB").save(out_path)
        prev_img = src.copy()
        out_index += 1
        print(f"Rendered frame {out_index}", end='\r')

    print("\nAssembling video with ffmpeg...")

    temp_vid = "temp_video.mp4"
    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(SETTINGS['fps']),
        "-i", os.path.join(t_out, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", temp_vid
    ])

    if has_audio and os.path.exists(t_audio):
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", temp_vid, "-i", t_audio,
            "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", final_output
        ])
    else:
        os.replace(temp_vid, final_output)

    print(f"\nDONE — output: {final_output}")

if __name__ == "__main__":
    main()

