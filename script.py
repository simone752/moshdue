#!/usr/bin/env python3
"""
Extreme Datamosh — Robust rewrite (Fixed for Defaults)

Requirements:
    - ffmpeg (and ffprobe) on PATH
    - Python 3.8+
    - pip install pillow numpy

Usage:
    python script.py
    OR
    python script.py --v1 myvideo.mp4 --v2 myimage.png
"""
import argparse
import os
import shutil
import subprocess
import sys
import random
import io
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image, ImageChops

# -------------------------
# Configuration / Defaults
# -------------------------
DEFAULTS = {
    "fps": 24,
    "transition_fraction": 0.20,
    "mosh_threshold": 36.0,       # average per-pixel diff threshold (0-765)
    "block_size": 16,
    "block_spread": 2,
    "frame_dup_chance": 0.12,
    "jpeg_intensity": 0.85,
    "jpeg_passes": 2,
    "pixel_sort_chance": 0.6,
    "channel_shift_chance": 0.6,
    "pixel_sort_band_div": 12,     # controls max band height for pixel sort
}

# -------------------------
# Utilities: ffmpeg / ffprobe
# -------------------------
def run(cmd: List[str], capture_output=True) -> Tuple[int, str, str]:
    """Run command, return (retcode, stdout, stderr)."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE if capture_output else None,
                          stderr=subprocess.PIPE if capture_output else None,
                          text=True)
    out = proc.stdout if proc.stdout is not None else ""
    err = proc.stderr if proc.stderr is not None else ""
    return proc.returncode, out, err

def check_is_image(path: str) -> bool:
    """Use Pillow to probe whether a path is an image (safe try)."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False

# -------------------------
# Robust frame extraction
# -------------------------
def extract_frames_with_ffmpeg(input_path: str, out_folder: str, fps: int) -> None:
    """
    Extract frames using ffmpeg into out_folder/frame_%05d.png
    """
    os.makedirs(out_folder, exist_ok=True)
    pattern = os.path.join(out_folder, "frame_%05d.png")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path, "-vf", f"fps={fps}", "-vsync", "0",
        pattern
    ]
    rc, out, err = run(cmd)
    if rc != 0:
        print(f"[ffmpeg extraction failed] rc={rc}\n{err.strip()}\n")
        raise RuntimeError(f"ffmpeg failed to extract frames from {input_path}")

    # verify at least one frame exists
    produced = [f for f in os.listdir(out_folder) if f.lower().endswith(".png")]
    if not produced:
        raise RuntimeError(f"ffmpeg reported success but produced 0 frames in {out_folder}")

def image_fallback_generate_frames(image_path: str, out_folder: str, fps: int, duration_secs: int = 3) -> None:
    """If input is a static image, produce fps*duration repeated frames."""
    os.makedirs(out_folder, exist_ok=True)
    from PIL import Image
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        count = fps * duration_secs
        for i in range(count):
            im.save(os.path.join(out_folder, f"frame_{i:05d}.png"))

# -------------------------
# Audio processing
# -------------------------
def extract_and_glitch_audio(input_path: str, out_audio: str) -> bool:
    """
    Extract audio stream and process to produce a 'glitched' audio track.
    """
    temp = "temp_raw_audio.aac"
    # extract audio
    rc, _, err = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", input_path, "-vn", "-c:a", "aac", temp])
    if rc != 0 or not os.path.exists(temp):
        if os.path.exists(temp): os.remove(temp)
        return False
    # apply filters
    rc2, _, err2 = run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", temp,
        "-af", "aecho=0.8:0.9:1000:0.3,vibrato=f=5.0:d=0.4,atempo=0.92",
        "-c:a", "aac", out_audio
    ])
    if os.path.exists(temp): os.remove(temp)
    return rc2 == 0 and os.path.exists(out_audio)

# -------------------------
# Visual effect primitives
# -------------------------
def jpeg_bloom(img: Image.Image, intensity: float = 0.7, passes: int = 1) -> Image.Image:
    if random.random() > intensity:
        return img
    out = img.convert("RGB")
    for _ in range(max(1, passes)):
        buf = io.BytesIO()
        q = random.randint(6, 20)
        out.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    return out

def channel_shift(img: Image.Image, max_offset: int = 8, chance: float = 0.6) -> Image.Image:
    if random.random() > chance:
        return img
    r, g, b = img.split()
    def ofs(band):
        dx = random.randint(-max_offset, max_offset)
        dy = random.randint(-max_offset, max_offset)
        return ImageChops.offset(band, dx, dy)
    return Image.merge("RGB", (ofs(r), ofs(g), ofs(b)))

def pixel_sort(img: Image.Image, chance: float = 0.6, band_div: int = 12) -> Image.Image:
    if random.random() > chance:
        return img
    arr = np.array(img)
    h, w, _ = arr.shape
    band_h = random.randint(1, max(1, h // band_div))
    start = random.randint(0, max(0, h - band_h))
    end = start + band_h
    for r in range(start, end):
        row = arr[r].copy()
        lum = np.dot(row[..., :3], [0.299, 0.587, 0.114])
        idx = np.argsort(lum)
        if random.random() > 0.5:
            idx = idx[::-1]
        arr[r] = row[idx]
    return Image.fromarray(arr)

# -------------------------
# P-frame macroblock smear
# -------------------------
def macroblock_smear(current: Image.Image, previous: Optional[Image.Image],
                     block_size: int = 16, threshold: float = 36.0, spread: int = 1) -> Image.Image:
    if previous is None:
        return current

    if current.size != previous.size:
        previous = previous.resize(current.size)

    cur = np.array(current).astype(np.int32)
    prev = np.array(previous).astype(np.int32)
    h, w, c = cur.shape

    bs = max(4, int(block_size))
    pad_h = (bs - (h % bs)) % bs
    pad_w = (bs - (w % bs)) % bs

    if pad_h or pad_w:
        cur = np.pad(cur, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        prev = np.pad(prev, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')

    H, W, _ = cur.shape
    bh = H // bs
    bw = W // bs

    cur_blocks = cur.reshape(bh, bs, bw, bs, c)
    prev_blocks = prev.reshape(bh, bs, bw, bs, c)

    # sum absolute diff per block, normalize to per-pixel average
    diff_blocks = np.sum(np.abs(cur_blocks - prev_blocks), axis=(1, 3, 4))
    avg_blocks = diff_blocks / (bs * bs)

    mask = avg_blocks < threshold

    if spread > 0:
        expanded = mask.copy()
        for y in range(bh):
            for x in range(bw):
                if mask[y, x]:
                    y0 = max(0, y - spread)
                    y1 = min(bh, y + spread + 1)
                    x0 = max(0, x - spread)
                    x1 = min(bw, x + spread + 1)
                    expanded[y0:y1, x0:x1] = True
        mask = expanded

    out = cur.copy()
    for by in range(bh):
        y0, y1 = by * bs, (by + 1) * bs
        for bx in range(bw):
            if mask[by, bx]:
                x0, x1 = bx * bs, (bx + 1) * bs
                out[y0:y1, x0:x1, :] = prev[y0:y1, x0:x1, :]

    if pad_h or pad_w:
        out = out[:h, :w, :]

    return Image.fromarray(out.astype("uint8"))

# -------------------------
# Main Processing Pipeline
# -------------------------
def process(v1_frames: List[str], v2_frames: List[str], out_folder: str, settings: dict):
    os.makedirs(out_folder, exist_ok=True)
    target_size = Image.open(v1_frames[0] if v1_frames else v2_frames[0]).size
    prev_img = None
    out_idx = 0

    total_count = len(v1_frames) + len(v2_frames)
    print(f"Processing ~{total_count} frames at {target_size}...")

    def open_rgb(path):
        return Image.open(path).convert("RGB")

    for i in range(len(v1_frames) + len(v2_frames)):
        # Source selection & transition morph
        if i < len(v1_frames):
            src = open_rgb(v1_frames[i])
            frames_left = len(v1_frames) - i
            transition_len = max(1, int(len(v1_frames) * settings["transition_fraction"]))
            if frames_left <= transition_len and len(v2_frames) > 0:
                tnorm = 1.0 - (frames_left / float(max(1, transition_len)))
                v2_idx = min(len(v2_frames)-1, int(tnorm * (len(v2_frames)-1)))
                img_v2 = open_rgb(v2_frames[v2_idx]).resize(src.size)
                # weirder morph: additive of diff + blend
                src = ImageChops.add(ImageChops.difference(src, img_v2), img_v2, scale=1.2, offset=-8)
        else:
            v2_idx = i - len(v1_frames)
            if v2_idx >= len(v2_frames):
                break
            src = open_rgb(v2_frames[v2_idx])

        if src.size != target_size:
            src = src.resize(target_size)

        # Random frame duplication (simulate missing I-frames)
        if prev_img is not None and random.random() < settings["frame_dup_chance"]:
            dup_path = os.path.join(out_folder, f"frame_{out_idx:05d}.png")
            prev_img.save(dup_path)
            out_idx += 1

        # Macroblock smear (P-frame)
        src = macroblock_smear(src, prev_img,
                               block_size=settings["block_size"],
                               threshold=settings["mosh_threshold"],
                               spread=settings["block_spread"])

        # Pixel sort (occasionally)
        src = pixel_sort(src, chance=settings["pixel_sort_chance"], band_div=settings["pixel_sort_band_div"])

        # channel shift (occasionally)
        src = channel_shift(src, max_offset=8, chance=settings["channel_shift_chance"])

        # JPEG bloom (aggressive)
        src = jpeg_bloom(src, intensity=settings["jpeg_intensity"], passes=settings["jpeg_passes"])

        # Save
        out_path = os.path.join(out_folder, f"frame_{out_idx:05d}.png")
        src.convert("RGB").save(out_path)
        prev_img = src.copy()
        out_idx += 1

        if out_idx % 50 == 0:
            print(f"Processed {out_idx} frames...")

    print(f"Finished processing {out_idx} frames into {out_folder}")

# -------------------------
# Assemble video
# -------------------------
def assemble_video_from_frames(frame_folder: str, fps: int, tmp_video: str):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_folder, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium",
        tmp_video
    ]
    rc, out, err = run(cmd)
    if rc != 0:
        print(err)
        raise RuntimeError("ffmpeg failed to assemble video")

# -------------------------
# Main CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(prog="extreme_datamosh")
    # CHANGED: arguments are now optional with defaults
    parser.add_argument("--v1", default="input.mp4", help="Primary input (video or image)")
    parser.add_argument("--v2", default="image2.mp4", help="Secondary input (video or image)")
    parser.add_argument("--out", default="output_horrifying_mosh.mp4")
    parser.add_argument("--fps", type=int, default=DEFAULTS["fps"])
    parser.add_argument("--clean", action="store_true", help="Remove temp folders after run")
    args = parser.parse_args()

    v1 = args.v1
    v2 = args.v2
    final_out = args.out
    fps = args.fps

    # verify files exist
    for p in (v1, v2):
        if not os.path.exists(p):
            print(f"ERROR: input not found: {p}")
            sys.exit(2)

    # inspect with ffprobe
    def is_video(path):
        rc, out, err = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type", "-of", "default=nokey=1:noprint_wrappers=1", path])
        return rc == 0 and out.strip() != ""

    v1_is_video = is_video(v1)
    v2_is_video = is_video(v2)

    print(f"v1_is_video={v1_is_video}, v2_is_video={v2_is_video}")

    # temp folders
    t_v1 = "frames_v1"
    t_v2 = "frames_v2"
    t_out = "frames_out"
    t_audio = "glitched_audio.aac"
    temp_vid = "temp_video.mp4"

    # cleanup start
    for d in (t_v1, t_v2, t_out, t_audio, temp_vid):
        if os.path.exists(d):
            if os.path.isdir(d):
                shutil.rmtree(d)
            else:
                try:
                    os.remove(d)
                except Exception:
                    pass

    try:
        # Extract frames v1
        if v1_is_video:
            extract_frames_with_ffmpeg(v1, t_v1, fps)
        else:
            if not check_is_image(v1):
                raise RuntimeError(f"{v1} is neither a video nor a valid image.")
            image_fallback_generate_frames(v1, t_v1, fps, duration_secs=3)

        # Extract frames v2
        if v2_is_video:
            extract_frames_with_ffmpeg(v2, t_v2, fps)
        else:
            if not check_is_image(v2):
                raise RuntimeError(f"{v2} is neither a video nor a valid image.")
            image_fallback_generate_frames(v2, t_v2, fps, duration_secs=3)

        # Try audio (from v1)
        has_audio = extract_and_glitch_audio(v1, t_audio)

        # Collect sorted frames
        files_v1 = sorted([os.path.join(t_v1, f) for f in os.listdir(t_v1) if f.lower().endswith(".png")])
        files_v2 = sorted([os.path.join(t_v2, f) for f in os.listdir(t_v2) if f.lower().endswith(".png")])

        if not files_v1 and not files_v2:
            raise RuntimeError("No frames available after extraction.")

        # Process effects
        process(files_v1, files_v2, t_out, DEFAULTS)

        # Assemble
        assemble_video_from_frames(t_out, fps, temp_vid)

        # Mux Audio
        if has_audio and os.path.exists(t_audio):
            rc, out, err = run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", temp_vid, "-i", t_audio,
                "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", final_out
            ])
            if rc != 0:
                print("Failed to mux audio. Error from ffmpeg:", err)
                raise RuntimeError("ffmpeg failed to mux audio")
            else:
                os.remove(temp_vid)
        else:
            os.replace(temp_vid, final_out)

        print(f"Success. Output saved to: {final_out}")

    except Exception as e:
        print("CRITICAL ERROR:", str(e))
        sys.exit(1)
    finally:
        if args.clean:
            for d in (t_v1, t_v2, t_out, t_audio, temp_vid):
                if os.path.exists(d):
                    if os.path.isdir(d):
                        shutil.rmtree(d)
                    else:
                        try:
                            os.remove(d)
                        except Exception:
                            pass

if __name__ == "__main__":
    main()

