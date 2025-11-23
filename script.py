#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import random
import io
import numpy as np
from PIL import Image, ImageChops

# -------------------------
# Configuration
# -------------------------
DEFAULTS = {
    "fps": 24,
    "transition_fraction": 0.20,
    "mosh_threshold": 36.0,
    "block_size": 16,
    "block_spread": 2,
    "frame_dup_chance": 0.12,
    "jpeg_intensity": 0.85,
    "jpeg_passes": 2,
    "pixel_sort_chance": 0.6,
    "channel_shift_chance": 0.6,
    "pixel_sort_band_div": 12,
}

# -------------------------
# Utilities
# -------------------------
def run(cmd, capture_output=True):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE if capture_output else None,
                          stderr=subprocess.PIPE if capture_output else None,
                          text=True)
    out = proc.stdout if proc.stdout is not None else ""
    err = proc.stderr if proc.stderr is not None else ""
    return proc.returncode, out, err

def check_is_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False

def check_file_sanity(path):
    """Debugs if the file is an LFS pointer."""
    size = os.path.getsize(path)
    print(f"DEBUG: Checking '{path}' - Size: {size} bytes")
    if size < 2000:
        print(f"WARNING: '{path}' is dangerously small ({size} bytes).")
        print("It is likely a Git LFS pointer, not a real video.")
        try:
            with open(path, 'r', errors='ignore') as f:
                head = f.read(100)
                print(f"File content preview: {head}")
        except:
            pass

# -------------------------
# Robust frame extraction
# -------------------------
def extract_frames_with_ffmpeg(input_path, out_folder, fps):
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

    produced = [f for f in os.listdir(out_folder) if f.lower().endswith(".png")]
    if not produced:
        raise RuntimeError(f"ffmpeg reported success but produced 0 frames in {out_folder}")

def image_fallback_generate_frames(image_path, out_folder, fps, duration_secs=3):
    os.makedirs(out_folder, exist_ok=True)
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        count = fps * duration_secs
        for i in range(count):
            im.save(os.path.join(out_folder, f"frame_{i:05d}.png"))

# -------------------------
# Audio processing
# -------------------------
def extract_and_glitch_audio(input_path, out_audio):
    temp = "temp_raw_audio.aac"
    rc, _, _ = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", input_path, "-vn", "-c:a", "aac", temp])
    if rc != 0 or not os.path.exists(temp):
        if os.path.exists(temp): os.remove(temp)
        return False
    
    rc2, _, _ = run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", temp,
        "-af", "aecho=0.8:0.9:1000:0.3,vibrato=f=5.0:d=0.4,atempo=0.92",
        "-c:a", "aac", out_audio
    ])
    if os.path.exists(temp): os.remove(temp)
    return rc2 == 0 and os.path.exists(out_audio)

# -------------------------
# Visual Effects
# -------------------------
def jpeg_bloom(img, intensity=0.7, passes=1):
    if random.random() > intensity: return img
    out = img.convert("RGB")
    for _ in range(max(1, passes)):
        buf = io.BytesIO()
        q = random.randint(6, 20)
        out.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    return out

def channel_shift(img, max_offset=8, chance=0.6):
    if random.random() > chance: return img
    r, g, b = img.split()
    def ofs(band):
        dx = random.randint(-max_offset, max_offset)
        dy = random.randint(-max_offset, max_offset)
        return ImageChops.offset(band, dx, dy)
    return Image.merge("RGB", (ofs(r), ofs(g), ofs(b)))

def pixel_sort(img, chance=0.6, band_div=12):
    if random.random() > chance: return img
    arr = np.array(img)
    h, w, _ = arr.shape
    band_h = random.randint(1, max(1, h // band_div))
    start = random.randint(0, max(0, h - band_h))
    end = start + band_h
    for r in range(start, end):
        row = arr[r].copy()
        lum = np.dot(row[..., :3], [0.299, 0.587, 0.114])
        idx = np.argsort(lum)
        if random.random() > 0.5: idx = idx[::-1]
        arr[r] = row[idx]
    return Image.fromarray(arr)

def macroblock_smear(current, previous, block_size=16, threshold=36.0, spread=1):
    if previous is None: return current
    if current.size != previous.size: previous = previous.resize(current.size)
    
    cur = np.array(current).astype(np.int32)
    prev = np.array(previous).astype(np.int32)
    h, w, c = cur.shape
    bs = max(4, int(block_size))
    
    # Pad to block alignment
    pad_h = (bs - (h % bs)) % bs
    pad_w = (bs - (w % bs)) % bs
    if pad_h or pad_w:
        cur = np.pad(cur, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        prev = np.pad(prev, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        
    H, W, _ = cur.shape
    bh, bw = H // bs, W // bs
    
    cur_blocks = cur.reshape(bh, bs, bw, bs, c)
    prev_blocks = prev.reshape(bh, bs, bw, bs, c)
    
    # Difference calc
    diff_blocks = np.sum(np.abs(cur_blocks - prev_blocks), axis=(1, 3, 4))
    avg_blocks = diff_blocks / (bs * bs)
    mask = avg_blocks < threshold
    
    if spread > 0:
        expanded = mask.copy()
        # Simple expansion loop
        for y in range(bh):
            for x in range(bw):
                if mask[y, x]:
                    y0, y1 = max(0, y-spread), min(bh, y+spread+1)
                    x0, x1 = max(0, x-spread), min(bw, x+spread+1)
                    expanded[y0:y1, x0:x1] = True
        mask = expanded
        
    out = cur.copy()
    # Apply smear
    for by in range(bh):
        y0, y1 = by * bs, (by + 1) * bs
        for bx in range(bw):
            if mask[by, bx]:
                x0, x1 = bx * bs, (bx + 1) * bs
                out[y0:y1, x0:x1] = prev[y0:y1, x0:x1]
                
    if pad_h or pad_w: out = out[:h, :w]
    return Image.fromarray(out.astype("uint8"))

# -------------------------
# Processing Pipeline
# -------------------------
def process(v1_frames, v2_frames, out_folder, settings):
    os.makedirs(out_folder, exist_ok=True)
    target_size = Image.open(v1_frames[0] if v1_frames else v2_frames[0]).size
    prev_img = None
    out_idx = 0
    total_count = len(v1_frames) + len(v2_frames)
    print(f"Processing ~{total_count} frames at {target_size}...")

    def open_rgb(path): return Image.open(path).convert("RGB")

    for i in range(total_count):
        # Frame selection
        if i < len(v1_frames):
            src = open_rgb(v1_frames[i])
            frames_left = len(v1_frames) - i
            transition_len = max(1, int(len(v1_frames) * settings["transition_fraction"]))
            if frames_left <= transition_len and len(v2_frames) > 0:
                tnorm = 1.0 - (frames_left / float(max(1, transition_len)))
                v2_idx = min(len(v2_frames)-1, int(tnorm * (len(v2_frames)-1)))
                img_v2 = open_rgb(v2_frames[v2_idx]).resize(src.size)
                src = ImageChops.add(ImageChops.difference(src, img_v2), img_v2, scale=1.2, offset=-8)
        else:
            v2_idx = i - len(v1_frames)
            if v2_idx >= len(v2_frames): break
            src = open_rgb(v2_frames[v2_idx])

        if src.size != target_size: src = src.resize(target_size)

        # Effects
        if prev_img is not None and random.random() < settings["frame_dup_chance"]:
            prev_img.save(os.path.join(out_folder, f"frame_{out_idx:05d}.png"))
            out_idx += 1
            
        src = macroblock_smear(src, prev_img, settings["block_size"], settings["mosh_threshold"], settings["block_spread"])
        src = pixel_sort(src, settings["pixel_sort_chance"], settings["pixel_sort_band_div"])
        src = channel_shift(src, 8, settings["channel_shift_chance"])
        src = jpeg_bloom(src, settings["jpeg_intensity"], settings["jpeg_passes"])

        src.save(os.path.join(out_folder, f"frame_{out_idx:05d}.png"))
        prev_img = src.copy()
        out_idx += 1
        
        if out_idx % 50 == 0: print(f"Processed {out_idx} frames...", end='\r')

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", default="input.mp4")
    parser.add_argument("--v2", default="image2.mp4")
    parser.add_argument("--out", default="output_horrifying_mosh.mp4")
    parser.add_argument("--fps", type=int, default=DEFAULTS["fps"])
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    # Check Sanity
    check_file_sanity(args.v1)
    check_file_sanity(args.v2)

    # Check if files are video
    def is_video(path):
        rc, out, _ = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type", "-of", "default=nokey=1:noprint_wrappers=1", path])
        return rc == 0 and out.strip() != ""

    v1_is_video = is_video(args.v1)
    v2_is_video = is_video(args.v2)
    print(f"v1_is_video={v1_is_video}, v2_is_video={v2_is_video}")

    # Paths
    t_v1, t_v2, t_out = "frames_v1", "frames_v2", "frames_out"
    t_audio, temp_vid = "glitched_audio.aac", "temp_video.mp4"
    
    # Cleanup
    for d in (t_v1, t_v2, t_out, t_audio, temp_vid):
        if os.path.exists(d): 
            if os.path.isdir(d): shutil.rmtree(d)
            else: os.remove(d)

    # Execute
    try:
        if v1_is_video: extract_frames_with_ffmpeg(args.v1, t_v1, args.fps)
        else: 
            if not check_is_image(args.v1): raise RuntimeError(f"CRITICAL ERROR: {args.v1} is neither a video nor a valid image.")
            image_fallback_generate_frames(args.v1, t_v1, args.fps)

        if v2_is_video: extract_frames_with_ffmpeg(args.v2, t_v2, args.fps)
        else:
            if not check_is_image(args.v2): raise RuntimeError(f"CRITICAL ERROR: {args.v2} is neither a video nor a valid image.")
            image_fallback_generate_frames(args.v2, t_v2, args.fps)

        has_audio = extract_and_glitch_audio(args.v1, t_audio)
        
        files_v1 = sorted([os.path.join(t_v1, f) for f in os.listdir(t_v1) if f.endswith(".png")])
        files_v2 = sorted([os.path.join(t_v2, f) for f in os.listdir(t_v2) if f.endswith(".png")])
        
        process(files_v1, files_v2, t_out, DEFAULTS)
        
        # Assemble
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(args.fps),
             "-i", os.path.join(t_out, "frame_%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", temp_vid])
        
        # Mux
        if has_audio and os.path.exists(t_audio):
            run(["ffmpeg", "-y", "-i", temp_vid, "-i", t_audio, "-c:v", "copy", "-c:a", "aac", "-shortest", args.out])
        else:
            os.replace(temp_vid, args.out)
            
        print(f"Success! {args.out}")

    except Exception as e:
        print(e)
        sys.exit(1)
    finally:
        if args.clean:
            shutil.rmtree(t_v1, ignore_errors=True)
            shutil.rmtree(t_v2, ignore_errors=True)
            shutil.rmtree(t_out, ignore_errors=True)

if __name__ == "__main__":
    main()
