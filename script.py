#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# HYPER-LUDOVICO: FAST ABSTRACT ENGINE
# -----------------------------------------------------------------------------
# Speed: ~10x faster (Processes at 240p, Upscales to 720p)
# Style: Abstract, melting P-frames, blocky glitches, color drifts.
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

# -------------------- CONFIG --------------------
# Base settings - randomized at runtime
SETTINGS = {
    "fps": 24,
    "internal_res": (426, 240), # 240p = Blocky & Insanely Fast
    
    # Physics
    "threshold": 25,      # High = More Drag/Smear
    "decay": 0.98,        # How long trails last (0.0 - 1.0)
    "zoom_drift": 1.01,   # The "melting" forward movement
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

def fast_color_jitter(img, intensity=10):
    """Drifts RGB channels using array slicing (Instant)."""
    h, w, c = img.shape
    dx = random.randint(-intensity, intensity)
    dy = random.randint(-intensity, intensity)
    
    # Create output
    out = img.copy()
    
    # Roll channels (Cyclic shift) - Super fast
    # Blue Channel Drift
    out[:, :, 0] = np.roll(out[:, :, 0], dx, axis=1)
    out[:, :, 0] = np.roll(out[:, :, 0], dy, axis=0)
    
    return out

def fast_pixel_shuffle(img, intensity=0.1):
    """Takes random blocks and shuffles them. Cheaper than sorting."""
    h, w, c = img.shape
    if random.random() > intensity: return img
    
    # Extract a random slice
    y = random.randint(0, h-20)
    h_slice = random.randint(10, 50)
    
    # Reverse the slice horizontally
    img[y:y+h_slice, :, :] = img[y:y+h_slice, ::-1, :]
    return img

def apply_feedback_warp(prev_img, zoom, rotate):
    """
    The Core "Melt" Mechanic. 
    Instead of flow, we just slightly zoom/rotate the previous frame.
    """
    h, w = prev_img.shape[:2]
    center = (w // 2, h // 2)
    
    # Create Affine Matrix
    M = cv2.getRotationMatrix2D(center, rotate, zoom)
    
    # Apply Warp (Linear for smoothness in trails)
    return cv2.warpAffine(prev_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

# -------------------- PROCESSOR --------------------

def process_video(v1_path, v2_path, out_path, drag_mult, chaos_mult):
    # 1. RANDOMIZE SESSION
    seed = random.randint(0, 9999)
    random.seed(seed)
    print(f">>> SEED: {seed}")
    
    # Mutate Settings
    s = SETTINGS.copy()
    s['threshold'] = int(s['threshold'] * drag_mult)
    s['zoom_drift'] = 1.0 + (random.uniform(-0.02, 0.03) * chaos_mult)
    rotate_drift = random.uniform(-1, 1) * chaos_mult
    
    # Randomize Color Mapping
    color_map = [0, 1, 2] # BGR
    if random.random() < (0.3 * chaos_mult):
        random.shuffle(color_map) # Randomize channel order (Psychedelic)
    
    print(f" -> Zoom: {s['zoom_drift']:.3f} | Rotate: {rotate_drift:.2f}")
    print(f" -> Channels: {color_map}")

    # Temp workspace
    tmp = tempfile.mkdtemp()
    
    try:
        # 1. EXTRACT (Fast BMP sequence at 240p)
        w, h = s['internal_res']
        print(" -> Extracting low-res frames...")
        
        # We define the command list properly to avoid syntax errors
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
        
        if not f1_files: raise Exception("Input 1 failed to extract.")
        
        total_frames = len(f1_files) + len(f2_files)
        
        # 2. FAST PROCESSING LOOP
        prev_frame = None
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        
        # Pre-calculate threshold mask for speed
        thresh_val = s['threshold']
        decay_val = s['decay']
        
        print(f" -> Moshing {total_frames} frames...")
        
        for i in range(total_frames):
            if i % 50 == 0: print(f"    Frame {i}/{total_frames}...", end='\r')
            
            # --- Source Logic (Linear Mix) ---
            if i < len(f1_files):
                curr = cv2.imread(os.path.join(tmp, f1_files[i]))
                # Check transition
                if f2_files and i > len(f1_files) * 0.8:
                    # Quick Blend at end of clip 1
                    idx2 = random.randint(0, len(f2_files)-1)
                    alt = cv2.imread(os.path.join(tmp, f2_files[idx2]))
                    curr = cv2.addWeighted(curr, 0.7, alt, 0.3, 0)
            else:
                idx = i - len(f1_files)
                if idx >= len(f2_files): break
                curr = cv2.imread(os.path.join(tmp, f2_files[idx]))

            if curr is None: continue
            
            # Apply Color Scramble (Psychedelic)
            curr = curr[:, :, color_map]

            # --- THE ABSTRACT MOSH (Vectorized) ---
            if prev_frame is None:
                final = curr
            else:
                # 1. Mutate Previous Frame (The "Melt")
                # Instead of optical flow, we just Zoom/Rotate the ghost trail
                ghost = apply_feedback_warp(prev_frame, s['zoom_drift'], rotate_drift)
                
                # 2. Calculate Difference (Vectorized)
                # Euclidean distance in color space
                diff = cv2.absdiff(curr, ghost)
                diff_mag = np.sum(diff, axis=2)
                
                # 3. Create Mask (Where did the image NOT change much?)
                # This keeps the ghost trail in static areas
                mask = diff_mag < thresh_val
                
                # 4. Composite (Fast Boolean Indexing)
                # If mask is true, use Ghost. Else use Current.
                mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                
                # Blend ghost with decay to prevent total whiteout
                ghost_decayed = cv2.addWeighted(ghost, decay_val, curr, 1-decay_val, 0)
                
                final = np.where(mask_3d, ghost_decayed, curr)

            # --- FAST CHAOS ---
            # 1. RGB Drift (cheap)
            if random.random() < (0.1 * chaos_mult):
                final = fast_color_jitter(final, int(5 * chaos_mult))
                
            # 2. Negative Flash (cheap)
            if random.random() < (0.02 * chaos_mult):
                final = cv2.bitwise_not(final)
                
            # 3. Slice Shuffle (cheap)
            if random.random() < (0.05 * chaos_mult):
                final = fast_pixel_shuffle(final)

            # Save as BMP (Fastest IO)
            cv2.imwrite(f"{out_dir}/frame_{i:05d}.bmp", final)
            prev_frame = final

        # 3. RENDER & UPSCALING
        print("\n -> Rendering...")
        temp_vid = os.path.join(tmp, "temp.mp4")
        
        # Upscale 240p -> 720p using Nearest Neighbor
        # This keeps the blocks sharp and creates the "Digital Artifact" look
        cmd_render = [
            "ffmpeg", "-y", "-framerate", str(s['fps']),
            "-i", f"{out_dir}/frame_%05d.bmp",
            "-vf", "scale=1280:720:flags=neighbor",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p",
            temp_vid
        ]
        subprocess.run(cmd_render, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 4. AUDIO DESTRUCTION
        print(" -> Muxing Audio...")
        try:
            # Extract
            subprocess.run(["ffmpeg", "-y", "-i", v1_path, "-vn", "-c:a", "aac", f"{tmp}/audio.aac"], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Glitch
            af = get_abrasive_audio_filter()
            cmd_mux = [
                "ffmpeg", "-y", "-i", temp_vid, "-i", f"{tmp}/audio.aac",
                "-af", af, "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", out_path
            ]
            subprocess.run(cmd_mux, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            # Fallback if no audio
            shutil.move(temp_vid, out_path)

        print(f"DONE: {out_path}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", required=True)
    parser.add_argument("--v2", required=True)
    parser.add_argument("--out", default="output.mp4")
    parser.add_argument("--drag", type=float, default=1.0)
    parser.add_argument("--chaos", type=float, default=1.0)
    args = parser.parse_args()
    
    process_video(args.v1, args.v2, args.out, args.drag, args.chaos)
