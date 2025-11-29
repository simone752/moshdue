#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# NEO-LUDOVICO: Vectorized Datamosh Engine
# -----------------------------------------------------------------------------
# Implements "Block-Quantized Optical Flow" to simulate P-Frame corruption.
# Fast, abrasive, and highly configurable.
#
# Usage:
#   python script.py --v1 input.mp4 --v2 overlay.mp4 --drag 2.0 --chaos 1.5
# -----------------------------------------------------------------------------

import os
import sys
import shutil
import subprocess
import random
import argparse
import tempfile
import time
import math
from collections import deque
import numpy as np
import cv2

# -------------------- DEFAULT CONFIG --------------------
SETTINGS = {
    "fps": 24,
    # Process at 360p for speed + crunchy block aesthetics.
    "internal_res": (640, 360),

    # --- MOSH PHYSICS ---
    "mosh_threshold": 15,
    "drag_momentum": 0.90,
    "block_size": 16,

    ### AGGIUNTO ### - Questo è il FIX necessario
    "transition_frac": 0.10,
    "transition_mode": "diff",   # 'diff', 'xor', 'add'

    # --- CHAOS PROBABILITIES (Per Frame) ---
    "prob_sort": 0.15,
    "prob_bloom": 0.10,
    "prob_invert": 0.02,
    "prob_pulse": 0.05,

    # --- INTENSITIES ---
    "sort_magnitude": 0.5,
    "bloom_power": 4,
}

# -------------------- HELPERS --------------------

def run_ffmpeg(cmd):
    try:
        # Run silently but capture error if it fails
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error: {e.stderr}", file=sys.stderr)
        raise

def get_audio_filter(intensity=1.0):
    """Generates a dynamic, harsh audio filter chain based on intensity."""
    # Bitcrusher + Chopper + Reverb
    bits = max(2, int(10 - (4 * intensity)))
    freq = 5 + (intensity * 10)
    return f"acrusher=level_in=8:level_out=18:bits={bits}:mode=log:aa=1, vibrato=f={freq}:d=0.5, aecho=0.8:0.9:40:0.5"

# -------------------- VISUAL FX ENGINES --------------------

def quantized_optical_flow(prev_gray, curr_gray, block_size=16):
    """
    Calculates Optical Flow but snaps vectors to a grid.
    This creates the "Block Drag" effect seen in video compression errors.
    """
    h, w = prev_gray.shape
    
    # 1. Calculate dense flow (Farneback algorithm)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # 2. Downscale Flow to "Block" resolution (Simulate Macroblocks)
    # We resize the flow map down to (w/16, h/16) then blow it back up.
    # This forces 16x16 chunks of pixels to move together.
    small_flow = cv2.resize(flow, (w // block_size, h // block_size), interpolation=cv2.INTER_NEAREST)
    block_flow = cv2.resize(small_flow, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 3. Create Remap Coordinates
    # Map(x,y) = (x,y) + Flow(x,y)
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (map_x - block_flow[..., 0]).astype(np.float32)
    map_y = (map_y - block_flow[..., 1]).astype(np.float32)
    
    return map_x, map_y

def pixel_sort_slice(img, intensity=0.5):
    """
    Sorts pixels based on brightness, but only on random horizontal slices.
    Using OpenCV/Numpy for speed.
    """
    h, w, c = img.shape
    out = img.copy()
    
    # Number of bands to sort
    num_bands = int(1 + (intensity * 5))
    
    for _ in range(num_bands):
        y = random.randint(0, h-10)
        h_slice = random.randint(2, 50)
        
        # Extract slice
        sl = out[y:y+h_slice, :, :]
        
        # Calculate luminance
        lum = np.sum(sl, axis=2) # Simple sum is faster than correct luminance
        
        # Get sort indices
        indices = np.argsort(lum, axis=1)
        
        # Apply sort (Advanced Numpy indexing)
        # Create a grid of indices for the rows
        rows = np.arange(sl.shape[0])[:, None]
        # Shuffle pixels
        sl[:] = sl[rows, indices]
        
        out[y:y+h_slice, :, :] = sl
        
    return out

def jpeg_crush(img, power=2):
    """
    Simulates bitrate starvation/JPEG artifacts.
    """
    h, w, _ = img.shape
    # Downscale extremely, then upscale with Nearest Neighbor
    factor = max(1, int(power * 4)) # 4x to 16x reduction
    small = cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def rgb_drift(img, offset):
    """
    Splits RGB channels and drifts them apart.
    """
    b, g, r = cv2.split(img)
    h, w = b.shape
    
    # Shift Matrices
    M_r = np.float32([[1, 0, offset], [0, 1, 0]])
    M_b = np.float32([[1, 0, -offset], [0, 1, 0]])
    
    r = cv2.warpAffine(r, M_r, (w, h))
    b = cv2.warpAffine(b, M_b, (w, h))
    
    return cv2.merge([b, g, r])

# -------------------- MAIN PROCESSOR --------------------

def process_video(v1_path, v2_path, out_path, settings):
    print(">>> INITIALIZING NEO-LUDOVICO ENGINE...")
    
    # Create temp workspace
    tmp = tempfile.mkdtemp()
    try:
        # 1. EXTRACT FRAMES (Low Res for Speed + Aesthetic)
        w, h = settings['internal_res']
        print(f" -> Extracting frames to internal buffer ({w}x{h})...")
        
        run_ffmpeg(["ffmpeg", "-y", "-i", v1_path, "-vf", f"fps={settings['fps']},scale={w}:{h}", f"{tmp}/f1_%05d.bmp"])
        run_ffmpeg(["ffmpeg", "-y", "-i", v2_path, "-vf", f"fps={settings['fps']},scale={w}:{h}", f"{tmp}/f2_%05d.bmp"])
        
        f1_list = sorted([f for f in os.listdir(tmp) if f.startswith("f1_")])
        f2_list = sorted([f for f in os.listdir(tmp) if f.startswith("f2_")])
        
        if not f1_list: raise Exception("Input 1 extraction failed.")
        
        total_frames = len(f1_list) + len(f2_list)
        print(f" -> Processing {total_frames} frames...")
        
        # State Variables for the Feedback Loop
        prev_processed = None
        prev_gray = None
        
        # Output directory
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        
        # ---------------- PROCESS LOOP ----------------
        for i in range(total_frames):
            if i % 20 == 0: print(f"Rendering: {i}/{total_frames}...", end='\r')
            
            # A. Source Selection & Transition
            if i < len(f1_list):
                # Video 1
                curr = cv2.imread(os.path.join(tmp, f1_list[i]))
                
                # Check for transition overlap
                left = len(f1_list) - i
                trans_len = int(len(f1_list) * settings['transition_frac'])
                
                if left < trans_len and f2_list:
                    # Transition Logic: Difference Blend
                    idx2 = int((1 - (left/trans_len)) * (len(f2_list)-1))
                    alt = cv2.imread(os.path.join(tmp, f2_list[idx2]))
                    if alt is not None:
                        curr = cv2.absdiff(curr, alt) # Difference blend = inverted colors overlap
            else:
                # Video 2
                idx = i - len(f1_list)
                if idx >= len(f2_list): break
                curr = cv2.imread(os.path.join(tmp, f2_list[idx]))

            if curr is None: continue
            
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            
            # --- THE MOSH LOGIC (Pixel Dragging) ---
            if prev_processed is None:
                final = curr
            else:
                # 1. Calculate Blocky Motion Vectors
                map_x, map_y = quantized_optical_flow(prev_gray, curr_gray, settings['block_size'])
                
                # 2. Warp the PREVIOUS result forward (This is the smear)
                # We warp the *processed* frame, accumulating distortions over time
                warped_prev = cv2.remap(prev_processed, map_x, map_y, interpolation=cv2.INTER_NEAREST)
                
                # 3. Decision: Update or Drag?
                # Calculate difference between the "predicted" pixels and the "actual" new pixels
                diff = cv2.absdiff(curr, warped_prev)
                diff_mag = np.sum(diff, axis=2) # Sum RGB differences
                
                # THE KEY: High threshold means we ignore small changes and keep the smear
                mask = diff_mag < settings['mosh_threshold']
                
                # Apply mask: Where true, use Smear. Where false, use New Data.
                mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                
                # Blend: Mostly smear, but allow some new data to ghost in based on drag_momentum
                final = np.where(mask_3ch, 
                                 cv2.addWeighted(warped_prev, settings['drag_momentum'], curr, 1-settings['drag_momentum'], 0), 
                                 curr)

            # --- CHAOS INJECTION ---
            
            # Random Pixel Sort
            if random.random() < settings['prob_sort']:
                final = pixel_sort_slice(final, settings['sort_magnitude'])
                
            # Random JPEG Bloom
            if random.random() < settings['prob_bloom']:
                final = jpeg_crush(final, settings['bloom_power'])
                
            # Random RGB Split
            if random.random() < settings['prob_pulse']:
                final = rgb_drift(final, random.randint(5, 20))
                
            # Random Invert
            if random.random() < settings['prob_invert']:
                final = cv2.bitwise_not(final)

            # Save
            cv2.imwrite(f"{out_dir}/frame_{i:05d}.bmp", final)
            
            # Feedback Loop update
            prev_processed = final.copy()
            prev_gray = curr_gray.copy()

        # ---------------- ASSEMBLY ----------------
        print("\n -> Encoding Final Output...")
        temp_vid = os.path.join(tmp, "temp_render.mp4")
        
        # Scale back up to 720p using Nearest Neighbor (keep the blocks sharp!)
        run_ffmpeg([
            "ffmpeg", "-y", "-framerate", str(settings['fps']),
            "-i", f"{out_dir}/frame_%05d.bmp",
            "-vf", "scale=1280:720:flags=neighbor", 
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            temp_vid
        ])
        
        # Audio Muxing
        has_audio = False
        try:
            run_ffmpeg(["ffmpeg", "-y", "-i", v1_path, "-vn", "-c:a", "aac", f"{tmp}/audio.aac"])
            has_audio = True
        except: pass
        
        if has_audio:
            print(" -> Applying Audio Destruction...")
            # Calculate total intensity for audio filter
            intensity = (settings['mosh_threshold'] / 50.0) + (settings['prob_sort'] * 2)
            flt = get_audio_filter(intensity)
            
            run_ffmpeg([
                "ffmpeg", "-y", "-i", temp_vid, "-i", f"{tmp}/audio.aac",
                "-af", flt,
                "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", out_path
            ])
        else:
            shutil.move(temp_vid, out_path)
            
        print(f"COMPLETE. Saved to {out_path}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# -------------------- CLI --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neo-Ludovico Datamosher")
    
    # Required
    parser.add_argument("--v1", required=True, help="Main input video")
    parser.add_argument("--v2", required=True, help="Overlay/Transition input")
    parser.add_argument("--out", default="output_neo.mp4")
    
    # Modifiers
    parser.add_argument("--drag", type=float, default=1.0, help="Multiplier for pixel drag intensity (Default: 1.0)")
    parser.add_argument("--chaos", type=float, default=1.0, help="Multiplier for glitch frequency (Default: 1.0)")
    parser.add_argument("--block-size", type=int, default=16, help="Size of macroblocks (8, 16, 32)")
    
    args = parser.parse_args()
    
    # Apply Modifiers to Settings
    s = SETTINGS.copy()
    
    # Drag Multiplier: Increases threshold (more sticky) and momentum
    s['mosh_threshold'] = int(s['mosh_threshold'] * args.drag)
    s['drag_momentum'] = min(0.99, s['drag_momentum'] * (1 + (args.drag - 1) * 0.1))
    
    # Chaos Multiplier: Increases probabilities
    for k in ['prob_sort', 'prob_bloom', 'prob_invert', 'prob_pulse']:
        s[k] = min(1.0, s[k] * args.chaos)
        
    s['block_size'] = args.block_size
    
    process_video(args.v1, args.v2, args.out, s)
