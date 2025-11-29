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
    # Upscaled at the end.
    "internal_res": (640, 360), 
    
    # --- MOSH PHYSICS ---
    "mosh_threshold": 15,       # Higher = More pixels refuse to update (More Drag)
    "drag_momentum": 0.90,      # 0.0-1.0: How much of the previous frame persists
    "block_size": 16,           # Size of the "glitch blocks" (8, 16, 32)
    
    # --- CHAOS PROBABILITIES (Per Frame) ---
    "prob_sort": 0.15,          # Pixel Sorting
    "prob_bloom": 0.10,         # JPEG Compression/Fry
    "prob_invert": 0.02,        # Negative Flash
    "prob_pulse": 0.05,         # RGB Channel Split Pulse
    
    # --- INTENSITIES ---
    "sort_magnitude": 0.5,      # How much of the screen gets sorted
    "bloom_power": 4,           # Compression factor (Higher = more destroyed)
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
        run_ffmpeg(["ffmpeg", "-y", "-i", v2_path, "-vf", f"fps={settings['fps']},scale={w}:{h}",

