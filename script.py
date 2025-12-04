#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# HYPER-LUDOVICO: Procedural Decay Engine
# -----------------------------------------------------------------------------
# Generates structured, narrative glitch art with stateful effects.
# 
# Features:
# - Stateful Glitch Entities (Duration-based effects)
# - "Ghosting" Feedback Loop (Long-term frame persistence)
# - Adaptive Palette Crushing
# - Temporal Stuttering
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
import numpy as np
import cv2

# -------------------- DEFAULT CONFIG --------------------
SETTINGS = {
    "fps": 24,
    "internal_res": (640, 360), # Low res = Crunchier glitches + Speed
    
    # --- MOSH PHYSICS ---
    "mosh_threshold": 12,       # Sensitivity to motion (Lower = More Smear)
    "drag_momentum": 0.92,      # How much smear persists (0.0 - 1.0)
    "block_size": 16,           # Macroblock size
    
    # --- GHOSTING (Long Exposure) ---
    "ghost_decay": 0.85,        # How fast the "ghost" layer fades (Lower = Faster)
    "ghost_mix": 0.3,           # Visibility of the ghost layer
    
    # --- CHAOS PROBABILITIES (Per Frame spawn chance) ---
    "prob_stutter": 0.02,       # Chance to freeze/stutter time
    "prob_melt": 0.015,         # Chance to start a "Liquid Melt" event
    "prob_crush": 0.01,         # Chance to start a "Palette Crush" event
    "prob_invert": 0.005,       # Chance to flash negative
}

# -------------------- HELPERS --------------------

def run_ffmpeg(cmd):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error: {e.stderr}", file=sys.stderr)
        raise

def get_audio_filter(intensity=1.0):
    """Generates a randomized, abrasive audio chain."""
    # Randomly select a "flavor" of audio destruction
    flavor = random.choice(['void', 'shred', 'demon'])
    
    if flavor == 'void':
        return "aecho=0.8:0.9:1000:0.3, lowpass=f=400, volume=1.2"
    elif flavor == 'shred':
        return f"acrusher=level_in=8:level_out=18:bits={random.randint(2,6)}:mode=log:aa=1, tremolo=f=12:d=0.8"
    else: # demon
        return "asetrate=44100*0.8, aresample=44100, aecho=0.8:0.8:40:0.5"

# -------------------- VISUAL FX ENGINES --------------------

def quantized_optical_flow(prev_gray, curr_gray, block_size=16):
    """Calculates Optical Flow snapped to a grid (Block Drag)."""
    h, w = prev_gray.shape
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Downscale flow to block size to simulate macroblocks
    small_flow = cv2.resize(flow, (w // block_size, h // block_size), interpolation=cv2.INTER_NEAREST)
    block_flow = cv2.resize(small_flow, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create remap coordinates
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (map_x - block_flow[..., 0]).astype(np.float32)
    map_y = (map_y - block_flow[..., 1]).astype(np.float32)
    return map_x, map_y

def liquid_melt(img, time_step, intensity=10):
    """
    Applies a sine-wave displacement to create a melting effect.
    Vectorized using remap for speed.
    """
    h, w, _ = img.shape
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Create wavy displacement
    displacement = np.sin(map_y / 20.0 + time_step) * intensity
    map_x = (map_x + displacement).astype(np.float32)
    
    return cv2.remap(img, map_x, map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)

def palette_crush(img, mode='neon'):
    """
    Remaps image colors to a restricted, aggressive palette.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    if mode == 'neon':
        # Boost Saturation, Quantize Hue to Cyan/Magenta
        h = np.where(h > 90, 150, 30).astype(np.uint8) # Magenta / Yellow
        s = np.clip(s * 1.5, 0, 255).astype(np.uint8)
    elif mode == 'rot':
        # Dark Red/Green
        h = np.where(h > 100, 60, 0).astype(np.uint8) 
        v = np.clip(v * 0.8, 0, 255).astype(np.uint8)
    
    final = cv2.merge([h, s, v])
    return cv2.cvtColor(final, cv2.COLOR_HSV2BGR)

# -------------------- GLITCH STATE MACHINE --------------------

class GlitchDirector:
    """
    Manages the 'Story' of the glitches. 
    Spawns events that persist over time.
    """
    def __init__(self, settings):
        self.s = settings
        self.active_events = {}
        self.frame_count = 0
        self.palette_mode = None
        self.stutter_buffer = None
        self.stutter_frames_left = 0

    def update(self):
        self.frame_count += 1
        
        # 1. Spawn Events
        if random.random() < self.s['prob_melt'] and 'melt' not in self.active_events:
            # Spawn Melt: Lasts 20-60 frames
            self.active_events['melt'] = {'start': self.frame_count, 'duration': random.randint(20, 60)}
            
        if random.random() < self.s['prob_crush'] and 'crush' not in self.active_events:
            # Spawn Crush: Lasts 10-40 frames
            self.active_events['crush'] = {'start': self.frame_count, 'duration': random.randint(10, 40)}
            self.palette_mode = random.choice(['neon', 'rot'])

        if random.random() < self.s['prob_stutter'] and self.stutter_frames_left == 0:
            # Start Stutter
            self.stutter_frames_left = random.randint(3, 8)
            self.stutter_buffer = None # Will grab next frame

        # 2. Cleanup Dead Events
        ended = []
        for name, data in self.active_events.items():
            if self.frame_count > data['start'] + data['duration']:
                ended.append(name)
        for name in ended:
            del self.active_events[name]

    def apply(self, img):
        out = img
        
        # --- STUTTER (Time Freeze) ---
        if self.stutter_frames_left > 0:
            if self.stutter_buffer is None:
                self.stutter_buffer = out.copy()
            
            # Degrade the stutter buffer slightly each frame (Bitrot)
            noise = np.random.normal(0, 5, self.stutter_buffer.shape).astype(np.uint8)
            self.stutter_buffer = cv2.add(self.stutter_buffer, noise)
            
            out = self.stutter_buffer
            self.stutter_frames_left -= 1
            return out # Return early, freeze frame ignores other logic

        # --- MELT ---
        if 'melt' in self.active_events:
            # Calculate intensity curve (Ease in/out)
            elapsed = self.frame_count - self.active_events['melt']['start']
            duration = self.active_events['melt']['duration']
            # Parabolic curve for intensity (starts 0, peaks, ends 0)
            intensity = 20 * math.sin((elapsed / duration) * math.pi)
            out = liquid_melt(out, self.frame_count * 0.1, intensity)

        # --- PALETTE CRUSH ---
        if 'crush' in self.active_events:
            out = palette_crush(out, self.palette_mode)

        # --- INSTANT FLASHES ---
        if random.random() < self.s['prob_invert']:
            out = cv2.bitwise_not(out)

        return out

# -------------------- MAIN PROCESSOR --------------------

def process_video(v1_path, v2_path, out_path, settings):
    print(">>> INITIALIZING HYPER-LUDOVICO ENGINE...")
    
    tmp = tempfile.mkdtemp()
    try:
        # 1. EXTRACT FRAMES
        w, h = settings['internal_res']
        print(f" -> Extracting frames ({w}x{h})...")
        
        run_ffmpeg(["ffmpeg", "-y", "-i", v1_path, "-vf", f"fps={settings['fps']},scale={w}:{h}", f"{tmp}/f1_%05d.bmp"])
        run_ffmpeg(["ffmpeg", "-y", "-i", v2_path, "-vf", f"fps={settings['fps']},scale={w}:{h}", f"{tmp}/f2_%05d.bmp"])
        
        f1_list = sorted([f for f in os.listdir(tmp) if f.startswith("f1_")])
        f2_list = sorted([f for f in os.listdir(tmp) if f.startswith("f2_")])
        
        if not f1_list: raise Exception("Extraction failed.")
        
        total_frames = len(f1_list) + len(f2_list)
        print(f" -> Processing {total_frames} frames...")
        
        # State & Director
        director = GlitchDirector(settings)
        prev_processed = None
        prev_gray = None
        ghost_buffer = None # Stores the accumulated "Ghost" layer
        
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        
        for i in range(total_frames):
            if i % 20 == 0: print(f"Rendering: {i}/{total_frames}...", end='\r')
            
            # --- SOURCE SELECTION ---
            if i < len(f1_list):
                curr = cv2.imread(os.path.join(tmp, f1_list[i]))
                # Transition Logic
                left = len(f1_list) - i
                trans_len = int(len(f1_list) * 0.2) # 20% transition
                if left < trans_len and f2_list:
                    idx2 = int((1 - (left/trans_len)) * (len(f2_list)-1))
                    alt = cv2.imread(os.path.join(tmp, f2_list[idx2]))
                    if alt is not None: curr = cv2.absdiff(curr, alt)
            else:
                idx = i - len(f1_list)
                if idx >= len(f2_list): break
                curr = cv2.imread(os.path.join(tmp, f2_list[idx]))

            if curr is None: continue
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            
            # --- PIXEL DRAG (The Mosh) ---
            if prev_processed is None:
                mosh_layer = curr
            else:
                map_x, map_y = quantized_optical_flow(prev_gray, curr_gray, settings['block_size'])
                warped_prev = cv2.remap(prev_processed, map_x, map_y, interpolation=cv2.INTER_NEAREST)
                
                diff = cv2.absdiff(curr, warped_prev)
                diff_mag = np.sum(diff, axis=2)
                
                mask = diff_mag < settings['mosh_threshold']
                mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                
                mosh_layer = np.where(mask_3ch, 
                                 cv2.addWeighted(warped_prev, settings['drag_momentum'], curr, 1-settings['drag_momentum'], 0), 
                                 curr)

            # --- GHOSTING (The Feedback Loop) ---
            if ghost_buffer is None:
                ghost_buffer = np.float32(mosh_layer)
            else:
                # Accumulate current mosh layer into ghost buffer
                cv2.accumulateWeighted(mosh_layer, ghost_buffer, 1.0 - settings['ghost_decay'])
            
            # Blend Ghost back onto Mosh
            ghost_uint8 = cv2.convertScaleAbs(ghost_buffer)
            final = cv2.addWeighted(mosh_layer, 1.0, ghost_uint8, settings['ghost_mix'], 0)

            # --- DIRECTOR EFFECTS (Melt, Crush, Stutter) ---
            director.update()
            final = director.apply(final)

            # Save & Update State
            cv2.imwrite(f"{out_dir}/frame_{i:05d}.bmp", final)
            prev_processed = final.copy()
            prev_gray = curr_gray.copy()

        # ---------------- ASSEMBLY ----------------
        print("\n -> Encoding...")
        temp_vid = os.path.join(tmp, "temp_render.mp4")
        
        run_ffmpeg([
            "ffmpeg", "-y", "-framerate", str(settings['fps']),
            "-i", f"{out_dir}/frame_%05d.bmp",
            "-vf", "scale=1280:720:flags=neighbor", 
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            temp_vid
        ])
        
        # Audio
        has_audio = False
        try:
            run_ffmpeg(["ffmpeg", "-y", "-i", v1_path, "-vn", "-c:a", "aac", f"{tmp}/audio.aac"])
            has_audio = True
        except: pass
        
        if has_audio:
            print(" -> Audio Destruction...")
            # Use random intensity for filter generation
            flt = get_audio_filter()
            run_ffmpeg([
                "ffmpeg", "-y", "-i", temp_vid, "-i", f"{tmp}/audio.aac",
                "-af", flt,
                "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", out_path
            ])
        else:
            shutil.move(temp_vid, out_path)
            
        print(f"COMPLETE: {out_path}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# -------------------- CLI --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-Ludovico Datamosher")
    parser.add_argument("--v1", required=True)
    parser.add_argument("--v2", required=True)
    parser.add_argument("--out", default="output_hyper.mp4")
    
    # Modifiers
    parser.add_argument("--drag", type=float, default=1.0, help="Pixel drag intensity (default: 1.0)")
    parser.add_argument("--chaos", type=float, default=1.0, help="Glitch frequency (default: 1.0)")
    
    args = parser.parse_args()
    
    # Scale Settings
    s = SETTINGS.copy()
    s['mosh_threshold'] = int(s['mosh_threshold'] * args.drag)
    s['drag_momentum'] = min(0.99, s['drag_momentum'] * (1 + (args.drag - 1) * 0.05))
    
    for k in ['prob_stutter', 'prob_melt', 'prob_crush', 'prob_invert']:
        s[k] = min(1.0, s[k] * args.chaos)
        
    process_video(args.v1, args.v2, args.out, s)
