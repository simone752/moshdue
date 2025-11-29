#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import random
import math
import numpy as np
from PIL import Image, ImageChops, ImageOps, ImageEnhance

# -------------------------
# Configuration / Defaults
# -------------------------
DEFAULTS = {
    "fps": 24,
    "transition_fraction": 0.25,
    "block_size": 16,
    
    # Probabilities (0.0 - 1.0) of an event STARTING per frame
    "prob_pixel_sort": 0.05,
    "prob_liquid": 0.02,
    "prob_rgb_split": 0.08,
    "prob_invert": 0.01,
    "prob_stutter": 0.03,
    "prob_mirror": 0.01,
    
    # Intensities
    "intensity_liquid": 20, # How much the screen melts
    "intensity_rgb": 15,    # How far channels drift
}

# -------------------------
# Audio Destruction Engines
# -------------------------
def get_random_audio_filter():
    """Returns a random ffmpeg audio filter chain for variety."""
    chains = [
        # 1. The Deep Void (Echo + Lowpass)
        "aecho=0.8:0.9:1000:0.3,lowpass=f=500",
        # 2. The Broken Modem (Bitcrush + Tremolo)
        "acrusher=level_in=8:level_out=18:bits=4:mode=log:aa=1,tremolo=f=10.0:d=0.7",
        # 3. Alien Radio (Bandpass + Flanger)
        "highpass=f=200,lowpass=f=3000,flanger=delay=0:depth=2:regen=50:speed=2",
        # 4. Demon (Pitch Down + Reverse Reverb)
        "asetrate=44100*0.8,aresample=44100,aecho=0.8:0.8:40:0.5",
        # 5. Fast Chopper
        "vibrato=f=15:d=1"
    ]
    return random.choice(chains)

# -------------------------
# Visual Effects (Numpy Optimized)
# -------------------------

def effect_liquid_displacement(img, intensity, time_step):
    """
    Creates a waving, melting effect using sine waves.
    """
    arr = np.array(img)
    rows, cols, ch = arr.shape
    
    # Create sine wave offsets based on row index and time
    # This creates the "wavy" look
    x_indices = np.arange(cols)
    y_indices = np.arange(rows)
    
    # Shift amount varies by row (y)
    shift_x = (np.sin(y_indices / 20.0 + time_step) * intensity).astype(int)
    
    # Apply row-wise roll (horizontal displacement)
    for i in range(rows):
        arr[i] = np.roll(arr[i], shift_x[i], axis=0)
        
    return Image.fromarray(arr)

def effect_rgb_split(img, intensity):
    """
    Separates R, G, B channels spatially.
    """
    r, g, b = img.split()
    
    # Random drift direction
    x_drift = random.randint(-intensity, intensity)
    y_drift = random.randint(-intensity, intensity)
    
    r = ImageChops.offset(r, x_drift, 0)
    b = ImageChops.offset(b, -x_drift, y_drift)
    
    return Image.merge("RGB", (r, g, b))

def effect_pixel_sort(img, threshold=100):
    """
    Sorts pixels, but only in bright areas to keep structure.
    """
    arr = np.array(img)
    # Convert to grayscale for thresholding
    gray = np.dot(arr[...,:3], [0.299, 0.587, 0.114])
    
    # Find rows that are "bright enough" to sort
    mask = np.mean(gray, axis=1) > threshold
    
    # Only sort selected rows to save time and keep aesthetic
    for i in np.where(mask)[0]:
        # Sort by luminance
        row = arr[i]
        lum = np.dot(row[...,:3], [0.299, 0.587, 0.114])
        arr[i] = row[np.argsort(lum)]
        
    return Image.fromarray(arr)

def macroblock_smear(current, previous, block_size=16, threshold=25):
    """
    The classic P-Frame datamosh simulation.
    """
    if previous is None: return current
    if current.size != previous.size: previous = previous.resize(current.size)
    
    cur_arr = np.array(current).astype(int)
    prev_arr = np.array(previous).astype(int)
    
    # Calculate difference
    diff = np.sum(np.abs(cur_arr - prev_arr), axis=2)
    
    # Blockify the difference (simple average pooling simulation)
    # If a pixel hasn't changed much, we keep the OLD pixel
    mask = diff < threshold
    
    cur_arr[mask] = prev_arr[mask]
    return Image.fromarray(cur_arr.astype('uint8'))

# -------------------------
# Glitch State Manager
# -------------------------
class GlitchManager:
    """
    Manages the 'Story' of the video. 
    Instead of random noise every frame, effects have durations.
    """
    def __init__(self, settings):
        self.s = settings
        self.active_effects = {
            'pixel_sort': 0,
            'liquid': 0,
            'rgb_split': 0,
            'invert': 0,
            'stutter': 0,
            'mirror': 0
        }
        self.time_counter = 0.0
        self.stutter_buffer = None

    def update(self):
        """Called every frame to update state/dice rolls."""
        self.time_counter += 0.2
        
        # Decrement active timers
        for k in self.active_effects:
            if self.active_effects[k] > 0:
                self.active_effects[k] -= 1

        # Roll for new events (only if not already active)
        if self.active_effects['pixel_sort'] == 0 and random.random() < self.s['prob_pixel_sort']:
            self.active_effects['pixel_sort'] = random.randint(5, 20) # Lasts 5-20 frames
            
        if self.active_effects['liquid'] == 0 and random.random() < self.s['prob_liquid']:
            self.active_effects['liquid'] = random.randint(10, 40)
            
        if self.active_effects['rgb_split'] == 0 and random.random() < self.s['prob_rgb_split']:
            self.active_effects['rgb_split'] = random.randint(2, 10)

        if self.active_effects['invert'] == 0 and random.random() < self.s['prob_invert']:
            self.active_effects['invert'] = random.randint(1, 5) # Short flashes

        if self.active_effects['stutter'] == 0 and random.random() < self.s['prob_stutter']:
            self.active_effects['stutter'] = random.randint(3, 8)
            self.stutter_buffer = None # Reset buffer

        if self.active_effects['mirror'] == 0 and random.random() < self.s['prob_mirror']:
            self.active_effects['mirror'] = random.randint(10, 30)

    def apply(self, img):
        """Applies active effects to the image."""
        
        # Stutter (Time Freeze) - Must be first
        if self.active_effects['stutter'] > 0:
            if self.stutter_buffer is None:
                self.stutter_buffer = img.copy()
            return self.stutter_buffer # Return the frozen frame
        
        res = img
        
        # Liquid Melting
        if self.active_effects['liquid'] > 0:
            # Intensity fades in and out based on remaining duration
            res = effect_liquid_displacement(res, self.s['intensity_liquid'], self.time_counter)

        # Pixel Sort
        if self.active_effects['pixel_sort'] > 0:
            res = effect_pixel_sort(res)

        # RGB Split
        if self.active_effects['rgb_split'] > 0:
            res = effect_rgb_split(res, self.s['intensity_rgb'])

        # Invert Colors (Negative)
        if self.active_effects['invert'] > 0:
            res = ImageOps.invert(res)

        # Mirroring (Kaleidoscope-ish)
        if self.active_effects['mirror'] > 0:
            res = ImageOps.mirror(res)

        return res

# -------------------------
# Main Workflow
# -------------------------
def run_ffmpeg(cmd):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg Error: {e.stderr}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", default="input.mp4")
    parser.add_argument("--v2", default="image2.mp4")
    parser.add_argument("--out", default="output_psychedelic.mp4")
    
    # Parameter Controls
    parser.add_argument("--chaos", type=float, default=1.0, help="Multiplier for all probabilities")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    # Scale probabilities by chaos factor
    settings = DEFAULTS.copy()
    for k in settings:
        if k.startswith("prob_"):
            settings[k] *= args.chaos

    t_v1, t_v2, t_out = "frames_v1", "frames_v2", "frames_out"
    t_audio, temp_vid = "audio.aac", "temp.mp4"
    
    for d in [t_v1, t_v2, t_out]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    try:
        # 1. Extract
        print("Extracting frames...")
        # Check files exist
        if not os.path.exists(args.v1) or not os.path.exists(args.v2):
            raise FileNotFoundError("Input files missing. Check Git LFS.")

        run_ffmpeg(["ffmpeg", "-i", args.v1, "-vf", f"fps={args.fps}", f"{t_v1}/frame_%05d.png"])
        run_ffmpeg(["ffmpeg", "-i", args.v2, "-vf", f"fps={args.fps}", f"{t_v2}/frame_%05d.png"])
        
        # Audio Extraction
        has_audio = False
        try:
            run_ffmpeg(["ffmpeg", "-i", args.v1, "-vn", "-c:a", "aac", t_audio])
            has_audio = True
        except:
            pass

        # 2. Process
        files_v1 = sorted([os.path.join(t_v1, f) for f in os.listdir(t_v1) if f.endswith(".png")])
        files_v2 = sorted([os.path.join(t_v2, f) for f in os.listdir(t_v2) if f.endswith(".png")])
        
        total_frames = len(files_v1) + len(files_v2)
        print(f"Processing {total_frames} frames with Event System...")
        
        # Initialize Logic
        gm = GlitchManager(settings)
        previous_img = None
        
        # Get target size
        with Image.open(files_v1[0]) as ref: target_size = ref.size

        for i in range(total_frames):
            print(f"Rendering {i}/{total_frames}", end='\r')
            
            # --- Source Mixing Logic ---
            if i < len(files_v1):
                src = Image.open(files_v1[i]).convert("RGB")
                
                # Transition Blend
                left = len(files_v1) - i
                trans_len = int(len(files_v1) * settings['transition_fraction'])
                
                if left < trans_len and files_v2:
                    # Map to V2
                    idx2 = int((1 - (left/trans_len)) * (len(files_v2)-1))
                    src2 = Image.open(files_v2[idx2]).convert("RGB").resize(src.size)
                    # Use Difference blend for transition (Trippy)
                    src = ImageChops.difference(src, src2)
            else:
                idx = i - len(files_v1)
                if idx >= len(files_v2): break
                src = Image.open(files_v2[idx]).convert("RGB")

            # Enforce size
            if src.size != target_size: src = src.resize(target_size)

            # --- Update & Apply Glitch State ---
            gm.update()
            
            # 1. Apply Datamosh Smearing (P-Frame simulation)
            # This runs ALWAYS to maintain the "mosh" feel
            src = macroblock_smear(src, previous_img)
            
            # 2. Apply Event-Based Effects
            src = gm.apply(src)
            
            src.save(f"{t_out}/frame_{i:05d}.png")
            previous_img = src.copy()

        # 3. Reassemble
        print("\nReassembling...")
        run_ffmpeg(["ffmpeg", "-y", "-framerate", str(args.fps), "-i", f"{t_out}/frame_%05d.png", 
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", temp_vid])
        
        # 4. Audio Destruction
        if has_audio:
            filter_chain = get_random_audio_filter()
            print(f"Applying Audio Chain: {filter_chain}")
            run_ffmpeg(["ffmpeg", "-y", "-i", temp_vid, "-i", t_audio, 
                        "-af", filter_chain, 
                        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", 
                        "-shortest", args.out])
        else:
            os.rename(temp_vid, args.out)
            
        print(f"Done: {args.out}")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    finally:
        # Cleanup
        for d in [t_v1, t_v2, t_out]: 
            if os.path.exists(d): shutil.rmtree(d)
        if os.path.exists(t_audio): os.remove(t_audio)
        if os.path.exists(temp_vid): os.remove(temp_vid)

if __name__ == "__main__":
    main()
