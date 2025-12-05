#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# HYPER-LUDOVICO: DIGITAL ENTROPY ENGINE
# -----------------------------------------------------------------------------
# GOAL:
#   - Relentless visual collapse / data corruption
#   - Extreme recursive pixel dragging, tectonic shifts, non-linear flows
#   - Bit-level corruption of frame buffers
#   - YUV-channel abuse: VHS-from-hell psychedelic bleed
#   - Pure harsh-noise audio synthesis (no source audio)
#   - Still: fast render (240p → 720p, ultrafast x264, light output)
# -----------------------------------------------------------------------------

import os
import shutil
import subprocess
import random
import argparse
import tempfile
import numpy as np
import cv2
import wave
import struct

# -------------------- BASE CONFIG --------------------

BASE_SETTINGS = {
    "fps": 24,
    "internal_res": (426, 240),   # 240p for speed & blockiness

    # Feedback / drag physics
    "threshold": 28.0,            # Base threshold for ghost masking
    "decay": 0.995,               # Trails: higher = longer persistence
    "zoom_base": 1.012,           # Base zoom drift
    "zoom_jitter": 0.02,          # Per-frame zoom variation
    "rotate_range": 2.0,          # +/- degrees

    # Recursive smear
    "smear_passes": 3,            # Recursive warps per frame

    # P-frame hold (for stretching smears)
    "hold_prob": 0.05,
    "hold_frames_max": 25,

    # Block glitches
    "block_prob": 0.15,
    "block_iterations": 5,
    "block_size_min": 16,
    "block_size_max": 72,

    # Bit-level corruption
    # (probability per BYTE, applied on selected stripes; burst mode multiplies)
    "bitflip_prob": 0.002,

    # Color jitter / channel drift
    "color_jitter_base": 8,

    # Phase lengths (in frames)
    "phase_min": 40,
    "phase_max": 160,
}

# -------------------- BIT-LEVEL CORRUPTION --------------------

def bitflip_stripes(img, base_prob, chaos_mult=1.0, burst_factor=1.0):
    """
    Bit-level corruption on raw pixel bytes.
    Applies flipping in full or striped regions for tectonic chroma breakdown.
    """
    if base_prob <= 0:
        return img

    h, w, c = img.shape
    out = img.copy()

    # Decide corruption mode
    mode = random.choice(["full", "hstripe", "vstripe"])
    if mode == "full":
        region = out
    elif mode == "hstripe":
        stripe_h = random.randint(h // 8, max(2, h // 3))
        y1 = random.randint(0, max(0, h - stripe_h))
        region = out[y1:y1 + stripe_h, :, :]
    else:  # vstripe
        stripe_w = random.randint(w // 8, max(2, w // 3))
        x1 = random.randint(0, max(0, w - stripe_w))
        region = out[:, x1:x1 + stripe_w, :]

    flat = region.reshape(-1)  # uint8
    prob = base_prob * chaos_mult * burst_factor

    if prob <= 0:
        return out

    mask = np.random.rand(flat.size) < prob
    if not mask.any():
        return out

    bits = (1 << np.random.randint(0, 8, size=mask.sum(), dtype=np.uint8)).astype(np.uint8)
    flat_masked = flat[mask]
    flat_masked ^= bits
    flat[mask] = flat_masked
    return out

# -------------------- VISUAL FX --------------------

def fast_block_shuffle(img, prob=0.1, iterations=4, size_min=16, size_max=64):
    """Structural block glitches: swaps random rectangular blocks."""
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


def fast_color_jitter_yuv(img_bgr, chaos_mult=1.0):
    """
    Independent, violent YUV channel abuse.
    Converts to YCrCb, mutates channels, and returns BGR.
    """
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y, Cr, Cb = cv2.split(img_yuv)

    # Luma: flicker, inversions
    if random.random() < 0.4 * chaos_mult:
        Y = 255.0 - Y
    gain_Y = 0.6 + 1.4 * random.random() * chaos_mult
    Y = np.clip(Y * gain_Y + random.uniform(-40, 40) * chaos_mult, 0, 255)

    # Chroma: insane saturation, bias shifting
    gain_C = 0.8 + 2.5 * random.random() * chaos_mult
    shift_Cr = random.uniform(-80, 80) * chaos_mult
    shift_Cb = random.uniform(-80, 80) * chaos_mult
    Cr = np.clip((Cr - 128) * gain_C + 128 + shift_Cr, 0, 255)
    Cb = np.clip((Cb - 128) * gain_C + 128 + shift_Cb, 0, 255)

    img_yuv = cv2.merge([Y, Cr, Cb])
    img_yuv = np.clip(img_yuv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)


def apply_feedback_warp(prev_img, zoom, rotate, smear_passes=1):
    """Recursive zoom/rotate smear for melting feedback."""
    h, w = prev_img.shape[:2]
    center = (w // 2, h // 2)
    ghost = prev_img

    for _ in range(max(1, smear_passes)):
        M = cv2.getRotationMatrix2D(center, rotate, zoom)
        ghost = cv2.warpAffine(ghost, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return ghost


def apply_global_shift(frame, chaos_mult=1.0):
    """Non-linear, brutal frame roll / tearing."""
    h, w, c = frame.shape
    out = frame

    if random.random() < 0.4 * chaos_mult:
        # Horizontal tear
        shift = random.randint(-w // 3, w // 3)
        band_h = random.randint(h // 6, h // 2)
        y1 = random.randint(0, max(0, h - band_h))
        band = out[y1:y1+band_h]
        out[y1:y1+band_h] = np.roll(band, shift, axis=1)

    if random.random() < 0.3 * chaos_mult:
        # Vertical roll
        shift = random.randint(-h // 4, h // 4)
        out = np.roll(out, shift, axis=0)

    return out

# -------------------- AUDIO: HARSH NOISE SYNTHESIS --------------------

def synth_harsh_noise(duration_sec, sample_rate=22050, chaos=3.0, drag=2.0):
    """
    Pure harsh noise / experimental:
    - White noise + warped square/sine
    - Non-repeating envelopes
    - Granular stutter & time-smear
    - Heavy saturation & clipping
    """
    if duration_sec <= 0:
        duration_sec = 1.0

    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False)

    # Base white noise
    white = np.random.uniform(-1.0, 1.0, n_samples)

    # Random-walking frequency for square & sine
    freq = np.zeros(n_samples, dtype=np.float32)
    freq[0] = random.uniform(40, 4000)
    step_scale = 40.0 * chaos
    for i in range(1, n_samples):
        freq[i] = freq[i-1] + np.random.uniform(-step_scale, step_scale)
    freq = np.clip(freq, 20, 10000)

    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    square = np.sign(np.sin(phase))
    sine = np.sin(phase * random.uniform(0.5, 2.5))

    # Amplitude envelope: random segments w/ ramps (non-repeating feel)
    env = np.zeros(n_samples, dtype=np.float32)
    idx = 0
    while idx < n_samples:
        seg_len = random.randint(int(0.01 * sample_rate), int(0.3 * sample_rate))
        seg_len = min(seg_len, n_samples - idx)
        amp_start = random.random() ** 2
        amp_end = random.random() ** 2
        seg = np.linspace(amp_start, amp_end, seg_len, endpoint=False)
        env[idx:idx+seg_len] = seg
        idx += seg_len
    env = np.clip(env * (0.2 + 1.5 * chaos), 0, 10.0)

    # Combine components
    signal = 0.6 * white + 0.8 * square + 0.4 * sine
    signal *= env

    # Time-smear / stutter: repeat or zero random chunks
    chunk_size = int(0.02 * sample_rate)
    if chunk_size < 1:
        chunk_size = 1
    n_chunks = n_samples // chunk_size
    for c in range(n_chunks):
        if random.random() < 0.1 * chaos:
            start = c * chunk_size
            end = start + chunk_size
            mode = random.choice(["repeat_prev", "mute", "invert"])
            if mode == "repeat_prev" and c > 0:
                prev_start = (c - 1) * chunk_size
                signal[start:end] = signal[prev_start:prev_start+chunk_size]
            elif mode == "mute":
                signal[start:end] = 0.0
            elif mode == "invert":
                signal[start:end] = -signal[start:end]

    # Ring-mod type effect: multiply by a slower modulator
    mod_freq = random.uniform(2.0, 40.0 * chaos)
    mod = np.sin(2 * np.pi * mod_freq * t + random.uniform(0, 2*np.pi))
    signal *= (0.5 + 0.5 * mod)

    # Extreme waveshaping / clipping
    drive = 3.0 * chaos * drag
    signal = np.tanh(signal * drive)

    # Normalize to int16 range
    signal /= (np.max(np.abs(signal)) + 1e-6)
    signal_int16 = (signal * 32767).astype(np.int16)
    return signal_int16, sample_rate


def write_wav_mono(path, data_int16, sample_rate):
    """Write mono 16-bit PCM WAV using standard library."""
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        frames = struct.pack('<' + 'h' * len(data_int16), *data_int16.tolist())
        wf.writeframes(frames)

# -------------------- MAIN PROCESSOR --------------------

def process_video(
    v1_path,
    v2_path,
    out_path,
    drag_mult=2.0,
    chaos_mult=3.0,
    fps=None,
    internal_res=None,
    seed=-1,
    crf=24,
):
    # Seed for reproducibility / chaos
    if seed < 0:
        seed = random.randint(0, 9999999)
    random.seed(seed)
    np.random.seed(seed)
    print(f">>> SEED: {seed}")

    s = BASE_SETTINGS.copy()

    # Overrides
    if fps is not None:
        s["fps"] = int(fps)
    if internal_res is not None:
        s["internal_res"] = internal_res

    # Drag & chaos injection
    s["threshold"] *= drag_mult
    s["decay"] = 1.0 - (1.0 - s["decay"]) / max(0.4, drag_mult)
    s["zoom_jitter"] *= chaos_mult
    s["block_prob"] = min(1.0, s["block_prob"] * chaos_mult)
    s["block_iterations"] = max(1, int(s["block_iterations"] * chaos_mult))
    s["color_jitter_base"] = max(2, int(s["color_jitter_base"] * chaos_mult))
    s["bitflip_prob"] *= chaos_mult

    w, h = s["internal_res"]
    print(f" -> FPS: {s['fps']}")
    print(f" -> Internal res: {w}x{h}")
    print(f" -> Threshold: {s['threshold']:.2f} | Decay: {s['decay']:.4f}")
    print(f" -> Zoom base: {s['zoom_base']:.4f} | Zoom jitter: {s['zoom_jitter']:.4f}")
    print(f" -> Block prob: {s['block_prob']:.2f} | iters: {s['block_iterations']}")
    print(f" -> Bitflip prob: {s['bitflip_prob']:.5f}")
    print(f" -> Drag mult: {drag_mult} | Chaos mult: {chaos_mult}")
    print(f" -> CRF: {crf}")

    tmp = tempfile.mkdtemp()
    try:
        # -------------------- EXTRACT FRAMES --------------------
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
            raise RuntimeError("Input v1 produced no frames.")

        total_frames = len(f1_files) + len(f2_files)
        print(f" -> Total frames: {total_frames}")

        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)

        prev_frame = None
        thresh_val = s["threshold"]
        decay_val = s["decay"]

        # P-frame hold
        hold_counter = 0
        hold_frames_max = int(s["hold_frames_max"] * drag_mult)

        # Phase system: corrosive vs burst
        current_phase = "corrosive"
        phase_remaining = random.randint(s["phase_min"], s["phase_max"])

        # Pre-randomized color channel map
        base_color_map = [0, 1, 2]

        for i in range(total_frames):
            if i % 40 == 0:
                print(f"   Frame {i}/{total_frames} [{current_phase}]      ", end="\r")

            # Phase handling
            phase_remaining -= 1
            if phase_remaining <= 0:
                current_phase = "burst" if current_phase == "corrosive" else "corrosive"
                phase_remaining = random.randint(s["phase_min"], s["phase_max"])

            # Base frame selection & blending
            if i < len(f1_files):
                curr = cv2.imread(os.path.join(tmp, f1_files[i]))
                if curr is None:
                    continue
                if f2_files and i > len(f1_files) * 0.7:
                    idx2 = random.randint(0, len(f2_files) - 1)
                    alt = cv2.imread(os.path.join(tmp, f2_files[idx2]))
                    if alt is not None:
                        alpha = 0.5 + 0.5 * random.random()
                        curr = cv2.addWeighted(curr, 1.0 - alpha, alt, alpha, 0)
            else:
                idx = i - len(f1_files)
                if idx >= len(f2_files):
                    break
                curr = cv2.imread(os.path.join(tmp, f2_files[idx]))
                if curr is None:
                    continue

            # Randomize color map occasionally for psychedelic BGR scrambles
            color_map = base_color_map[:]
            if random.random() < 0.4 * chaos_mult:
                random.shuffle(color_map)
            curr = curr[:, :, color_map]

            if prev_frame is None:
                final = curr
                prev_frame = curr
                cv2.imwrite(f"{out_dir}/frame_{i:05d}.bmp", final)
                continue

            # -------------------- FEEDBACK / DRAGGING --------------------
            if current_phase == "corrosive":
                local_zoom = s["zoom_base"] + random.uniform(-s["zoom_jitter"], s["zoom_jitter"])
                local_rotate = random.uniform(-s["rotate_range"], s["rotate_range"])
                smear_passes = s["smear_passes"]
                burst_factor = 0.7
                local_thresh = thresh_val * 1.2   # more ghost kept
            else:  # burst
                local_zoom = s["zoom_base"] + random.uniform(-s["zoom_jitter"] * 3, s["zoom_jitter"] * 3)
                local_rotate = random.uniform(-s["rotate_range"] * 3, s["rotate_range"] * 3)
                smear_passes = s["smear_passes"] + 1
                burst_factor = 3.0
                local_thresh = max(5.0, thresh_val * 0.5)  # less mask, more violent change

            ghost = apply_feedback_warp(prev_frame, local_zoom, local_rotate, smear_passes)

            diff = cv2.absdiff(curr, ghost)
            diff_mag = np.sum(diff, axis=2)
            mask = diff_mag < local_thresh
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

            ghost_decayed = cv2.addWeighted(ghost, decay_val, curr, 1.0 - decay_val, 0.0)
            final = np.where(mask_3d, ghost_decayed, curr)

            # P-frame hold mechanic
            if hold_counter > 0:
                hold_counter -= 1
            else:
                if random.random() < s["hold_prob"] * drag_mult:
                    hold_counter = random.randint(1, max(2, hold_frames_max))
                else:
                    prev_frame = final

            # Global tectonic rolling
            final = apply_global_shift(final, chaos_mult=chaos_mult)

            # Bit-level corruption
            final = bitflip_stripes(
                final,
                base_prob=s["bitflip_prob"],
                chaos_mult=chaos_mult,
                burst_factor=burst_factor,
            )

            # YUV-channel abuse (VHS-from-hell)
            if random.random() < 0.8:
                final = fast_color_jitter_yuv(final, chaos_mult=chaos_mult)

            # Block-level glitches
            final = fast_block_shuffle(
                final,
                prob=s["block_prob"],
                iterations=s["block_iterations"],
                size_min=s["block_size_min"],
                size_max=s["block_size_max"],
            )

            cv2.imwrite(f"{out_dir}/frame_{i:05d}.bmp", final)

        print("\n -> Rendering video (upscale to 720p)...")
        temp_vid = os.path.join(tmp, "temp.mp4")
        cmd_render = [
            "ffmpeg", "-y", "-framerate", str(s["fps"]),
            "-i", f"{out_dir}/frame_%05d.bmp",
            "-vf", "scale=1280:720:flags=neighbor",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            temp_vid,
        ]
        subprocess.run(cmd_render, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # -------------------- SYNTHESIZE HARSH NOISE AUDIO --------------------
        duration_sec = total_frames / float(s["fps"])
        print(f" -> Synthesizing harsh noise audio ({duration_sec:.2f}s)...")
        noise_data, sr = synth_harsh_noise(duration_sec, sample_rate=22050, chaos=chaos_mult, drag=drag_mult)
        audio_path = os.path.join(tmp, "noise.wav")
        write_wav_mono(audio_path, noise_data, sr)

        # -------------------- MUX --------------------
        print(" -> Muxing video + noise...")
        cmd_mux = [
            "ffmpeg", "-y",
            "-i", temp_vid,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            out_path,
        ]
        subprocess.run(cmd_mux, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"DONE: {out_path}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# -------------------- CLI --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyper-Ludovico: Digital Entropy Engine (extreme datamosh + harsh noise)."
    )
    parser.add_argument("--v1", required=True, help="Primary input video.")
    parser.add_argument("--v2", required=True, help="Secondary input / texture video.")
    parser.add_argument("--out", default="output_entropy.mp4", help="Output video path.")
    parser.add_argument("--drag", type=float, default=2.0,
                        help="Drag multiplier (trail strength/length). Default=2.0")
    parser.add_argument("--chaos", type=float, default=3.0,
                        help="Chaos multiplier (glitches, corruption, noise). Default=3.0")
    parser.add_argument("--fps", type=int, default=None,
                        help="Override FPS (default from BASE_SETTINGS).")
    parser.add_argument("--internal-width", type=int, default=None,
                        help="Internal processing width (default 426).")
    parser.add_argument("--internal-height", type=int, default=None,
                        help="Internal processing height (default 240).")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed (-1 = random).")
    parser.add_argument("--crf", type=int, default=24,
                        help="CRF for x264; higher = smaller, more compressed, more artifacted. Default=24.")
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
        seed=args.seed,
        crf=args.crf,
    )


