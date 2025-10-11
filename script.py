import os
import shutil
import subprocess
import random
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageChops

# --- FFMPEG CORE FUNCTIONS (VIDEO & AUDIO) ---

def run_ffmpeg(command):
    """A helper to run ffmpeg commands quietly."""
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_frames(video_path, output_folder):
    """Extracts frames from a video into a specified folder."""
    print(f"Extracting frames from '{video_path}'...")
    os.makedirs(output_folder, exist_ok=True)
    run_ffmpeg(["ffmpeg", "-i", video_path, "-vsync", "0", os.path.join(output_folder, "frame_%05d.png")])

def extract_audio(video_path, audio_path):
    """Extracts the audio track from a video."""
    print(f"Extracting audio from '{video_path}'...")
    if os.path.exists(audio_path):
        os.remove(audio_path)
    run_ffmpeg(["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path])

def glitch_audio(input_audio, output_audio):
    """Applies horrifying glitches to an audio file."""
    print("Destroying audio...")
    if not os.path.exists(input_audio):
        print("No audio track found to glitch.")
        return False
    # Chain of ffmpeg audio filters: echo, vibrato, tempo shifts
    run_ffmpeg([
        "ffmpeg", "-y", "-i", input_audio, "-af",
        "aecho=0.8:0.9:500:0.3,vibrato=f=7.0:d=0.8,atempo=0.85",
        output_audio
    ])
    return True

def reassemble_video_with_audio(frames_folder, audio_path, output_video, fps=30):
    """Reassembles frames and merges with audio."""
    print("Reassembling final horrifying video...")
    video_only_output = "temp_video_only.mp4"
    # Create video from frames
    run_ffmpeg([
        "ffmpeg", "-y", "-framerate", str(fps), "-i",
        os.path.join(frames_folder, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", video_only_output
    ])
    # Merge with audio
    if os.path.exists(audio_path):
        run_ffmpeg([
            "ffmpeg", "-y", "-i", video_only_output, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-shortest", output_video
        ])
        os.remove(audio_path)
    else:
        os.rename(video_only_output, output_video)
    
    if os.path.exists(video_only_output):
        os.remove(video_only_output)


# --- NEW DESTRUCTIVE IMAGE EFFECTS ---

def apply_pixel_sort(image, intensity=0.5):
    """
    Applies a horizontal pixel sorting effect to a random portion of the image.
    """
    if random.random() > intensity:
        return image

    img_data = image.load()
    width, height = image.size
    
    start_y = random.randint(0, height - 1)
    chunk_height = random.randint(10, int(height * 0.2))
    end_y = min(start_y + chunk_height, height)

    for y in range(start_y, end_y):
        line_pixels = [img_data[x, y] for x in range(width)]
        # Sort pixels by their luminance (brightness)
        line_pixels.sort(key=lambda p: 0.299*p[0] + 0.587*p[1] + 0.114*p[2])
        for x, pixel in enumerate(line_pixels):
            img_data[x, y] = pixel
            
    return image

def create_random_mask(size):
    """Creates a random black and white mask for blending."""
    mask = Image.new("L", size, 0)
    data = []
    # Create chaotic vertical or horizontal bars
    if random.random() > 0.5: # Vertical bars
        for x in range(size[0]):
            color = 255 if random.random() > 0.5 else 0
            for y in range(size[1]):
                data.append(color)
    else: # Noisy blocks
         for _ in range(size[0] * size[1]):
             data.append(random.randint(0, 255))
    mask.putdata(data)
    return mask.filter(ImageFilter.GaussianBlur(radius=random.randint(5, 25)))


def blend_frames_chaotically(img1, img2):
    """
    Blends two frames together using a random, horrifying method.
    """
    blend_mode = random.choice(['screen', 'difference', 'multiply', 'mask'])
    
    if blend_mode == 'difference':
        return ImageChops.difference(img1, img2)
    elif blend_mode == 'screen':
        return ImageChops.screen(img1, img2)
    elif blend_mode == 'multiply':
        return ImageChops.multiply(img1, img2)
    elif blend_mode == 'mask':
        mask = create_random_mask(img1.size)
        return Image.composite(img1, img2, mask)
    return img1 # Failsafe

# --- MAIN WORKFLOW ---

def datamosh_and_destroy(video1_path, video2_path, output_video, settings):
    """Main function to orchestrate the entire psychedelic datamoshing process."""
    if not os.path.exists(video1_path) or not os.path.exists(video2_path):
        print("Error: One or both input videos not found.")
        return

    # Setup temporary directories
    temp_v1 = "temp_v1"; temp_v2 = "temp_v2"; temp_out = "temp_out"
    for d in [temp_v1, temp_v2, temp_out]:
        if os.path.exists(d): shutil.rmtree(d)

    try:
        # 1. Extract frames and audio
        extract_frames(video1_path, temp_v1)
        extract_frames(video2_path, temp_v2)
        temp_audio = "temp_audio.aac"
        temp_audio_glitched = "temp_audio_glitched.aac"
        extract_audio(video1_path, temp_audio)
        glitch_audio(temp_audio, temp_audio_glitched)
        
        # 2. Process and glitch frames
        frames1 = sorted([os.path.join(temp_v1, f) for f in os.listdir(temp_v1)])
        frames2 = sorted([os.path.join(temp_v2, f) for f in os.listdir(temp_v2)])
        total_frames = len(frames1) + len(frames2)
        os.makedirs(temp_out)

        transition_point = int(len(frames1) * settings['transition_point'])
        transition_duration = int(len(frames1) * settings['transition_duration'])
        transition_end = transition_point + transition_duration

        print(f"Processing a total of {total_frames} frames...")
        for i in range(total_frames):
            print(f"  -> Generating frame {i+1}/{total_frames}", end='\r')
            
            # Determine which source frame(s) to use
            if i < transition_point: # Before transition
                current_img = Image.open(frames1[i]).convert("RGB")
            elif transition_point <= i < transition_end: # During transition
                idx1 = min(i, len(frames1) - 1)
                # Map the transition progress to the frames of video 2
                progress = (i - transition_point) / transition_duration
                idx2 = int(progress * (len(frames2) - 1))
                img1 = Image.open(frames1[idx1]).convert("RGB")
                img2 = Image.open(frames2[idx2]).convert("RGB")
                current_img = blend_frames_chaotically(img1, img2)
            else: # After transition
                idx2 = i - len(frames1)
                if idx2 >= len(frames2): break # Stop if we run out of frames
                current_img = Image.open(frames2[idx2]).convert("RGB")

            # Apply additional destructive layers
            current_img = apply_pixel_sort(current_img, settings['pixel_sort_chance'])
            if random.random() < settings['vhs_glitch_chance']:
                enhancer = ImageEnhance.Color(current_img)
                current_img = enhancer.enhance(random.uniform(0.1, 3.0)) # Color saturation flicker
                current_img = current_img.filter(ImageFilter.SHARPEN)

            # Save the corrupted frame
            output_frame_path = os.path.join(temp_out, f"frame_{i:05d}.png")
            current_img.save(output_frame_path)
        
        print("\nFrame generation complete.")

        # 3. Reassemble video with glitched audio
        reassemble_video_with_audio(temp_out, temp_audio_glitched, output_video, settings['fps'])
        print(f"\nSUCCESS! Your avant-garde film is saved as '{output_video}'")

    except Exception as e:
        print(f"\nAn error destroyed the process: {e}")
    finally:
        # 4. Cleanup
        for d in [temp_v1, temp_v2, temp_out]:
            if os.path.exists(d): shutil.rmtree(d)
        if os.path.exists("temp_audio.aac"): os.remove("temp_audio.aac")
        print("Temporary files cleaned up.")


if __name__ == "__main__":
    input_video_1 = "input.mp4"       # Your first video file
    input_video_2 = "image2.mp4"    # The video to transition into
    output_video_file = "output_horrifying_mosh.mp4"

    # --- TWEAK YOUR NIGHTMARE HERE ---
    destruction_settings = {
        'fps': 24,

        # When the transition starts (0.8 = 80% into the first video)
        'transition_point': 0.8,

        # How long the transition lasts (0.2 = 20% of the first video's length)
        'transition_duration': 0.2,
        
        # Chance per frame to apply the pixel sorting effect
        'pixel_sort_chance': 0.35,
        
        # Chance per frame to apply random color shifts and sharpening
        'vhs_glitch_chance': 0.6
    }

    datamosh_and_destroy(input_video_1, input_video_2, output_video_file, destruction_settings)
