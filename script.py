import os
import shutil
import subprocess
import random
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

# --- CORE FFMPEG FUNCTIONS ---

def extract_frames(video_path, output_folder):
    """Extracts all frames from a video using ffmpeg."""
    print(f"Extracting frames from '{video_path}'...")
    os.makedirs(output_folder, exist_ok=True)
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vsync", "0",
        os.path.join(output_folder, "frame_%05d.png") # Use PNG for lossless processing
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Frame extraction complete.")

def reassemble_video(input_folder, output_video, fps=30):
    """Reassembles frames into a video using ffmpeg."""
    print("Reassembling video...")
    command = [
        "ffmpeg",
        "-y", # Overwrite output file if it exists
        "-framerate", str(fps),
        "-i", os.path.join(input_folder, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Video reassembly complete.")

# --- NEW PSYCHEDELIC & DESTRUCTIVE EFFECTS ---

def apply_vhs_color_bleed(image, intensity=0.5):
    """
    Shifts color channels to simulate analog VHS color bleed.
    Intensity (0.0 to 1.0) controls the max shift distance.
    """
    if random.random() > intensity:
        return image

    max_offset = int(image.width * 0.02 * intensity)
    if max_offset == 0: return image
    
    r, g, b = image.split()
    
    r_offset = (random.randint(-max_offset, max_offset), random.randint(-max_offset, max_offset))
    g_offset = (random.randint(-max_offset, max_offset), random.randint(-max_offset, max_offset))
    
    # Create shifted channels
    r = r.transform(image.size, Image.AFFINE, (1, 0, r_offset[0], 0, 1, r_offset[1]))
    g = g.transform(image.size, Image.AFFINE, (1, 0, g_offset[0], 0, 1, g_offset[1]))
    
    return Image.merge("RGB", (r, g, b))

def apply_psychedelic_filters(image, intensity=0.3):
    """
    Applies random intense color filters.
    Intensity (0.0 to 1.0) is the probability of applying a filter.
    """
    if random.random() < intensity:
        filter_choice = random.choice(['solarize', 'posterize', 'invert', 'colorize'])
        if filter_choice == 'solarize':
            return ImageOps.solarize(image, threshold=random.randint(50, 200))
        elif filter_choice == 'posterize':
            return ImageOps.posterize(image, bits=random.randint(1, 4))
        elif filter_choice == 'invert':
            return ImageOps.invert(image)
        elif filter_choice == 'colorize':
            # Create a random two-color map for a duotone effect
            black_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            white_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            return ImageOps.colorize(image.convert("L"), black=black_color, white=white_color)
    return image

def apply_analog_damage(image, intensity=0.5):
    """
    Simulates analog damage like noise, blur, and contrast flicker.
    Intensity (0.0 to 1.0) controls the magnitude of the effects.
    """
    if random.random() < intensity:
        # Brightness/Contrast flicker
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(1.0 - intensity, 1.0 + intensity))
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(1.0 - (intensity * 0.5), 1.0 + (intensity * 0.5)))

    if random.random() < intensity * 0.5:
        # Add blur
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2 * intensity)))
        
    return image

def corrupt_frame_bytes(image_path, intensity=10):
    """
    Directly corrupts the bytes of the saved image file for digital artifacts.
    This is the original datamoshing technique.
    """
    try:
        with open(image_path, 'r+b') as f:
            data = bytearray(f.read())
            data_len = len(data)
            # Corrupt a larger chunk of bytes
            num_corruptions = int(intensity * random.uniform(0.5, 1.5))

            for _ in range(num_corruptions):
                # Start corruption after the header to avoid completely breaking the file
                index = random.randint(int(data_len * 0.1), data_len - 1)
                data[index] = random.randint(0, 255)
            
            f.seek(0)
            f.write(data)
    except Exception as e:
        print(f"Warning: Could not corrupt bytes for {image_path}. Reason: {e}")

# --- MAIN WORKFLOW ---

def process_and_glitch_frames(folder, settings):
    """The main loop to process, modify, and rearrange frames."""
    frames = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    num_frames = len(frames)
    
    last_good_frame_path = None

    for i, frame_name in enumerate(frames):
        print(f"Processing frame {i+1}/{num_frames}...", end='\r')
        frame_path = os.path.join(folder, frame_name)

        # Frame Repetition / Freezing Effect
        if random.random() < settings['frame_repeat_chance'] and last_good_frame_path:
            shutil.copy(last_good_frame_path, frame_path)

        try:
            with Image.open(frame_path) as img:
                img = img.convert("RGB") # Ensure it's in RGB mode

                # Apply a chain of destructive effects
                img = apply_vhs_color_bleed(img, settings['vhs_bleed_intensity'])
                img = apply_psychedelic_filters(img, settings['color_glitch_chance'])
                img = apply_analog_damage(img, settings['analog_damage_intensity'])
                
                img.save(frame_path, 'PNG')
                last_good_frame_path = frame_path

            # Apply byte corruption AFTER saving for digital artifacts
            if random.random() < settings['byte_corruption_chance']:
                corrupt_frame_bytes(frame_path, settings['byte_corruption_intensity'])

        except Exception as e:
            print(f"\nError processing {frame_name}: {e}. Attempting to replace with last good frame.")
            if last_good_frame_path:
                shutil.copy(last_good_frame_path, frame_path)

    print("\nAll frames processed.")


def datamosh_psychedelic(video_path, output_video, fps=30, temp_folder="temp_frames_glitch", settings=None):
    """
    Main function to orchestrate the entire psychedelic datamoshing process.
    """
    if not os.path.exists(video_path):
        print(f"Error: Input video not found at '{video_path}'")
        return

    # Define default settings if none are provided
    if settings is None:
        settings = {
            'vhs_bleed_intensity': 0.6,       # 0.0 to 1.0: How much color channels shift
            'color_glitch_chance': 0.4,       # 0.0 to 1.0: Chance to apply a solarize/posterize effect
            'analog_damage_intensity': 0.5,   # 0.0 to 1.0: How much contrast/brightness flicker and blur
            'frame_repeat_chance': 0.1,       # 0.0 to 1.0: Chance to repeat the previous frame (stutter)
            'byte_corruption_chance': 0.25,    # 0.0 to 1.0: Chance to apply classic byte corruption
            'byte_corruption_intensity': 200  # How many bytes to corrupt if triggered
        }

    try:
        extract_frames(video_path, temp_folder)
        process_and_glitch_frames(temp_folder, settings)
        reassemble_video(temp_folder, output_video, fps)
        print(f"\nDatamoshing complete! Your masterpiece is saved as '{output_video}'")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Clean up the temporary frame folder
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
            print(f"Temporary folder '{temp_folder}' removed.")


if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_glitched.mp4"

    # --- TWEAK YOUR GLITCHES HERE ---
    # Feel free to experiment with these values.
    # Higher values = more destruction.
    glitch_settings = {
        # How much the red/green channels separate from blue. (0.0 to 1.0)
        'vhs_bleed_intensity': 0.7,
        
        # Chance per frame to apply a wild color filter. (0.0 to 1.0)
        'color_glitch_chance': 0.5,
        
        # Intensity of fake TV noise, contrast flicker, and blur. (0.0 to 1.0)
        'analog_damage_intensity': 0.6,
        
        # Chance per frame to get stuck, creating stutters. (0.0 to 1.0)
        'frame_repeat_chance': 0.15,
        
        # Chance to apply the classic "digital data" glitch. (0.0 to 1.0)
        'byte_corruption_chance': 0.3,
        
        # How many bytes to scramble for the digital glitch. Higher is more chaotic.
        'byte_corruption_intensity': 500
    }

    datamosh_psychedelic(input_video, output_video, fps=24, settings=glitch_settings)
