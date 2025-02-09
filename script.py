import os
import shutil
import subprocess
from PIL import Image
import random

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    command = [
        "ffmpeg", "-i", video_path, "-vsync", "0", 
        os.path.join(output_folder, "frame%04d.jpg")
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def corrupt_image(image_path, intensity=5):
    with open(image_path, 'rb') as f:
        data = bytearray(f.read())
    
    for _ in range(intensity):
        index = random.randint(50, len(data) - 50)
        data[index] = random.randint(0, 255)
    
    with open(image_path, 'wb') as f:
        f.write(data)

def glitch_frames(folder):
    frames = sorted(os.listdir(folder))
    for frame in frames:
        if random.random() < 0.3:  # 30% of frames will be glitched
            corrupt_image(os.path.join(folder, frame))

def reassemble_video(input_folder, output_video, fps=30):
    command = [
        "ffmpeg", "-framerate", str(fps), "-i", os.path.join(input_folder, "frame%04d.jpg"), 
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_video
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def datamosh(video_path, output_video, temp_folder="temp_frames", fps=30, intensity=5):
    extract_frames(video_path, temp_folder)
    glitch_frames(temp_folder)
    reassemble_video(temp_folder, output_video, fps)
    shutil.rmtree(temp_folder)
    print("Datamoshing complete! Output saved as", output_video)

if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_glitched.mp4"
    datamosh(input_video, output_video)
