import cv2
import os
import numpy as np

def generate_idle_video(image_path: str, output_path: str, duration_sec: int = 5, fps: int = 25):
    """
    Generates a looping idle video from a static image.
    In a fully advanced setup, you'd run this image through a weak idle animation 
    generator (like SadTalker or LivePortrait blinking) to create `idle.mp4`.
    
    Here, we generate a static video with subtle zooming/breathing effects just to prove the concept.
    """
    if not os.path.exists(image_path):
        print(f"Warning: Could not find image at {image_path}, using dummy black image.")
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.putText(img, "MISSING AVATAR", (100, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not load image from {image_path}, using dummy black image.")
            img = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.putText(img, "MISSING AVATAR", (100, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    h, w, _ = img.shape
    
    # We want to output a reasonable size, e.g. 512x512
    out_size = 512
    # Ensure square crop
    min_dim = min(h, w)
    start_y = (h - min_dim) // 2
    start_x = (w - min_dim) // 2
    cropped = img[start_y:start_y+min_dim, start_x:start_x+min_dim]
    base_frame = cv2.resize(cropped, (out_size, out_size))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_size, out_size))
    
    total_frames = duration_sec * fps
    
    print(f"Generating {duration_sec}s idle video to {output_path}...")
    
    for i in range(total_frames):
        # Add a very subtle "breathing" zoom effect to make it look alive
        scale = 1.0 + 0.02 * (1.0 - abs((i % (fps * 4)) / (fps * 2) - 1.0)) # 4 second breath cycle
        
        M = cv2.getRotationMatrix2D((out_size/2, out_size/2), 0, scale)
        frame = cv2.warpAffine(base_frame, M, (out_size, out_size))
        
        # Add some text just to show it's the idle loop
        cv2.putText(frame, "IDLE LOOP", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        
        out.write(frame)
        
    out.release()
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="../Gemini_Generated_Image_gx6wbwgx6wbwgx6w.png")
    parser.add_argument("--output", default="../client/static/idle.mp4")
    args = parser.parse_args()
    
    # Create static dir if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    generate_idle_video(args.image, args.output)
