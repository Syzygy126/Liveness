import cv2

def capture_frames(camera_index, output_path, num_frames=100):
    cap = cv2.VideoCapture(camera_index)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    
    for i in range(num_frames):
        ret, frame = cap.read()

        
        if ret:
            
            image_filename = f"{output_path}/frame_{i+1:04d}.png"

            
            cv2.imwrite(image_filename, frame)

            print(f"Frame {i+1}/{num_frames} saved: {image_filename}")

    cap.release()

if __name__ == "__main__":
    camera_index = 0

    
    output_path = "/Users/syzygy/Documents/Liveness Detection/model_test/VideoFrame"

    
    num_frames = 100

    capture_frames(camera_index, output_path, num_frames)