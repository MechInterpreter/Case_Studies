import cv2 as cv

def capture_video(stream_url, drop_rate):
    # Use OpenCV's VideoCapture to read from a UDP stream using FFMPEG
    cap = cv.VideoCapture(stream_url, cv.CAP_FFMPEG)
    
    # Check if the capture is successfully opened
    if not cap.isOpened():
        print("Error: Unable to open video stream.")
        return
    
    frame_count = 0
    
    # Loop through video frames
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Stream ended or cannot read frame.")
            break
        
        # Process only every 'drop_rate' frame
        if frame_count % drop_rate == 0:
            # Instead of showing the frame, save it to disk
            frame_filename = f"output_frames/frame_{frame_count}.jpg"
            cv.imwrite(frame_filename, frame)
            print(f"Frame {frame_count} saved to {frame_filename}")
        
        frame_count += 1
    
    # Release resources
    cap.release()