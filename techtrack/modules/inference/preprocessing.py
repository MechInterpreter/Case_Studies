import cv2 as cv

def capture_video(filename, drop_rate):
    # Instantiate VideoCapture object
    cap = cv.VideoCapture(filename)
    
    # Establish frame count
    frame_count = 0
    
    # Ensure object is opened, read in frames, and increment frame count
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % drop_rate == 0:
            yield frame
            
        frame_count += 1
        
    cap.release()