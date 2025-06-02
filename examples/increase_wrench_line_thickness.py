import cv2
import numpy as np

def process_frame(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the green color range
    lower_green = np.array([50, 5, 5])
    upper_green = np.array([100, 255, 255])
    
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Use Canny edge detection
    edges = cv2.Canny(mask, 1, 3, apertureSize=3)

    # Find contours from the detected edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours
    for cnt in contours:
        # Approximate the contour to a polygon and filter out small or non-linear shapes
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        
        # Calculate contour properties to find long, thin lines
        if len(approx) >= 2:
            # Length of the contour
            length = cv2.arcLength(cnt, False)
            
            # Calculate a bounding box and use it to filter for thin shapes
            x, y, w, h = cv2.boundingRect(approx)
            
            # Check width to height ratio and filter for thin lines
            if length > 100 and (w < 10 or h < 10):
                # Draw a thicker line over the original one
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)  # Change the thickness as needed

    return frame

def main(input_video, output_video):
    # Open the input video
    cap = cv2.VideoCapture(input_video)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame = process_frame(frame)
        
        # Write the frame to the output video
        out.write(processed_frame)
    
    # Release everything once the job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = "videos/TODR_1080_clipchamped_small.mp4"  # Path to your input video
    output_video_path = "videos/TODR_1080_clipchamped_small_green_thicker.mp4"  # Path to save the output video
    main(input_video_path, output_video_path)