import cv2
import numpy as np

# Configuration
video_path = "Bully1Final.mp4"  # Replace with your video file path
frame_number_to_display = 119   # Frame number to display
rows, cols = 4, 10  # Number of segments (m x n grid)

# Colors for the lists (RGBA format for transparency)
colors = [
    (255, 0, 0, 100),  # Red with transparency
    (0, 255, 0, 100),  # Green with transparency
    (0, 0, 255, 100),  # Blue with transparency
    (255, 255, 0, 100),  # Yellow with transparency
    (0, 255, 255, 100),  # Cyan with transparency
]

# Example lists of segments (x, y) positions
list_1 = [(a,1) for a in range(5,10)]
list_2 =  [(a,1) for a in range(0,5)]
list_3 = [(6, 2),(7,2),(5,3),(6,3)]
list_4 = [(3,2),(4,2),(5,2)]
list_5 = [(0,2),(1,2),(2,2)]
lists = [list_1, list_2, list_3, list_4, list_5]

# Open video and read frame dimensions
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Error opening video file")

# Read the specific frame
for i in range(frame_number_to_display):
    ret, frame = cap.read()
if not ret:
    raise IOError("Error reading the video file")

# Get video frame dimensions
frame_height, frame_width = frame.shape[:2]
# segment_width = 192
segment_width = frame_width // cols
# segment_height = 270
segment_height = frame_height // rows
# print(segment_height,segment_width)


# Create an overlay for the segments
overlay = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)

# Draw segments, assign colors, and add text inside each segment
for y in range(rows):
    for x in range(cols):
        # Determine if this segment belongs to any list
        segment_pos = (x, y)
        color = (0, 0, 0, 255)  # Default to black
        for i, lst in enumerate(lists):
            if segment_pos in lst:
                color = colors[i]
                break
        
        # Define the segment rectangle
        x1, y1 = x * segment_width, y * segment_height
        x2, y2 = x1 + segment_width, y1 + segment_height
        
        # Draw the rectangle with the assigned color
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Overlay the (x, y) value inside each segment
        segment_text = f"({x},{y})"
        font = cv2.FONT_HERSHEY_DUPLEX
        text_size = cv2.getTextSize(segment_text, font, 0.9, 1)[0]
        text_x = x1 + (segment_width - text_size[0]) // 2  # Center the text horizontally
        text_y = y1 + (segment_height + text_size[1]) // 2  # Center the text vertically
        cv2.putText(overlay, segment_text, (text_x, text_y), font, 0.9, (255, 255, 255, 255), 1, cv2.LINE_AA)

# Draw white lines for grid divisions
for row in range(1, rows):
    y = row * segment_height
    cv2.line(overlay, (0, y), (frame_width, y), (255, 255, 255, 255), thickness=2)  # Horizontal line

for col in range(1, cols):
    x = col * segment_width
    cv2.line(overlay, (x, 0), (x, frame_height), (255, 255, 255, 255), thickness=2)  # Vertical line

# Combine the overlay with the original frame
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)  # Convert frame to BGRA for transparency
colored_frame = cv2.addWeighted(frame, 1, overlay, 0.5, 0)

# Display the result
cv2.imwrite("segmented_frame_with_lines_and_text.png", colored_frame)

# Release the video capture object
cap.release()
