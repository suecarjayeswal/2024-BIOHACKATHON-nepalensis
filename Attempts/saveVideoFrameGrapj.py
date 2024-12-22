import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import cv2
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# Load data
data = json.load(open('results.json'))

# Load video
video_path = 'Bully1Final.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Error opening video file")

# Initialize the graph
G = nx.Graph()

# Function to update graph for each frame
def update_graph(frame_data):
    G.clear()  # Clear the graph for each frame
    for detection in frame_data['detections']:
        node_id = detection['node_id']
        emotion = detection['emotion']
        percentage = detection['percentage']
        
        # Add or update the node
        G.add_node(node_id, emotion=emotion, percentage=percentage, bbox=detection['bbox'])
        
        # Add edges and distances
        for edge in detection['distances']:
            other_node = edge['node_id']
            distance = edge['distance']
            G.add_edge(node_id, other_node, distance=distance)

# Function to visualize the graph and video frame
def update_frame(frame_number):
    # Get the current frame from video
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        return
    
    # Clear the current graph visualization
    plt.clf()
    
    # Update graph for the current frame
    update_graph(data[frame_number])

    # Draw the graph with updated information
    plt.title(f"Frame: {frame_number}")
    pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistency
    
    # Draw nodes and labels
    for node in G.nodes:
        bbox = G.nodes[node]['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        face_crop = frame[y1:y2, x1:x2]
    
        if face_crop.size > 0:  # Check if face_crop is not empty
            # Convert BGR to RGB for matplotlib
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Create an annotation box for the face in the graph
            ab = AnnotationBbox(OffsetImage(face_crop_rgb, zoom=0.3), pos[node], frameon=False)
            plt.gca().add_artist(ab) 
    
    # Draw the graph itself
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=700)
    nx.draw_networkx_labels(G, pos, labels={n: f"{n}\n{G.nodes[n]['emotion']} ({G.nodes[n]['percentage']}%)" for n in G.nodes}, font_size=8)
    nx.draw_networkx_edges(G, pos)
    edge_labels = nx.get_edge_attributes(G, 'distance')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Add the video frame as an inset at the top-left corner
    inset_box = (0.05, 0.75, 0.2, 0.2)  # Position and size of inset (left, bottom, width, height)
    ax = plt.gca()
    ax_inset = ax.inset_axes(inset_box)
    ax_inset.imshow(frame[..., ::-1])  # Convert from BGR to RGB for imshow
    ax_inset.axis('off')  # Hide axis for video frame

    plt.axis("off")

# Set up the figure
fig = plt.figure(figsize=(10, 8))

# Create an animation
ani = FuncAnimation(fig, update_frame, frames=len(data), interval=100, repeat=False)

# Save the animation as a video
output_video_path = 'output_animation2.mp4'
writer = FFMpegWriter(fps=5, metadata=dict(artist='Your Name'), bitrate=1800)
ani.save(output_video_path, writer=writer)
print(f"Animation saved as {output_video_path}")

# Release the video capture object
cap.release()
