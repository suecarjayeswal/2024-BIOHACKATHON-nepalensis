import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # Corrected import

# Load data
data = json.load(open('updated_results.json'))

# Load video
video_path = 'Bully1Final.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Error opening video file")

# Initialize the graph
G = nx.Graph()

# Predefine 22 nodes with fixed positions (e.g., circular layout)
num_nodes = 5
radius = 0.9  # Radius of the circle
theta = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)  # Evenly spaced angles

# Create a fixed circular layout for nodes
fixed_positions = {i: (radius * np.cos(angle*2), radius * np.sin(angle*2)) for i, angle in enumerate(theta)}

# Function to update graph for each frame
def update_graph(frame_data):
    G.clear()  # Clear the graph for each frame
    for detection in frame_data['detections']:
        node_id = detection['node_unique_id']
        emotion = detection['emotion']
        percentage = detection['percentage']
        
        # Add or update the node
        G.add_node(node_id, emotion=emotion, percentage=percentage, bbox=detection['bbox'])
        
        # Add edges and distances
        for edge in detection['updated_distances']:
            other_node = edge['node_unique_id']
            distance = f"{float(edge['distance']):.2f}"
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
    
    # Draw nodes only for those that are detected
    for node in G.nodes:
        bbox = G.nodes[node]['bbox']
        centroid = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        new_bbox = (centroid[0] - 60, centroid[1] - 60, centroid[0] + 60, centroid[1] + 60)
        bbox = new_bbox
        x1, y1, x2, y2 = map(int, bbox)
        face_crop = frame[y1:y2, x1:x2]
    
        if face_crop.size > 0:  # Check if face_crop is not empty
            # Resize face_crop to 80% of its original size
            face_crop_resized = cv2.resize(face_crop, (int(face_crop.shape[1] * 0.8), int(face_crop.shape[0] * 0.8)))
            
            # Convert BGR to RGB for matplotlib
            face_crop_rgb = cv2.cvtColor(face_crop_resized, cv2.COLOR_BGR2RGB)
            
            # Get the fixed position for this node
            pos = fixed_positions[node]
            
            # Create an annotation box for the face in the graph
            ab = AnnotationBbox(OffsetImage(face_crop_rgb), pos, frameon=False)
            plt.gca().add_artist(ab) 
    
    # Draw the graph itself
    nx.draw_networkx_nodes(G, fixed_positions, node_color="skyblue", node_size=700)
    nx.draw_networkx_labels(G, fixed_positions, labels={n: f"{n}\n{G.nodes[n]['emotion']} ({G.nodes[n]['percentage']}%)" for n in G.nodes}, font_size=8)
    nx.draw_networkx_edges(G, fixed_positions)
    edge_labels = nx.get_edge_attributes(G, 'distance')
    nx.draw_networkx_edge_labels(G, fixed_positions, edge_labels=edge_labels)

    # Add the video frame as an inset at the top-left corner
    inset_box = (0.01, 0.89, 0.25, 0.25)  # Position and size of inset (left, bottom, width, height)
    ax = plt.gca()
    ax_inset = ax.inset_axes(inset_box)
    ax_inset.imshow(frame[..., ::-1])  # Convert from BGR to RGB for imshow
    ax_inset.axis('off')  # Hide axis for video frame

    plt.axis("off")

# Set up the figure
fig = plt.figure(figsize=(10, 8))

# Create an animation that updates every 0.1 second
ani = FuncAnimation(fig, update_frame, frames=len(data), interval=100, repeat=False)

# Show the animation
plt.show()

# Release the video capture object
cap.release()