import json
import networkx as nx

# Load data
data = json.load(open('data/output/results.json'))

# Initialize the graph
G = nx.Graph()

# Function to update and save graph data for each frame
def save_graph_data(frame_data, frame_number):
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
    
    # Save the graph structure to a file
    nx.write_gml(G, f"graph_frame_{frame_number}.gml")
    print(f"Saved graph for frame {frame_number}")

# Process and save graphs for all frames
for frame_number, frame_data in enumerate(data):
    save_graph_data(frame_data, frame_number)

print("All graphs have been saved.")
