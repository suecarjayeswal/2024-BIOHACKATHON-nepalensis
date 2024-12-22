import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load data
data = json.load(open('results.json'))

# Initialize the graph
G = nx.Graph()

# Function to update graph for each frame
def update_graph(frame_data):
    for detection in frame_data['detections']:
        node_id = detection['node_id']
        emotion = detection['emotion']
        percentage = detection['percentage']
        
        # Add or update the node
        G.add_node(node_id, emotion=emotion, percentage=percentage)
        
        # Add edges and distances
        for edge in detection['distances']:
            other_node = edge['node_id']
            distance = edge['distance']
            G.add_edge(node_id, other_node, distance=distance)

# Function to visualize the graph
def update_frame(frame_number):
    # Clear the current graph visualization
    plt.clf()
    
    # Update graph for the current frame
    update_graph(data[frame_number])

    # Draw the graph with updated information
    plt.title(f"Frame: {frame_number}")
    pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistency
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=700)
    nx.draw_networkx_labels(G, pos, labels={n: f"{n}\n{G.nodes[n]['emotion']} ({G.nodes[n]['percentage']}%)" for n in G.nodes})
    nx.draw_networkx_edges(G, pos)
    edge_labels = nx.get_edge_attributes(G, 'distance')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.axis("off")

# Set up the figure
fig = plt.figure(figsize=(10, 8))

# Create an animation that updates every 0.1 second
ani = FuncAnimation(fig, update_frame, frames=len(data), interval=100, repeat=False)

# Show the animation
plt.show()
