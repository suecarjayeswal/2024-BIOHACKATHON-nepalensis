import json
import numpy as np

# Load the results.json file
with open('results.json', 'r') as f:
    results = json.load(f)

# Function to check intersection of two bounding boxes
def bbox_intersection(bbox1, bbox2):
    # Calculate the intersection area between two bounding boxes
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    # If there's no intersection, return 0 area
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0

    # Calculate intersection area
    intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    return intersection_area

# Define the segment lists
lists = [
    [(a, 1) for a in range(5, 10)],  # list_1
    [(a, 1) for a in range(0, 5)],    # list_2
    [(6, 2), (7, 2), (5, 3), (6, 3)], # list_3
    [(3, 2), (4, 2), (5, 2)],         # list_4
    [(0, 2), (1, 2), (2, 2)]          # list_5
]

# Create a dictionary of node_id -> node_unique_id
for frame_data in results:
    frame = frame_data["frame"]
    detections = frame_data["detections"]
    
    # Initialize a list to hold the node_unique_id for each node
    node_unique_id_dict = {}
    
    # For each detection, calculate the segment with the largest intersection
    for detection in detections:
        node_id = detection["node_id"]
        bbox = detection["bbox"]
        
        # Calculate which list segment has the largest intersection
        max_intersection = 0
        node_unique_id = -1
        
        for i, segment_list in enumerate(lists):
            for segment in segment_list:
                # Define the bounding box for the segment
                segment_bbox = (segment[0] * 192, segment[1] * 270, (segment[0] + 1) * 192, (segment[1] + 1) * 270)  # Adjust this size
                intersection = bbox_intersection(bbox, segment_bbox)
                
                if intersection > max_intersection:
                    max_intersection = intersection
                    node_unique_id = i  # Assign the unique ID based on the list
        
        # Update the detection with the new node_unique_id
        node_unique_id_dict[node_id] = node_unique_id
        detection["node_unique_id"] = node_unique_id

    for detection in detections:    
        # Now update the distances with the node_unique_id
        updated_distances = []
        for distance_data in detection["distances"]:
            other_node_id = distance_data["node_id"]
            other_node_unique_id = node_unique_id_dict.get(other_node_id, None)
            if other_node_unique_id is not None:
                updated_distances.append({
                    "node_unique_id": other_node_unique_id,
                    "distance": distance_data["distance"]
                })
        
        detection["updated_distances"] = updated_distances

# Save the updated results back to a file
with open('updated_results.json', 'w') as f:
    json.dump(results, f, indent=4)

