import re

def extract_emotion_and_percentage(text):
    # Define a regular expression pattern to match emotion and percentage
    pattern = r"([a-zA-Z]+)\s\((\d+(\.\d+)?)%\)"
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    if match:
        emotion = match.group(1)  # Extract emotion (e.g., 'sad')
        percentage = float(match.group(2))  # Extract percentage and convert to float
        return emotion, percentage
    else:
        return None, None  # Return None if no match is found
    
def node_data_extractor(data):
    node_data = {f"{a}":[] for a in range(5)}
    print(node_data)
    for frame in data:
        detections = frame['detections']
        node_data_temp = {f"{a}":[] for a in range(5)}
        for node in detections:
            emotion, perc = extract_emotion_and_percentage(node['emotion'])
            multiplier = 1
            if emotion.find('sad')>-1: multiplier = -1
            if emotion.find('angry')>-1: multiplier = -1.5
            perc = float(perc)*multiplier
            node_data_temp[str(node['node_unique_id'])].append(perc)
        for node in node_data_temp.keys():
            node_perc_values = node_data_temp[node]
            length = len(node_perc_values)
            perc = sum(node_perc_values)/length if length>0 else 0
            node_data[node].append(perc)
            

if __name__ == '__main__':
    # Example usage
    text = 'sad (45.1%)'
    emotion, percentage = extract_emotion_and_percentage(text)

    print(f"Emotion: {emotion}, Percentage: {percentage}")
