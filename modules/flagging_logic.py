
import re
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# Function to extract emotion and percentage from a text
def extract_emotion_and_percentage(text):
    pattern = r"([a-zA-Z]+)\s\((\d+(\.\d+)?)%\)"
    match = re.search(pattern, text)
    
    if match:
        emotion = match.group(1)
        percentage = float(match.group(2))
        return emotion, percentage
    else:
        return None, None

# Function to calculate RSI for each node
def calculate_rsi(df, period=50):
    delta = df.diff()  # Calculate price change between consecutive periods
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()  # Average gain
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()  # Average loss
    
    rs = gain / loss  # Relative Strength
    rsi = 100 - (100 / (1 + rs))  # RSI formula
    return rsi

# Function to process data and extract influenced rate, save to file, and calculate RSI
def process_and_flag(data, period=50, save_to_file=False, filename="node_data.csv"):
    # Initialize node data dictionary
    node_data = {f"{a}": [] for a in range(5)}
    
    # Process the input data (e.g., frames with node detections)
    for frame in data:
        detections = frame['detections']
        node_data_temp = {f"{a}": [] for a in range(5)}
        
        for node in detections:
            emotion, perc = extract_emotion_and_percentage(node['emotion'])
            multiplier = 1
            if emotion.find('sad') > -1: multiplier = -1
            if emotion.find('angry') > -1: multiplier = -1.5
            perc = float(perc) * multiplier
            node_data_temp[str(node['node_unique_id'])].append(perc)
        
        for node in node_data_temp.keys():
            node_perc_values = node_data_temp[node]
            length = len(node_perc_values)
            perc = sum(node_perc_values) / length if length > 0 else 0
            node_data[node].append(perc)
    
    # Convert node data to DataFrame and save to CSV if requested
    df = pd.DataFrame(node_data)
    if save_to_file:
        df.to_csv(filename, header=[f'node_{i}' for i in range(5)])
    
    # Calculate RSI for each node
    rsi = calculate_rsi(df, period)
    
    # Optionally plot the RSI for each node
    plt.figure(figsize=(10, 8))
    for i in range(df.shape[1]):
        plt.subplot(df.shape[1], 1, i+1)
        plt.plot(df.index, rsi.iloc[:, i], label=f'RSI Node_{i}')
        plt.axhline(70, color='red', linestyle='--', label='Threshold (70)')
        plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        plt.title(f'Relative Strength Index for Node_{i}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return df, rsi

# Example usage
if __name__ == "__main__":
    # Example data (replace with actual frame data)

    data = pd.DataFrame(json.load(open('data/output/results.json')))
    
    # Process data and save to file
    node_data, rsi_values = process_and_flag(data, period=50, save_to_file=True)
    
    # Print node data and RSI values
    print(node_data)
    print(rsi_values)
