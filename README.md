# **Graph-Based Behavioral and Emotional Analysis System**

## **Team Name**: *nepalensis*  
## **Team Members**:  
- Swikar Jaiswal (CM-2021)  
- Sriya Adhikari (BBIS-2024)

---

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [How It Works](#how-it-works)
3. [Beyond Safety: Fostering Positive Connections](#beyond-safety-fostering-positive-connections)
4. [Why It Matters](#why-it-matters)
5. [Installation](#installation)
6. [Credits](#credits)
7. [Special Credits](#special-credits)
8. [License](#license)

---

## **Project Overview**

Imagine a bustling corridor in a large office building or school. People are constantly moving through, chatting, or lost in thought. Now imagine there are places off-camera, like restrooms or isolated rooms, where things might go wrong — perhaps subtle acts of misconduct or bullying that leave someone feeling distressed. How can we identify these issues without asking someone to come forward and speak, a step that can feel intimidating or impossible?

This is where our system comes in. It’s like an invisible observer that never judges, only notices patterns. Here’s how it works:

---

### **How It Works**

#### **Noticing People and Their Interactions**:
- Cameras capture faces of individuals walking through shared spaces. Using advanced AI models like YOLO, each person is detected and given a unique identity (a “node” in our graph).
- The system maps out how these individuals move and interact by creating connections (“edges”) between them based on how close they are and how long they stay near each other.

#### **Listening to Emotions**:
- The system doesn’t eavesdrop on conversations or read minds — instead, it observes emotional expressions. It assigns each person an “emotional intensity” score ranging from -100 to 100, based on their visible mood.
- It keeps track of this score as people enter and leave specific areas, like restrooms. If someone’s mood changes drastically between entering and exiting, the system takes note.

#### **Spotting Patterns**:
- Let’s say five people were seen walking down a corridor, and one of them later shows signs of distress. The system creates a temporary graph of these five nodes and examines their proximity and interactions.
- If similar situations keep happening involving certain individuals, the system flags them as potential causes for concern, but only after seeing repeated patterns over time. No immediate accusations, just careful pattern recognition.

#### **Building a Larger Picture**:
- Over time, the system connects these smaller graphs into a bigger network, showing how people’s interactions influence emotional shifts across the environment.
- When consistent negative patterns emerge, it alerts administrators, not to point fingers, but to encourage investigation and offer support.

#### **Anonymous Support**:
- One powerful feature is its ability to inform individuals that they’re not alone. If multiple people show signs of distress linked to the same pattern, they can be encouraged to come forward together, making it easier to address issues without fear of isolation.

---

### **Beyond Safety: Fostering Positive Connections**

But this system isn’t just about preventing harm. It also helps organizations build stronger, happier communities:

- **Who Lifts Others Up?** Some individuals have a knack for improving everyone’s mood. The system identifies these positive influencers, making them great candidates for leadership or mentoring roles.
- **Who Works Well Together?** By analyzing group dynamics, it can recommend team formations that foster positivity and collaboration, helping students excel in projects or employees thrive in teams.
- **Where Are the Stress Points?** The system identifies areas or interactions that frequently lead to emotional downturns, enabling administrators to make targeted improvements.

---

### **Why It Matters**

This system isn’t about watching people — it’s about understanding them. It’s about creating environments where people feel safe and supported, where emotional well-being is as important as productivity. By relying on data and patterns, not individual accusations, it ensures fairness and objectivity. And by fostering positive connections, it helps communities thrive.

With this technology, we’re not just solving problems — we’re building better spaces for everyone.

---

## **Installation**

To set up the environment for running the system, follow these steps:

1. **Clone the repository**:  
    ```bash
    git clone https://github.com/your-repo-name
    cd 2024-BIOHACKATHON-nepalensis
    ```

2. **Create the Conda environment**:  
    You can create the environment using the provided `env.yaml`:
    ```bash
    conda env create -f env.yaml
    ```

3. **Install the requirements**:  
    Install the required Python libraries with pip:
    ```bash
    pip install -r requirements.txt
    ```
4. **Run the Demo**:  
    You can simply run the demo.ipynb file or other files in the Attempts directory.
---

## **Credits**

- **YOLO**: For real-time object detection and face tracking.  
    [YOLO GitHub Repository](https://github.com/ultralytics/yolov5)

- **Emotion Detection**:  
    [Emotion Detection GitHub](https://github.com/George-Ogden/emotion)

- **FFmpeg**: For video processing and handling.  
    [FFmpeg Download](https://ffmpeg.org/download.html)

---

## **Special Credits**

- **OC & Mentors**: For the entire event and mentorship.
  - **Frienson Pradhan**
  - **Karishma Sharma** (for video)
  
- **Team Molecular Muses**:  
  - **Akshita Shrestha**
  - **Ditsu Baral** (for video)

- **Team Janai Maru**:  
  - **Ankur Baral** (for video)  
  - **Aayush Shrestha** (for coding references)

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Quick Links**

- [Demo Video Original](./Bully1Final.mp4)
- [Graph Network](./data/output/GraphWithEmotions.mp4)
