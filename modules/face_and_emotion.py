#### Desciption ####
# This script performs face detection, emotion recognition, and distance calculation from a video or image source.
# It uses YOLOv7 for face detection and DeepFace for face embeddings and emotion analysis.
# Key functionalities:
# - Assigns unique and consistent node IDs to faces using embeddings.
# - Detects emotions and their confidence levels for each face.
# - Computes pairwise Euclidean distances between detected faces.
# - Outputs results (bounding boxes, emotions, IDs, distances) in a structured JSON format.

#### Usage ####
# python face_and_emotion.py --source "data/samples/Bully1Final.mp4"


import argparse
import json
import math
import torch
import cv2
from pathlib import Path
from emotion import detect_emotion, init
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace


def calculate_distance(bbox1, bbox2):
    """Calculate Euclidean distance between the centers of two bounding boxes."""
    x1_center = (bbox1[0] + bbox1[2]) / 2
    y1_center = (bbox1[1] + bbox1[3]) / 2
    x2_center = (bbox2[0] + bbox2[2]) / 2
    y2_center = (bbox2[1] + bbox2[3]) / 2
    return math.sqrt((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2)


def get_face_embedding(face_image):
    """Get a unique embedding for a given face image."""
    embedding = DeepFace.represent(face_image, model_name='VGG-Face', enforce_detection=False)
    return embedding[0]['embedding']


def get_face_match(new_face_embedding, existing_faces_embeddings, threshold=0.5):
    """Compare the new face embedding with existing embeddings to find the closest match."""
    for i, existing_embedding in enumerate(existing_faces_embeddings):
        similarity = cosine_similarity([new_face_embedding], [existing_embedding])
        if similarity[0][0] > threshold:
            return i
    return None


def process_video(opt):
    source, imgsz = opt.source, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    set_logging()
    device = select_device(opt.device)
    init(device)
    half = device.type != 'cpu'

    model = attempt_load("weights/yolov7-tiny.pt", map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    if half:
        model.half()

    dataset = LoadStreams(source, img_size=imgsz, stride=stride) if webcam else LoadImages(source, img_size=imgsz, stride=stride)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    results = []
    frame_number = 0
    existing_faces_embeddings = []

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)

        frame_detections = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                images = [im0s[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] for *xyxy, conf, cls in det]

                if images:
                    emotions = detect_emotion(images, not opt.hide_conf)

                for i, det_item in enumerate(det):
                    xyxy = det_item[:4]
                    x1, y1, x2, y2 = map(int, xyxy)
                    emotion_label, emotion_confidence = emotions[i]

                    face_embedding = get_face_embedding(images[i])
                    matched_node_id = get_face_match(face_embedding, existing_faces_embeddings)

                    if matched_node_id is None:
                        node_id = len(existing_faces_embeddings)
                        existing_faces_embeddings.append(face_embedding)
                    else:
                        node_id = matched_node_id

                    detection = {
                        "node_id": node_id,
                        "bbox": [x1, y1, x2, y2],
                        "emotion": emotion_label,
                        "percentage": float(emotion_confidence)
                    }
                    frame_detections.append(detection)

        for det in frame_detections:
            det["distances"] = [
                {"node_id": other["node_id"], "distance": calculate_distance(det["bbox"], other["bbox"])}
                for other in frame_detections if det["node_id"] != other["node_id"]
            ]

        results.append({"frame": frame_number, "detections": frame_detections})
        frame_number += 1

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source (e.g., webcam or video path)')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e., 0 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidence scores')

    opt = parser.parse_args()
    with torch.no_grad():
        results = process_video(opt=opt)
        output_path = Path("data/output/results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_path.resolve()}")
