import argparse
import time
import math
import json
import torch
import cv2
from pathlib import Path
from emotion import detect_emotion, init
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    scale_coords, set_logging, create_folder
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace

def get_face_embedding(face_image):
    """Get a unique embedding for a given face image."""
    embedding = DeepFace.represent(face_image, model_name='VGG-Face', enforce_detection=False)
    return embedding[0]['embedding']  # Returns a 128D vector (for VGG-Face)

def get_face_match(new_face_embedding, existing_faces_embeddings, threshold=0.5):
    """Compare the new face embedding with existing embeddings to find the closest match."""
    for i, existing_embedding in enumerate(existing_faces_embeddings):
        similarity = cosine_similarity([new_face_embedding], [existing_embedding])
        if similarity > threshold:
            return i  # Return the index of the matched face
    return None  # No match found

def calculate_distance(bbox1, bbox2):
    """Calculate Euclidean distance between the centers of two bounding boxes."""
    x1_center = (bbox1[0] + bbox1[2]) / 2
    y1_center = (bbox1[1] + bbox1[3]) / 2
    x2_center = (bbox2[0] + bbox2[2]) / 2
    y2_center = (bbox2[1] + bbox2[3]) / 2
    return math.sqrt((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2)

def process_detections_with_distances(opt):
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

    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    all_results = []
    frame_number = 0
    existing_faces_embeddings = []  # Store embeddings of faces detected in previous frames

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
                images = []
                for det_item in det:
                    xyxy, conf, cls = det_item[:4], det_item[4], int(det_item[5])
                    x1, y1, x2, y2 = map(int, xyxy)
                    images.append(im0s[y1:y2, x1:x2])

                if images:
                    emotions = detect_emotion(images, not opt.hide_conf)

                for i, det_item in enumerate(det):
                    xyxy, conf, cls = det_item[:4], det_item[4], int(det_item[5])
                    x1, y1, x2, y2 = map(int, xyxy)
                    emotion_label, emotion_confidence = emotions[i]

                    face_embedding = get_face_embedding(images[i])  # Get embedding of the face
                    matched_node_id = get_face_match(face_embedding, existing_faces_embeddings)

                    if matched_node_id is None:
                        # Assign a new node ID if no match found
                        node_id = len(existing_faces_embeddings)
                        existing_faces_embeddings.append(face_embedding)
                    else:
                        # Assign the existing node ID if a match is found
                        node_id = matched_node_id

                    detection_data = {
                        "node_id": node_id,
                        "bbox": [x1, y1, x2, y2],
                        "emotion": emotion_label,
                        "percentage": float(emotion_confidence),
                    }
                    frame_detections.append(detection_data)

        # Compute distances between all detected nodes
        for det in frame_detections:
            det["distances"] = [
                {"node_id": other["node_id"], "distance": calculate_distance(det["bbox"], other["bbox"])}
                for other in frame_detections if det["node_id"] != other["node_id"]
            ]

        all_results.append({"frame": frame_number, "detections": frame_detections})
        frame_number += 1

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device')
    parser.add_argument('--hide-img', action='store_true', help='hide results')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidence')

    opt = parser.parse_args()
    with torch.no_grad():
        opt.source = "Bully1Final.mp4"  # Change source to your video path
        results = process_detections_with_distances(opt=opt)
        print(json.dumps(results, indent=4))
        with open("results.json", "w") as f:
            json.dump(results, f)
        cv2.destroyAllWindows()
