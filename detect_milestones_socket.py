import socket
import pickle
import os
from PIL import Image
from PIL import ImageDraw

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'
os.environ['TOKENIZERS_PARALLELISM'] = 'True'

import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection



def detect_milestone(scan_id, viewpoint_id, processor, detector, milestones, draw_boxes=False):
    scan_dir = '/data2/zhaoganlong/datasets/Matterport3D/v1/scans/'
    image_list = list()
    for ix in range(36):
        image_filename = '{}_viewpoint_{}_res_960x720.jpg'.format(viewpoint_id, ix)
        image_path = os.path.join(scan_dir, scan_id, 'matterport_skybox_images', image_filename)
        if not os.path.exists(image_path):
            print("Unpresented Viewpoint: {} {}".format(scan_id, viewpoint_id))
            return None
        image = Image.open(image_path)
        image_list.append(image)
    texts = [milestones for i in range(len(image_list))]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    step_size = 4
    all_keywords = list()
    all_boxes = list()
    all_scores = list()
    all_labels = list()
    for start in range(0, 36, step_size):
        with torch.no_grad():
            inputs = processor(text=texts[start:start+step_size], images=image_list[start:start+step_size], return_tensors="pt").to(device)
            outputs = detector(**inputs)
        
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1] for image in image_list[start:start+step_size]]).to(device)
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
        # i = 0  # Retrieve predictions for the first image for the corresponding text queries
        # text = texts[i]
        boxes = [result['boxes'].cpu().tolist() for result in results]
        scores = [[score.item() for score in result['scores']] for result in results]
        labels = [[texts[0][label] for label in result['labels']] for result in results] # NOTE: here you have to make sure all texts element are the same.
        all_boxes.extend(boxes)
        all_scores.extend(scores)
        all_labels.extend(labels)
        all_keywords.extend([milestones] * len(labels))

        if draw_boxes:
            for i, image in enumerate(image_list[start:start+step_size]):
                draw = ImageDraw.Draw(image)
                draw.text((0, 0), f' '.join(texts[0]), fill='white')

                for j, label in enumerate(labels[i]):
                    box, score = boxes[i][j], scores[i][j]
                    xmin, ymin, xmax, ymax = box
                    score = score
                    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
                    draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")
                
                if len(labels[i]) > 0:
                    image_path = '{}_viewpoint_{}_draw'.format(viewpoint_id, start+i)
                    image.save('./anno_images/' + image_path + '_anno.jpg')

    assert len(all_labels) == 36
    assert len(all_boxes) == 36
    assert len(all_scores) == 36
    assert len(all_keywords) == 36

    # keywords, boxes, scores, labels
    detection = list(zip(all_keywords, all_boxes, all_scores, all_labels))
    return detection
        


# Create a socket object
s = socket.socket()

# Bind the socket to a port
port = 23456
s.bind(('', port))

# Listen for incoming connections
s.listen(5)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
detector = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14").to(device)

print("Initialized Processor and Detector")

print("Waiting for connection...")

while True:
    # Accept a connection
    c, addr = s.accept()
    print('Connected with', addr)

    # Receive a Python object from the client
    data = c.recv(1024*1024)
    obj = pickle.loads(data)
    print('Received object:', obj)

    scan_id, viewpoint_id, milestones, draw_boxes = obj

    detection_results = detect_milestone(scan_id, viewpoint_id, processor, detector, milestones, draw_boxes)

    # Send a Python object to the client
    data = pickle.dumps(detection_results)
    c.send(data)
    print('Sent object:', detection_results)

    # Close the connection
    c.close()