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

def detect_milestone_image_list(image_list, processor, detector, milestones, draw_boxes=False):
    texts = [milestones for i in range(len(image_list))]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    step_size = 4
    all_keywords = list()
    all_boxes = list()
    all_scores = list()
    all_labels = list()
    image_num = len(image_list)
    for start in range(0, image_num, step_size):
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
                    image_path = '{}_viewpoint_{}_draw'.format('free_vp', start+i)
                    image.save('./anno_images/' + image_path + '_anno.jpg')

    # assert len(all_labels) == 12
    # assert len(all_boxes) == 12
    # assert len(all_scores) == 12
    # assert len(all_keywords) == 12

    # keywords, boxes, scores, labels
    detection = list(zip(all_keywords, all_boxes, all_scores, all_labels))
    return detection

device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
detector = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14").to(device)

print("Initialized Processor and Detector")

print("Waiting for connection...")


from flask import Flask, request, jsonify
# Create a Flask app
app = Flask(__name__)
# Define a route that handles the POST request
@app.route("/", methods=["POST"])
def receive_object():
    # Get the binary data from the request
    data = request.get_data()
    # Unpickle the object
    obj = pickle.loads(data)
    # Do something with the object
    # print(obj)
    rgb_lists, milestones, draw_boxes = obj

    detection_results = detect_milestone_image_list(rgb_lists, processor, detector, milestones, draw_boxes)

    # Send a Python object to the client
    # data = pickle.dumps(detection_results)
    # c.send(data)

    response = {
        'detection_results': detection_results,
        'response_status': 'success'
    }

    # Return a success message
    return jsonify(response)
# Run the app
if __name__ == "__main__":
    app.run()
