from flask import Flask, render_template,  redirect, url_for, request
import cv2
import numpy as np
import os
import matplotlib.image as mimg
from scipy.cluster.vq import kmeans
import pandas as pd
from scipy.spatial.distance import cdist
import openvino as ov
from notebook_utils import download_file, device_widget
import datetime
import time  # Optional for adding a delay


app = Flask(__name__)

ir_path = "static/vmodel_wound.xml"
core = ov.Core()
model = core.read_model(ir_path)
device = device_widget()
compiled_model = core.compile_model(model=model, device_name=device.value)
input_key = compiled_model.input(0)
output_key = compiled_model.output(0)



camera = cv2.VideoCapture(0)  # Initialize webcam


def overlay_mask_boundary(image, mask):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # mask = mask.squeeze(axis=-1)
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlayed_image = image.copy()
    cv2.drawContours(overlayed_image, contours, -1, (255, 0, 0), 2)
    return overlayed_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/open_camera', methods=['GET'])
def open_camera():
    # Open camera feed
    cleanup_old_images(folder="static/image", threshold=60)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Could not access the webcam.", 500

    print("Press 'c' to capture the image, or 'q' to quit.")
    captured_image = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Determine the desired square dimensions
        square_dim = min(frame.shape[0], frame.shape[1])  # Take the smaller dimension

        # Crop the frame to the square shape
        y_center, x_center = frame.shape[0] // 2, frame.shape[1] // 2
        y_start = y_center - square_dim // 2
        y_end = y_center + square_dim // 2
        x_start = x_center - square_dim // 2
        x_end = x_center + square_dim // 2
        square_frame = frame[y_start:y_end, x_start:x_end]

        square_frame = cv2.flip(square_frame, 1)

        # Resize to the desired display size (optional)
        display_size = 512
        square_frame_resized = cv2.resize(square_frame, (display_size, display_size))

        # Display the square frame
        cv2.imshow('Webcam', square_frame_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Capture image
            captured_image = square_frame
            break
        elif key == ord('q'):  # Quit
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_image is not None:
        # Resize image to 224x224 before saving
        resized_image = cv2.resize(captured_image, (224, 224))

        # Save the resized image
        # Generate a unique filename using timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        image_name = f"captured_image_{timestamp}.png"
        image_path = os.path.join("static", "image", image_name)

        cv2.imwrite(image_path, resized_image)

        # Pass the unique image name back to the template
        return render_template('index.html', image_name=image_name)


        # # Pass image name to the template for displaying
        # return render_template('index.html', image_name='captured_image.png')

    return redirect(url_for('index'))

@app.route('/process_image', methods=['GET'])
def process_image():

    # Retrieve the unique image name from query parameters
    image_name = request.args.get('image_name', None)

    if image_name is None:
        return "Image not found", 404

    image_path = os.path.join("static", "image", image_name)

    if not os.path.exists(image_path):
        return "Image not found", 404

    img = cv2.imread(image_path).astype("float32")
    img = img/255.0
    img = np.expand_dims(img, 0)

    y_pred = compiled_model(img)[output_key]
    y_pred = (y_pred>0.05).astype(np.uint8)
    print(y_pred.shape)
    y_pred = y_pred.squeeze(axis=0)
    # y_pred = y_pred.squeeze(axis=-1)
    # kernel = np.ones((3, 3), np.uint8) 
    # y_pred = cv2.morphologyEx(y_pred, cv2.MORPH_OPEN, kernel, iterations=1) 
    overlay = overlay_mask_boundary(cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB), y_pred)

    # overlay_path = "./static/image/overlayed_image.png"
    overlayed_name = f"overlayed_{image_name.split('_')[-1]}"  # Overlay name
    overlay_path = os.path.join("static", "image", overlayed_name)
    cv2.imwrite(overlay_path, overlay)

    image = cv2.imread(image_path)
    masked_image = (image * y_pred)
    # masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    # masked_image_path = "./static/image/masked_image.png"
    mask_name = f"mask_{image_name.split('_')[-1]}"  # Mask name from timestamp
    masked_image_path =  os.path.join("static", "image", mask_name)
    cv2.imwrite(masked_image_path, masked_image)

    batman_image = mimg.imread(masked_image_path)
    mask = np.squeeze(y_pred,-1)
    masked_pixels = batman_image[mask == True]
    r, g, b = masked_pixels[:, 0], masked_pixels[:, 1], masked_pixels[:, 2]
    batman_df = pd.DataFrame({'red': r, 'green': g, 'blue': b})
    cluster_centers, _ = kmeans(batman_df[['red', 'green', 'blue']], 3)

    dominant_colors = []    
    for cluster_center in cluster_centers:
        red, green, blue = cluster_center
        dominant_colors.append((
            max(0, min(255, red)),  # Ensure within 0-255 range
            max(0, min(255, green)),  # Ensure within 0-255 range
            max(0, min(255, blue))  # Ensure within 0-255 range
        ))

    input_dominant_colors = np.array(dominant_colors)

    def calculate_distance(input_colors, category_colors):
    # Calculate Euclidean distance between each input color and each category color
        distance_matrix = cdist(input_colors, category_colors, metric='euclidean')
        return distance_matrix
    
    healthy_colors = pd.read_csv('./static/healthy.csv')  
    ischemic_colors = pd.read_csv('./static/infected.csv')  
    infected_colors = pd.read_csv('./static/ischemic.csv')

    healthy_colors_array = healthy_colors[['red', 'green', 'blue']].values
    ischemic_colors_array = ischemic_colors[['red', 'green', 'blue']].values
    infected_colors_array = infected_colors[['red', 'green', 'blue']].values
    
    healthy_distance = calculate_distance(input_dominant_colors, healthy_colors_array)
    ischemic_distance = calculate_distance(input_dominant_colors, ischemic_colors_array)
    infected_distance = calculate_distance(input_dominant_colors, infected_colors_array)

    min_healthy_distance = np.min(healthy_distance)
    min_ischemic_distance = np.min(ischemic_distance)
    min_infected_distance = np.min(infected_distance)

    if min_healthy_distance <= min_ischemic_distance and min_healthy_distance <= min_infected_distance:
        category = "Healthy"
    elif min_ischemic_distance <= min_healthy_distance and min_ischemic_distance <= min_infected_distance:
        category = "Ischemic"
    else:
        category = "Infected"

    diagnosis = (f"The wound is in '{category}' category.")
    return render_template('index.html', image_name=image_name, overlayed_name=overlayed_name, diagnosis=diagnosis)

def cleanup_old_images(folder="static/image", threshold=60):
    """Delete images older than the threshold in seconds."""
    now = time.time()
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and file_path.endswith(".png"):
            if now - os.path.getmtime(file_path) > threshold:
                os.remove(file_path)
                print(f"Deleted old image: {file_path}")