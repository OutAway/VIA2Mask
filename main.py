import json
import cv2
import numpy as np
import os

# Load your JSON data
with open('.json', 'r') as json_file:
    data = json.load(json_file)

# Path to the directory containing your images
image_dir = ' '

# Create an output directory for the masked images
output_dir = ' '
os.makedirs(output_dir, exist_ok=True)

# Iterate through each image and its regions
for image_filename, image_info in data["_via_img_metadata"].items():
    # Load the image
    image_path = os.path.join(image_dir, image_info["filename"])
    image = cv2.imread(image_path)

    # Create a mask
    mask = np.zeros(image.shape, dtype=np.uint8)
    alpha_mask = np.zeros((image.shape[0], image.shape[1], 4 ), dtype=np.uint8)
    for region in image_info["regions"]:
        points_x = region["shape_attributes"]["all_points_x"]
        points_y = region["shape_attributes"]["all_points_y"]
        points = np.array(list(zip(points_x, points_y)), np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], (255, 255, 255, 0))  # RGB = White, 0 = alpha 

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask)

    # Save the masked image
    output_filename = os.path.join(output_dir, image_info["filename"])
    cv2.imwrite(output_filename, masked_image)

print("Masking completed. Masked images are saved in the output directory.")
