import re
from collections import defaultdict
from matplotlib.path import Path
import cv2, os
import numpy as np
from distutils.util import strtobool
import matplotlib.path as mpath

def process_annotation(annotation_path):
    regions = defaultdict(list)
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()    
    for annotation in annotations:
        # Split the annotation by the label
        parts = re.split(r'(Tumor|Stroma|Other)', annotation)
        # Process each polygon
        for i in range(1, len(parts), 2):
            label = parts[i]
            points_str = parts[i + 1].strip('[]')
            points = re.findall(r'Point: (.*?), (.*?)(?:,|$)', points_str)
            # Convert the points to integers
            points = [(int(float(x.replace(']',''))), int(float(y.replace(']','')))) for x, y in points]
            # # Create a path from the points
            path = Path(points)
            # Calculate the bounding box of the polygon
            min_x = max(0, min(x for x, y in points))
            max_x = max(0, max(x for x, y in points))
            min_y = max(0, min(y for x, y in points))
            max_y = max(0, max(y for x, y in points))
            regions[label].append((min_x, min_y, max_x - min_x, max_y - min_y, path))
    return regions

def process_mask(mask_path, slide_dimensions):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_ratio_width = slide_dimensions[0] / mask.shape[1]
    mask_ratio_height = slide_dimensions[1] / mask.shape[0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = {"Mask": []}
    for idx, contour in enumerate(contours):
        x, y, width, height = map(int, cv2.boundingRect(contour))
        # Apply the ratios to the coordinates and dimensions
        x, y, width, height = x * mask_ratio_width, y * mask_ratio_height, width * mask_ratio_width, height * mask_ratio_height
        region = [round(x), round(y), round(width), round(height)]
        # regions['Mask'].append(region)

        # Create a matplotlib.path.Path from the contour
        contour = contour.reshape(-1, 2)
        contour[:, 0] = contour[:, 0] * mask_ratio_width
        contour[:, 1] = contour[:, 1] * mask_ratio_height
        path = Path(contour)
        region.append(path)
        regions['Mask'].append(region)
    return regions

# from matplotlib.path import Path
def find_rotation_angle(centroids):
    # Perform PCA on the centroids to find the main orientation
    centroids = np.array(centroids)
    mean = np.mean(centroids, axis=0)
    centered_centroids = centroids - mean
    cov_matrix = np.cov(centered_centroids.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    principal_component = eigenvectors[:, np.argmax(eigenvalues)]
    angle = np.arctan2(principal_component[1], principal_component[0])
    rotation_angle = np.degrees(angle)
    if rotation_angle < 0:
        rotation_angle -= 5
    else:
        rotation_angle += 5
    return rotation_angle



def process_mask_(mask_path, slide_dimensions):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_ratio_width = slide_dimensions[0] / mask.shape[1]
    mask_ratio_height = slide_dimensions[1] / mask.shape[0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = {"Mask": []}
    centroids = []
    for idx, contour in enumerate(contours):
        x, y, width, height = map(int, cv2.boundingRect(contour))
        # Apply the ratios to the coordinates and dimensions
        x, y, width, height = x * mask_ratio_width, y * mask_ratio_height, width * mask_ratio_width, height * mask_ratio_height
        region = [round(x), round(y), round(width), round(height)]

        # Create a matplotlib.path.Path from the contour
        contour = contour.reshape(-1, 2)
        contour[:, 0] = contour[:, 0] * mask_ratio_width
        contour[:, 1] = contour[:, 1] * mask_ratio_height
        path = Path(contour)
        region.append(path)
        regions['Mask'].append(region)
        # Calculate the centroid of the region
        centroid_x = x + width / 2
        centroid_y = y + height / 2
        centroids.append([centroid_x, centroid_y])



    # Rotate the mask based on the calculated angle
    rotation_angle = find_rotation_angle(centroids)
    print(rotation_angle)
    (h, w) = mask.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_mask = cv2.warpAffine(mask, M, (w, h))
    output_path = os.path.splitext(mask_path)[0] + '_rottated.png'
    cv2.imwrite(output_path, rotated_mask)

    # Calculate new centroids after rotation
    new_centroids = []
    for region in regions['Mask']:
        x, y, width, height = region[:4]
        centroid_x = x + width / 2
        centroid_y = y + height / 2
        new_centroid = np.dot(M, np.array([centroid_x, centroid_y, 1]))
        new_centroids.append(new_centroid[:2])

    # Sort regions based on new centroids
    sorted_regions = [region for _, region in sorted(zip(new_centroids, regions['Mask']), key=lambda x: (x[0][1], x[0][0]))]

    # Reverse the sorted regions to start numbering from the bottom-right
    sorted_regions.reverse()

    # Rearrange regions in a snake pattern
    snake_pattern_regions = []
    current_y = None
    row = []
    for region in sorted_regions:
        if current_y is None or region[1] == current_y:
            row.append(region)
        else:
            if len(snake_pattern_regions) % 2 == 0:
                row.reverse()
            snake_pattern_regions.extend(row)
            row = [region]
        current_y = region[1]
    if len(snake_pattern_regions) % 2 == 0:
        row.reverse()
    snake_pattern_regions.extend(row)

    regions['Mask'] = snake_pattern_regions

    # Load the original image to draw the indices
    image = cv2.imread(mask_path)
    # Draw the region index on the image
    for idx, region in enumerate(regions['Mask']):
        x, y, width, height = region[:4]
        cv2.putText(image, str(idx + 1), (int((x + (width // 2)) // mask_ratio_width), int((y + (height // 2)) // mask_ratio_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the image to disk
    output_path = os.path.splitext(mask_path)[0] + '_numbered.png'
    cv2.imwrite(output_path, image)
    # exit(0)
    return regions

def map_labels_to_indices(dataset):
    label_map = {}
    labels = []
    letters = set()
    numbers = set()
    for row in dataset:
        label, missing=row[-4:-2]
        if bool(missing) == False:
            print(f'The spot {label} is missing, skipping!')
        else:
            labels.append(label)
            letter, number = label.split('-')
            letters.add(letter)
            numbers.add(int(number))
    rows = max(numbers)
    reverse = False
    for idx, letter in enumerate(sorted(letters, reverse=True)):
        for number in range(1 , rows + 1 ):
            if f'{letter}-{number}' in labels:
                label_map[f'{letter}-{number}'] = f'{idx * rows + (number if reverse else rows - number + 1)}'
        reverse = not reverse
    
    return label_map

def process_qupath_dearray(qupath_dearray_path, pyvips_slide, tmaspot_size=3200):
    bounds_x = float(pyvips_slide.get('openslide.bounds-x')) if pyvips_slide.get_typeof('openslide.bounds-x') != 0 else 0
    bounds_y = float(pyvips_slide.get('openslide.bounds-y')) if pyvips_slide.get_typeof('openslide.bounds-y') != 0 else 0
    ratio_x = 1.0/float(pyvips_slide.get('openslide.mpp-x'))
    ratio_y = 1.0/float(pyvips_slide.get('openslide.mpp-y'))
    dataset = np.loadtxt(qupath_dearray_path, dtype=str, skiprows=1)
    label_map = map_labels_to_indices(dataset)
    regions = {}
    for row in dataset:
        label, missing, x, y = row[-4:]
        label = label_map[label]
        if(not strtobool(missing)):
            radius = tmaspot_size * 0.5
            x = int((float(x)*ratio_x) + bounds_x - radius)
            y = int((float(y)*ratio_y) + bounds_y - radius)
            w = tmaspot_size
            h = tmaspot_size
            center = (x + radius, y + radius)
            
            # Create the Path object for a circle
            path = mpath.Path.arc(0, 360)
            # Scale and translate the path to the desired center and radius
            vertices = path.vertices * radius + center
            codes = path.codes
            # Create the Path object with the scaled and translated vertices
            circle_path = mpath.Path(vertices, codes)
            regions[f'QuPath,{label}'] = [[x, y, w, h, circle_path]]
            
            # print(f"Extracting spot {label} at location", (x, y))
            # region = slide.crop(int(x - tmaspot_size * 0.5), int(y-tmaspot_size*0.5), tmaspot_size, tmaspot_size)
            # region = Image.fromarray(region.numpy())
            # region.save(f'temp/{label}.png')
        else:
            print(f'The spot {label} is missing, skipping!')
    return regions

# import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt
# import pyvips
# slide_path = r'D:\Develop\UBC\Datasets\R204brafv600e\Slides\braf_107.ndpi'
# qupath_path = r'D:\test.txt'
# slide = pyvips.Image.new_from_file(slide_path)
# read_qupath_dearry(qupath_path, slide)
