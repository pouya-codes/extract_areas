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
        index = annotation.find(' [')
        label = annotation[:index]
        points_str = annotation[index+1:].strip('[]\n')
        # parts = re.split(r'(Tumor|Stroma|Other|Tumor Negative|Immune Cells|Necrosis)', annotation)
        # print(len(parts))
        # Process each polygon
        # for i in range(1, len(parts), 2):
            # label = parts[i]
            # points_str = parts[i + 1].strip('[]')
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
    # if rotation_angle < 0:
    #     rotation_angle -= 5
    # else:
    #     rotation_angle += 5
    return rotation_angle



def process_mask_(mask_path, slide_dimensions, num_cores= 80, num_rows= 7, num_columns= 12):
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
        if (width+height) < 2000:
            continue

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
    # Calculate the new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the translation part of the rotation matrix to center the image
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation with the adjusted matrix and new dimensions
    rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h))

    # rotated_mask = cv2.warpAffine(mask, M, (w, h))
    output_path = os.path.splitext(mask_path)[0] + '_rottated.png'

    mask_ratio_width_rotation = slide_dimensions[0] / rotated_mask.shape[1]
    mask_ratio_height_rotation = slide_dimensions[1] / rotated_mask.shape[0]
    

    # Calculate new centroids after rotation
    new_centroids = []
    for region in regions['Mask']:
        x, y, width, height = region[:4]
        # print(x, y, width, height)
        centroid_x = x + width / 2
        centroid_y = y + height / 2
        new_centroid = np.dot(M, np.array([centroid_x, centroid_y, 1]))
        new_centroids.append(new_centroid[:2])
        new_centroid = [int(new_centroid[0] // mask_ratio_width_rotation), int (new_centroid[1] // mask_ratio_height_rotation)]
        cv2.rectangle(rotated_mask, (new_centroid[0], new_centroid[1]), (new_centroid[0] + 20, new_centroid[1] + 20), (128), 2)
    cv2.imwrite(output_path, rotated_mask)
    exit(0)
    # Sort regions based on new centroids
    sorted_regions = [region for _, region in sorted(zip(new_centroids, regions['Mask']), key=lambda x: (round(x[0][1], -3) + round(x[0][0], -3)))]

    # Reverse the sorted regions to start numbering from the bottom-right
    sorted_regions.reverse()

    first_core = sorted_regions[0]
    last_core = sorted_regions[-1]



    # Find the inverse matrix to restore the rotation
    # Extract the rotation part and the translation part
    rotation_matrix = M[:, :2]
    translation_vector = M[:, 2]

    # Compute the inverse of the rotation part
    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)

    # Compute the inverse translation
    inverse_translation_vector = -np.dot(inverse_rotation_matrix, translation_vector)

    # Construct the inverse transformation matrix
    M_inverse = np.hstack([inverse_rotation_matrix, inverse_translation_vector.reshape(-1, 1)])

    image = cv2.imread(mask_path)
    centroid_first = np.array([first_core[0] + first_core[2] / 2, first_core[1] + first_core[3] / 2, 1])
    centroid_last = np.array([last_core[0] + last_core[2] / 2, last_core[1] + last_core[3] / 2, 1])
    a = int (centroid_first[0] // mask_ratio_width)
    b = int (centroid_first[1] // mask_ratio_height)
    cv2.rectangle(image, (a, b), (a + 20, b + 20), (128), -1)
    a = int (centroid_last[0] // mask_ratio_width)
    b = int (centroid_last[1] // mask_ratio_height)
    cv2.rectangle(image, (a, b), (a + 20, b + 20), (128), -1)
    output_path = os.path.splitext(mask_path)[0] + '_estimated_cores_.png'
    cv2.imwrite(output_path, image)


    centroid_first = np.dot(M, centroid_first ).astype(int)
    centroid_last = np.dot(M, centroid_last).astype(int)

    image = rotated_mask.copy()
    a = int (centroid_first[0] // mask_ratio_width_rotation)
    b = int (centroid_first[1] // mask_ratio_height_rotation)
    cv2.rectangle(image, (a, b), (a + 20, b + 20), (128), -1)
    a = int (centroid_last[0] // mask_ratio_width_rotation)
    b = int (centroid_last[1] // mask_ratio_height_rotation)
    cv2.rectangle(image, (a, b), (a + 20, b + 20), (128), -1)
    output_path = os.path.splitext(mask_path)[0] + '_estimated_cores.png'
    cv2.imwrite(output_path, image)
    exit(0)

    # Calculate the width between the first and last core
    width = centroid_first[0] - centroid_last[0]
    estimated_core_width = abs (width // num_columns)
    # Calculate the height between the first and last core   
    height = centroid_first[1] - centroid_last[1]
    # check if any other core has a smaller height then the last row is not complete
    is_last_row_complete = True
    for region in sorted_regions:
        if round(region[1], -2)< round(last_core[1], -2):
            height = last_core[1] + last_core[3] - first_core[1] - estimated_core_width
            is_last_row_complete = False
            break
    
    estimated_core_height = abs(height // (num_rows - (0 if is_last_row_complete else 1)))
    print(is_last_row_complete, estimated_core_height, estimated_core_width)

    # Draw the estimated cores on the original mask
    image = rotated_mask.copy()
    for row in range(num_rows):
        for col in range(num_columns):
            x = int(((first_core[0] + first_core[2] / 2 )  - (col * estimated_core_width)) // mask_ratio_width_rotation)
            y = int(((first_core[1] + first_core[3] / 2 ) - (row * estimated_core_height)) // mask_ratio_height_rotation)
            centroid = [x, y]
            # centroid = np.dot(M_inverse, np.array([x, y, 1])).astype(int)
            print(centroid)
            cv2.rectangle(image, (centroid[0], centroid[1]), (centroid[0] + 20, centroid[1] + 20), (128), 2)
    output_path = os.path.splitext(mask_path)[0] + '_estimated_cores.png'
    cv2.imwrite(output_path, image)


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

def process_qupath_dearray(qupath_dearray_path, pyvips_slide, tmaspot_size = 3200, dearray_mapping = None):
    bounds_x = float(pyvips_slide.get('openslide.bounds-x')) if pyvips_slide.get_typeof('openslide.bounds-x') != 0 else 0
    bounds_y = float(pyvips_slide.get('openslide.bounds-y')) if pyvips_slide.get_typeof('openslide.bounds-y') != 0 else 0
    ratio_x = 1.0/float(pyvips_slide.get('openslide.mpp-x'))
    ratio_y = 1.0/float(pyvips_slide.get('openslide.mpp-y'))
    dataset = np.loadtxt(qupath_dearray_path, dtype=str, skiprows=1)
    if dearray_mapping is None:
        label_map = map_labels_to_indices(dataset)
    else:
        label_map = dearray_mapping
    regions = {}
    for row in dataset:
        label, missing, x, y = row[-4:]
        if not label in label_map:
            print(f'The spot {label} is not in the mapping, skipping!')
            continue
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
