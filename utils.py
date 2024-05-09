import re
from collections import defaultdict
import cv2
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
            # path = Path(points)
            # Calculate the bounding box of the polygon
            min_x = min(x for x, y in points)
            max_x = max(x for x, y in points)
            min_y = min(y for x, y in points)
            max_y = max(y for x, y in points)
            regions[label].append((min_x, min_y, max_x - min_x, max_y - min_y))
    return regions

def process_mask(mask_path, slide_dimensions):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_ratio_width = round(slide_dimensions[0] / mask.shape[1])
    mask_ratio_height = round(slide_dimensions[1] / mask.shape[0])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = {"Mask": []}
    for idx, contour in enumerate(contours):
        x, y, width, height = map(int, cv2.boundingRect(contour))
        # Apply the ratios to the coordinates and dimensions
        x, y, width, height = x * mask_ratio_width, y * mask_ratio_height, width * mask_ratio_width, height * mask_ratio_height
        region = [x, y, width, height]
        regions['Mask'].append(region)
    return regions