import os, re
from matplotlib.path import Path
from PIL import Image
import numpy as np
from scipy.ndimage import binary_dilation, label, center_of_mass
import numpy as np
from PIL import Image, ImageDraw

class PatchExtractor:
    def __init__(self, patch_size, annotation_label, stride=1):
        self.patch_size = patch_size
        self.annotation_label = annotation_label
        self.stride = stride



    def save_centers(self, centers, image, window_size=64):
        # Convert the PIL Image to an RGBA image (if not already in that mode)
        image_rgba = image.convert("RGBA")
        
        # Create a transparent overlay
        overlay = Image.new("RGBA", image_rgba.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        img_array = np.array(image)
        half_window = window_size // 2
        for center in centers:
            y, x = center
            # Calculate the bounds of the window, ensuring they are within the image boundaries
            y_min = max(0, int(y) - half_window)
            y_max = min(img_array.shape[0], int(y) + half_window)
            x_min = max(0, int(x) - half_window)
            x_max = min(img_array.shape[1], int(x) + half_window)
            # Extract the window
            window = img_array[y_min:y_max, x_min:x_max, :]
            # Draw a semi-transparent rectangle
            draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0, 128))

            # return  window
            # Check if the window needs padding
            # if window.shape[0] < window_size or window.shape[1] < window_size:
            #     # Pad the window to make it 16x16
            #     window = np.pad(window, ((0, max(0, window_size - window.shape[0])), 
            #                             (0, max(0, window_size - window.shape[1]))), 
            #                     mode='constant', constant_values=0)
            img = Image.fromarray(window)
            # img.save(f"temp/center_{int(y)}_{int(x)}.png")
        combined = Image.alpha_composite(image_rgba, overlay)
        combined.save("temp/centers.png")



    def count_and_center_connected_areas(self, mask):
        # Label connected components
        labeled_array, num_features = label(mask)
        
        # Calculate the center of mass for each component
        centers = center_of_mass(mask, labels=labeled_array, index=np.arange(1, num_features+1))
        
        return num_features, centers

# Example usage
# num_connected_areas, centers = count_and_center_connected_areas(boundries)
# print(f"Number of connected areas: {num_connected_areas}")
# print("Centers of each area:", centers)
        
    def extend_pixels(self, arr):
        # Create a new array to store the result
        extended = np.zeros_like(arr)
        # Define the neighborhood size
        neighborhood_size = 5
        # Iterate over each element in boundries
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                if arr[y, x] > 0:  # Check if the point is positive
                    # Calculate the neighborhood bounds, ensuring they are within the array boundaries
                    y_min = max(0, y - neighborhood_size)
                    y_max = min(arr.shape[0], y + neighborhood_size + 1)
                    x_min = max(0, x - neighborhood_size)
                    x_max = min(arr.shape[1], x + neighborhood_size + 1)
                    
                    # Mark the neighborhood in the extended array
                    extended[y_min:y_max, x_min:x_max] = 255
        return extended
    
    def export_patches(self, slide, overlay_image, label , area, file_name):

        x, y, width, height, *path = area if len(area) == 5 else area + [None]
        if path is None:
            print("No path found for the area. Skipping...")
            return
        path = path[0]

        path.vertices -= np.array([x, y])
        overlay_array = np.array(overlay_image)
        positive, boundries, negatives  = [overlay_array[:,:,i] for i in range(overlay_array.shape[2])]

        binary_mask = boundries == 255
        
        # Define the structuring element for dilation (extend by 5 pixels)
        structuring_element = np.ones((11, 11), dtype=bool)  # 5 pixels in each direction + center
        
        # Apply binary dilation
        extended_mask = binary_dilation(binary_mask, structure=structuring_element)
        
        # Convert the extended mask back to an image format (255 for True, 0 for False)
        extended_image = np.where(extended_mask, 255, 0).astype(np.uint8)
        
        # Check if the extended pixels are within the path
        # This step is simplified and needs to be adjusted based on the actual 'path' definition
        y_indices, x_indices = np.where(extended_mask)
        points = np.vstack((x_indices, y_indices)).T  # Convert indices to points
        
        # Check if points are within the path
        if np.any(path.contains_points(points)):
            # Save the image if any of the extended pixels are within the path
            Image.fromarray(extended_image).save(output_path)
            print(f"Image saved to {output_path}")
        else:
            print("No extended pixels within the path")

        # extend_boundries = self.extend_pixels(boundries)

        #     # Iterate over the pixels in the bounding box
        # for x in range(0, width, self.patch_size):
        #     for y in range(0, height, self.patch_size):
        #         # Create a grid of points in the patch
        #         patch_points = [(x + dx, y + dy) for dx in range(self.patch_size) for dy in range(self.patch_size)]

        #         # Check if all points in the patch are inside the polygon
        #         if all(path.contains_point(point) for point in patch_points):
        #             # Extract the patch
        #             patch = slide.crop(x, y, self.patch_size, self.patch_size)
                    
        #             # Convert the patch to an image
        #             region = Image.fromarray(patch.numpy())

        #             # Save the patch
        #             region.save(os.path.join("temp", label, f'patch_{i}_{x}_{y}.png'))
        #             # counter += 1
        #             # if counter >= max_patches_per_area:
        #                 # break

        # pass

