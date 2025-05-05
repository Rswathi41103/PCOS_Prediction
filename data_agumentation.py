import matplotlib.pyplot as plt
import cv2
import os

def display_images(image_paths, titles, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    for i, (img_path, title) in enumerate(zip(image_paths, titles)):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format for display
        ax = axes[i // cols, i % cols]
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Paths to sample images for the advanced method
Not_infected_images = [
    'D:\\Periods\\PCOS\\notinfected\\img_0_0.jpg',
    'D:\\Periods\\PCOS\\notinfected\\img_0_28.jpg'
]

infected_images = [
    'D:\\Periods\\PCOS\\infected\\img2.jpg',
    'D:\\Periods\\PCOS\\infected\\img4.jpg'
]

# Combine image paths and titles
image_paths = Not_infected_images + infected_images
titles = [
    'Not Infected 1', 'Not Infected 2',
    'Infected 1', 'Infected 2'
]

# Display images in a grid
display_images(image_paths, titles, rows=2, cols=2)
