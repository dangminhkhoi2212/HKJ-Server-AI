import math
import pickle
import numpy as np
import faiss
from PIL import Image
import image_extract
import matplotlib.pyplot as plt

# Define the image to be searched
search_image = "testimage/test1.jpg"

# Initialize model
model = image_extract.get_extract_model()

# Extract features for the search image
search_vector = image_extract.extract_vector(model, search_image)

# Load 4700 vectors and paths from files
vectors = pickle.load(open("vectors.pkl", "rb"))
paths = pickle.load(open("paths.pkl", "rb"))

# Ensure vectors are in the correct shape (if not already a 2D array)
if len(vectors.shape) == 1:
    vectors = vectors.reshape(1, -1)

# Ensure the search vector is also in 2D (as FAISS expects a batch of vectors)
search_vector = np.expand_dims(search_vector, axis=0)

# Load the FAISS index (assuming you have already created and saved it)
index = faiss.read_index("faiss_index.index")

# Perform the search with FAISS to find the K nearest neighbors
K = 12  # You can change this value to any integer
distances, indices = index.search(search_vector, K)

# Retrieve the closest images and their distances
nearest_images = [(paths[idx], distances[0][i]) for i, idx in
                  enumerate(indices[0])]

# Determine the number of rows and columns for the grid
columns = math.ceil(math.sqrt(K))  # Number of columns
rows = math.ceil(K / columns)  # Number of rows

# Display the nearest images
fig, axes = plt.subplots(rows, columns, figsize=(15, 5))

# Flatten the axes array for easy iteration (handle both 1D and 2D cases)
axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

for i in range(K):
    draw_image = nearest_images[i]
    axes[i].set_title(f"Distance: {draw_image[1]:.4f}", fontsize=8)
    axes[i].imshow(Image.open(draw_image[0]))
    axes[i].axis('off')  # Hide axes for a cleaner look

# Turn off any unused axes
for j in range(K, len(axes)):
    axes[j].axis('off')

fig.tight_layout()
plt.show()
