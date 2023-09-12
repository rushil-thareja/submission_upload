from models import imagebind_model
import data
import torch
import pickle

# Load text and image paths
with open('text_list.pkl', 'rb') as f:
    texts = pickle.load(f)

with open('image_paths.pkl', 'rb') as f:
    images = pickle.load(f)

# Print loaded data
for text in texts:
    print(text)
    print("-")
print(images)

# Update image paths
images = ["images/" + img for img in images]

# Check loaded data again
for text in texts:
    print(text)
    print("-")
print(images)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Transform data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(texts, device),
    ModalityType.VISION: data.load_and_transform_vision_data(images, device)
}

# Extract embeddings
with torch.no_grad():
    embeddings = model(inputs)

# Calculate similarity
def cosine_similarity(matrix):
    normed_matrix = matrix / matrix.norm(dim=-1, keepdim=True)
    return normed_matrix @ normed_matrix.T

print("Text x Text Cosine Similarity: ", cosine_similarity(embeddings[ModalityType.TEXT]))

# Convert to numpy arrays
vision_text_similarity = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1).cpu().numpy()
vision_embeddings = embeddings[ModalityType.VISION].cpu().numpy()
text_embeddings = embeddings[ModalityType.TEXT].cpu().numpy()

# Save results
with open('vision_text_similarity.pkl', 'wb') as f:
    pickle.dump(vision_text_similarity, f)

with open('vision_embeddings.pkl', 'wb') as f:
    pickle.dump(vision_embeddings, f)

with open('text_embeddings.pkl', 'wb') as f:
    pickle.dump(text_embeddings, f)
