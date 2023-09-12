import os
import pickle
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models import imagebind_model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Check and create directory if not exists
if not os.path.exists('images_cleaned'):
    os.makedirs('images_cleaned')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

# Load model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load data
df = pd.read_excel('news.xlsx', engine='openpyxl')

texts = []
headers = []
imgs = {}
cleaned_imgs = {}

stop_words = set(stopwords.words('english'))

prev_source = None
article_num = 1

for idx, row in df.iterrows():
    source = row['NEWS SOURCE']

    if prev_source and source != prev_source:
        article_num = 1

    content = row['NEWS CONTENT']
    header = row['NEWS HEADER']

    tokens = word_tokenize(header)
    cleaned_header = ' '.join(tokens).replace('\n', ' ')
    headers.append(cleaned_header)

    tokens_content = word_tokenize(content)
    cleaned_content = ' '.join(tokens_content).replace('\n', ' ')
    texts.append(cleaned_content)

    img_path = os.path.join('images', f"{source}_{article_num}.png")
    img = np.array(Image.open(img_path))
    imgs[idx] = img

    img_pil = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
    img_torch = transform(img_pil).unsqueeze(0).to(device)

    cleaned_img_path = os.path.join('images_cleaned', f"{source}_{article_num}.png")
    img_pil.save(cleaned_img_path)
    cleaned_imgs[idx] = cleaned_img_path

    prev_source = source
    article_num += 1

# Saving processed data
with open('text_list.pkl', 'wb') as file:
    pickle.dump(headers, file)

with open('image_paths.pkl', 'wb') as file:
    pickle.dump(list(cleaned_imgs.values()), file)

with open('text_content.pkl', 'wb') as file:
    pickle.dump(texts, file)
