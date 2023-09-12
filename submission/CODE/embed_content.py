import pickle
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import torch

# Load articles from pickle
with open('text_content.pkl', 'rb') as file:
    articles = pickle.load(file)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('mrm8488/bert-mini-finetuned-age_news-classification')
bert_model = BertForSequenceClassification.from_pretrained('mrm8488/bert-mini-finetuned-age_news-classification')

predicted_categories = []
category_probs = []
article_embeddings = []

# Ensure model is in evaluation mode
bert_model.eval()

# Process each article
for article in articles:
    print(article)
    input_data = tokenizer(article, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        model_output = bert_model(**input_data, output_hidden_states=True)
        logits = model_output.logits
        probs = F.softmax(logits, dim=1)

        for i in range(logits.shape[0]):
            category_idx = torch.argmax(logits[i], dim=0).item()
            predicted_categories.append(category_idx)
            category_probs.append(probs[i].tolist())

        hidden_states = model_output.hidden_states
        article_embeddings.append(hidden_states[-1].squeeze().mean(dim=0).numpy())

print(predicted_categories)
print(category_probs)

# Save the generated embeddings
with open('news_embeddings.pkl', 'wb') as embed_file:
    pickle.dump(article_embeddings, embed_file)
