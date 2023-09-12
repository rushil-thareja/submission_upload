import pickle
import nltk
from transformers import pipeline
from nltk.tokenize import sent_tokenize

# Downloading nltk punkt tokenizer
nltk.download('punkt')

# Load articles
with open('text_content.pkl', 'rb') as file:
    articles = pickle.load(file)

# Split articles into sentences
sentences = [sent_tokenize(article) for article in articles]

# Initialize sentiment classifier
sentiment_classifier = pipeline('sentiment-analysis')

# Analyze sentiment for each sentence
results = []
for article in sentences:
    sentiments = [sentiment_classifier(line) for line in article if line]
    results.append(sentiments)

# Calculate overall sentiment for each article
scores = []

for sentiments in results:
    pos_count, neg_count = 0, 0
    for sentiment in sentiments:
        if sentiment[0]['label'] == "POSITIVE":
            pos_count += 1
        else:
            neg_count += 1

    total_count = pos_count + neg_count
    scores.append({
        "POS": pos_count / total_count,
        "NEG": neg_count / total_count
    })

print(scores)
