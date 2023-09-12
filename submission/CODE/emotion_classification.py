import pickle
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.tokenize import sent_tokenize

# Downloading nltk punkt tokenizer
nltk.download('punkt')

# Load articles
with open('text_content.pkl', 'rb') as file:
    articles = pickle.load(file)

# Split articles into sentences
sentences = [sent_tokenize(article) for article in articles]

# Initialize emotion classifier
tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
emotion_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
classifier = pipeline("text-classification", model=emotion_model, tokenizer=tokenizer)

# Classify emotions for each sentence
results = []

for article in sentences:
    detected_emotions = []
    for line in article:
        if line:
            emotion = classifier(line, truncation=True)
            # Keep only labels with score > 0.5
            emotions = [item['label'] for item in emotion if item['score'] > 0.5]
            detected_emotions.extend(emotions)
    results.append(detected_emotions)

# Calculate overall emotion counts for each article
emotion_counts = []

for i, detected_emotions in enumerate(results):
    count = {}
    num_lines = len(sentences[i])

    for emotion in detected_emotions:
        count[emotion] = count.get(emotion, 0) + 1

    # Normalize by number of lines
    for emotion in count:
        count[emotion] /= num_lines

    emotion_counts.append(count)

print(emotion_counts)
