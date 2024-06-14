import pandas as pd
import numpy as np
from collections import defaultdict
import string
import os

def preprocess(review):
    stopwords = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with"}
    review = review.lower()
    review = review.translate(str.maketrans('', '', string.punctuation))
    words = review.split()
    words = [word for word in words if word not in stopwords]
    return words

def train(data):
    model = {
        "prior": {},
        "likelihood": {
            "helpful": defaultdict(lambda: 0),
            "not helpful": defaultdict(lambda: 0)
        },
        "total_words": {
            "helpful": 0,
            "not helpful": 0
        },
        "vocab": set()
    }

    helpful_count = sum(1 for _, label in data if label == "helpful")
    not_helpful_count = len(data) - helpful_count

    model["prior"]["helpful"] = helpful_count / len(data)
    model["prior"]["not helpful"] = not_helpful_count / len(data)

    for review, label in data:
        words = preprocess(review)
        for word in words:
            model["likelihood"][label][word] += 1
            model["total_words"][label] += 1
            model["vocab"].add(word)

    return model

def classify(review, model):
    processed_review = preprocess(review)
    log_prob_helpful = np.log(model["prior"]["helpful"])
    log_prob_not_helpful = np.log(model["prior"]["not helpful"])

    for word in processed_review:
        if word in model["vocab"]:
            log_prob_helpful += np.log(model["likelihood"]["helpful"][word])
            log_prob_not_helpful += np.log(model["likelihood"]["not helpful"][word])
        else:
            log_prob_helpful += np.log(1 / (model["total_words"]["helpful"] + len(model["vocab"])))
            log_prob_not_helpful += np.log(1 / (model["total_words"]["not helpful"] + len(model["vocab"])))

    if log_prob_helpful > log_prob_not_helpful:
        predicted_class = "helpful"
    else:
        predicted_class = "not helpful"

    return predicted_class

# Load the training data
reviews_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reviews_data.csv')
reviews_data = pd.read_csv(reviews_data_path).values.tolist()

# Train the Naive Bayes classifier
model = train(reviews_data)

# Load the test data
test_reviews_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_reviews.csv')
test_reviews = pd.read_csv(test_reviews_path).values.tolist()

# Classify test reviews and calculate accuracy
correct_predictions = 0
total_reviews = len(test_reviews)

for review, actual_label in test_reviews:
    predicted_label = classify(review, model)
    if predicted_label == actual_label:
        correct_predictions += 1

accuracy = correct_predictions / total_reviews
print(f"Accuracy: {accuracy:.2f}")
