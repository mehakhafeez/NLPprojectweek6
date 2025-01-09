
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch
import nltk
nltk.download('punkt')
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from textblob import TextBlob
from rouge_score import rouge_scorer
import re
from nltk.translate.bleu_score import sentence_bleu

# %%
# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# %%
df1 = pd.read_csv('/content/1429_1.csv', quoting=3, on_bad_lines='skip')
# quoting=3 tells pandas to use the QUOTE_NONE strategy. This means that pandas will not treat any character as a quote character. This way if there are unclosed quotes, it ignores them.
# on_bad_lines='skip' replaces the deprecated 'error_bad_lines=False' to skip bad lines.
df2 = pd.read_csv('/content/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv', quoting=3, on_bad_lines='skip')
df3 = pd.read_csv('/content/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv', quoting=3, on_bad_lines='skip')

# %%
# Combine datasets into one
df = pd.concat([df1, df2, df3], ignore_index=True)

# %%
# Preprocess the text (remove stopwords, special characters, etc.)
def preprocess_text(text):
    # Check if the text is a string before processing
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in nltk.corpus.stopwords.words('english')])
        return text
    # If not a string (e.g., float), return it as is or handle it differently
    else:
        return str(text)  # Or handle NaN values appropriately

# %%
# Assuming the correct column name for reviews is 'reviews.text' based on later usage
df['review_clean'] = df['reviews.text'].apply(preprocess_text)

# %%
# Data Cleaning
df = df.drop_duplicates() # Changed 'data' to 'df'
df = df.dropna(subset=['reviews.text']) # Changed 'data' to 'df'
df['reviews.text'] = df['reviews.text'].str.strip() # Changed 'data' to 'df'
df['reviews.text'] = df['reviews.text'].str.replace(r'[^a-zA-Z\s]', '', regex=True) # Changed 'data' to 'df'

# %%
# Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['reviews.text'].apply(get_sentiment)


# %%
# Display the first few rows of the dataset with the new 'sentiment' column
print(df[['reviews.text', 'sentiment']].head(10))

# %%
# Sentiment Evaluation
X = df['reviews.text']
y = df['sentiment']

# %%
# Encode sentiments
y = y.map({'positive': 2, 'neutral': 1, 'negative': 0})

# %%
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# %%
# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# %%
# Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train_vec, y_train)

y_pred_lr = lr_model.predict(X_test_vec)

# %%
# Evaluation Metrics for Logistic Regression
print("Logistic Regression Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_lr, average='weighted'))
print("F1-Score:", f1_score(y_test, y_pred_lr, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# %%
# Confusion Matrix Plot for Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# %%
# Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

y_pred_nb = nb_model.predict(X_test_vec)

# %%
# Evaluation Metrics for Naive Bayes
print("Naive Bayes Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Precision:", precision_score(y_test, y_pred_nb, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_nb, average='weighted'))
print("F1-Score:", f1_score(y_test, y_pred_nb, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))

# %%
# Confusion Matrix Plot for Naive Bayes
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %%
# Task 2: Product Category Clustering using KMeans (Updated to 5 clusters)
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df['review_clean'])

# %%
# KMeans Clustering
pca = PCA(n_components=2)
# Fit PCA on the same data used for KMeans
X_pca = pca.fit_transform(X_tfidf.toarray())

kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(X_pca)

# %%
# Assign clusters to data
df['cluster'] = kmeans.predict(pca.transform(X_tfidf.toarray()))

# %%
# Print unique clusters and their corresponding data
num_clusters = 6  # Define num_clusters with the desired number of clusters
for cluster_id in range(num_clusters):
    print(f"\nCluster {cluster_id}:")
    cluster_data = df[df['cluster'] == cluster_id] # Changed 'data' to 'df'
# Print unique clusters and their corresponding data
num_clusters = 6  # Define num_clusters with the desired number of clusters
for cluster_id in range(num_clusters):
    print(f"\nCluster {cluster_id}:")
    cluster_data = df[df['cluster'] == cluster_id] # Changed 'data' to 'df'
    print(cluster_data[['reviews.text', 'cluster']].head(10))  # Print first 10 reviews in each cluster
    print(f"Total reviews in Cluster {cluster_id}: {len(cluster_data)}")

# %%
# Visualize Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans.labels_, palette='viridis')
plt.title("KMeans Clustering of Reviews")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()

# %%
# Task 3: Generative AI for Review Summarization and Product Recommendation (using T5)
# Fine-tune T5 for generating product reviews or summarizing
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

def generate_summary(text, max_length=150):
    input_text = f"summarize: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=max_length, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# %%
# Generate summary for a product category
sample_reviews = df[df['cluster'] == 0]['review_clean'].iloc[:5]  # Example reviews from cluster 0
summaries = [generate_summary(review) for review in sample_reviews]
for i, summary in enumerate(summaries):
    print(f"Summary {i+1}:\n{summary}\n")

# %%
def evaluate_generative_model(generated_summary, reference_summary):
    """
    Evaluates a generated summary against a reference summary using ROUGE and BLEU scores.

    Args:
        generated_summary (str): The generated summary.
        reference_summary (str): The reference summary.

    Returns:
        None: Prints the ROUGE and BLEU scores.
    """

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)

    print("ROUGE Scores:")
    print(f"ROUGE-1: {scores['rouge1']}")
    print(f"ROUGE-2: {scores['rouge2']}")
    print(f"ROUGE-L: {scores['rougeL']}")

    bleu_score = sentence_bleu([reference_summary.split()], generated_summary.split())
    print(f"BLEU Score: {bleu_score}")

# %%
# Summarize Reviews into Articles using T5 Model
def generate_summary(texts, model, tokenizer, max_input_length=512, max_output_length=150):
    inputs = tokenizer.encode("summarize: " + ' '.join(texts), return_tensors="pt", max_length=max_input_length, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_output_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# %%
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

print("T5 and tokenizer loaded successfully!")

# %%
# To view the first 5 elements of the summaries list:
print(summaries[:5])

# %%
# Generate summaries for each cluster
for i in range(6):
    cluster_data = df[df['cluster'] == i] # Changed 'data' to 'df'
    reviews = cluster_data['reviews.text'].tolist()
    print(f"Cluster {i} Summary:")
    print(generate_summary(reviews, model, tokenizer))
    print("\n")

# %%
# Install rouge-score if not already installed
# !pip install rouge-score

from rouge_score import rouge_scorer

# Rouge and BLEU Evaluation for Summaries
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
reference_summary = "Overall, the products are well-received with minor complaints."

for i in range(3):
    cluster_data = df[df['cluster'] == i] # Changed 'data' to 'df'
    reviews = cluster_data['reviews.text'].tolist()
    generated_summary = generate_summary(reviews, model, tokenizer)

    # Calculate ROUGE scores
    scores = scorer.score(reference_summary, generated_summary)
    print(f"Cluster {i} ROUGE Scores:")
    print(f"ROUGE-1: {scores['rouge1']}")
    print(f"ROUGE-2: {scores['rouge2']}")
    print(f"ROUGE-L: {scores['rougeL']}")

    # Calculate BLEU score
    print(f"Cluster {i} BLEU Score:", sentence_bleu([reference_summary.split()], generated_summary.split()))
    print("\n")

# %%
# Fine-tune T5 Model for Product Review Generation
class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, reviews, summaries, tokenizer, max_length=512):
        self.reviews = reviews
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        review = self.reviews[index]
        summary = self.summaries[index]

        inputs = self.tokenizer("summarize: " + review, max_length=self.max_length, truncation=True, return_tensors="pt")
        labels = self.tokenizer(summary, max_length=150, truncation=True, return_tensors="pt")

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }

# %%
# Prepare dataset for fine-tuning
summaries = ["Excellent product with high quality." for _ in range(len(df))]
dataset = ReviewsDataset(df['reviews.text'].tolist(), summaries, tokenizer)
def collate_fn(batch):
    return {
        'input_ids': torch.nn.utils.rnn.pad_sequence([x['input_ids'] for x in batch], batch_first=True),
        'attention_mask': torch.nn.utils.rnn.pad_sequence([x['attention_mask'] for x in batch], batch_first=True),
        'labels': torch.nn.utils.rnn.pad_sequence([x['labels'] for x in batch], batch_first=True)
    }

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500
)

def model_init():
    return T5ForConditionalGeneration.from_pretrained("t5-small")

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn
)

trainer.train()


# %%
# Test fine-tuned model on a sample
sample_input = "This product is amazing and exceeded all my expectations."
inputs = tokenizer.encode("summarize: " + sample_input, return_tensors="pt")

# Move inputs to the same device as the model
if torch.cuda.is_available():
    inputs = inputs.to(trainer.model.device)  # Move inputs to GPU

predicted_summary_ids = trainer.model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
predicted_summary = tokenizer.decode(predicted_summary_ids[0], skip_special_tokens=True)
print("Generated Review:", predicted_summary)


# %%
# Generate summaries for the fine-tuned model and evaluate using ROUGE and BLEU
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
reference_summary = "Excellent product with high quality."

# Sample input for testing the fine-tuned model
sample_input = "This product is amazing and exceeded all my expectations."
inputs = tokenizer.encode("summarize: " + sample_input, return_tensors="pt")

# Move inputs to the same device as the model
if torch.cuda.is_available():
    inputs = inputs.to(trainer.model.device)  # Move inputs to GPU if available

# Generate summary from the fine-tuned model
generated_summary_ids = trainer.model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
generated_summary = tokenizer.decode(generated_summary_ids[0], skip_special_tokens=True)

# Calculate ROUGE scores
scores = scorer.score(reference_summary, generated_summary)
print("ROUGE Scores for Fine-Tuned Model:")
print(f"ROUGE-1: {scores['rouge1']}")
print(f"ROUGE-2: {scores['rouge2']}")
print(f"ROUGE-L: {scores['rougeL']}")

# Calculate BLEU score
print(f"BLEU Score for Fine-Tuned Model:", sentence_bleu([reference_summary.split()], generated_summary.split()))


