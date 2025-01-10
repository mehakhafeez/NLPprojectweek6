import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# File paths
DATASETS = [
    '1429_1.csv',
    'Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv',
    'Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv'
]

# Load datasets
df_list = []
for file in DATASETS:
    try:
        temp_df = pd.read_csv(file, on_bad_lines='skip')
        df_list.append(temp_df)
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Combine datasets
data = pd.concat(df_list, ignore_index=True)

# Extract relevant column
review_col = [col for col in data.columns if 'review' in col.lower()]
if review_col:
    data['review'] = data[review_col[0]]
else:
    raise ValueError("No review column found")

# Preprocessing function
def preprocess_text(text):
    """Preprocesses input text by converting to lowercase and removing short words."""
    if pd.isna(text):
        return ''
    text = text.lower()
    text = ' '.join(word for word in text.split() if len(word) > 2)
    return text

data['cleaned_review'] = data['review'].apply(preprocess_text)

# Sentiment analysis
def get_sentiment(text):
    """Performs sentiment analysis using TextBlob."""
    analysis = TextBlob(text)
    return analysis.polarity

data['sentiment_score'] = data['cleaned_review'].apply(get_sentiment)
data['sentiment'] = data['sentiment_score'].apply(
    lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral'
)

# TF-IDF and clustering
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(data['cleaned_review'])

kmeans = KMeans(n_clusters=5, random_state=42)
data['cluster'] = kmeans.fit_predict(tfidf_matrix)

silhouette_avg = silhouette_score(tfidf_matrix, data['cluster'])
print(f"Silhouette Score: {silhouette_avg}")

# Visualization: Sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(data['sentiment'])
plt.title('Sentiment Distribution')
plt.show()

# WordCloud visualization for each sentiment
for sentiment in ['positive', 'negative', 'neutral']:
    sentiment_text = ' '.join(data[data['sentiment'] == sentiment]['cleaned_review'])
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate(sentiment_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'WordCloud for {sentiment.capitalize()} Sentiment')
    plt.show()

# Model training
X = tfidf_matrix
y = data['sentiment']

# Logistic Regression
lr = LogisticRegression()
lr.fit(X, y)
lr_pred = lr.predict(X)

print("Logistic Regression Classification Report")
print(classification_report(y, lr_pred))

# Naive Bayes
nb = MultinomialNB()
nb.fit(X, y)
nb_pred = nb.predict(X)

print("Naive Bayes Classification Report")
print(classification_report(y, nb_pred))

# Confusion matrix for Logistic Regression
lr_cm = confusion_matrix(
    y, lr_pred, labels=['positive', 'neutral', 'negative']
)
sns.heatmap(
    lr_cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['positive', 'neutral', 'negative'],
    yticklabels=['positive', 'neutral', 'negative']
)
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# Ranking products
if 'product_name' in data.columns:
    product_sentiment = data.groupby('product_name').agg(
        avg_sentiment=('sentiment_score', 'mean'),
        review_count=('review', 'count')
    )
    product_sentiment['sentiment_rank'] = product_sentiment['avg_sentiment'].rank(ascending=False)
    print(product_sentiment.sort_values('sentiment_rank').head(10))
else:
    print("Product name column not found, skipping product ranking.")

# Fine-tuning T5 model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Prepare dataset for T5 fine-tuning
data['input_text'] = "summarize: " + data['cleaned_review']
data['target_text'] = data['sentiment']

class ReviewDataset(torch.utils.data.Dataset):
    """Custom Dataset for T5 Fine-Tuning."""
    def __init__(self, inputs, targets, tokenizer, max_len):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_text = self.inputs[index]
        target_text = self.targets[index]

        input_ids = self.tokenizer.encode(
            input_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).squeeze()

        target_ids = self.tokenizer.encode(
            target_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != 0).long(),
            "labels": target_ids
        }

max_len = 128
dataset = ReviewDataset(
    data['input_text'].tolist(),
    data['target_text'].tolist(),
    tokenizer,
    max_len
)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
