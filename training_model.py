import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Load the dataset from CSV
df = pd.read_csv('Book_Dataset.csv')

# Preparing the data
X = df[['Keyword 1', 'Keyword 2', 'Keyword 3', 'Keyword 4']]  # Features (keywords)
y = df['Genre']  # Target (Genre)

# Convert the keywords into a single string
X['Keywords'] = X.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Vectorize the keywords into a bag-of-words representation
vectorizer = CountVectorizer()

# Convert the keywords into the vectorized form
X_vectorized = vectorizer.fit_transform(X['Keywords'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model and vectorizer to a pickle file
with open('genre_predictor.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)

print("Model trained and saved as genre_predictor.pkl")
