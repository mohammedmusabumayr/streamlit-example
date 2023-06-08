import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
data = pd.read_csv('reviews_dataset.csv')  # Replace 'reviews_dataset.csv' with your actual dataset file

# Step 2: Preprocess the data
# ... Perform any necessary data cleaning and preprocessing steps

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['review_text'], data['sentiment'], test_size=0.2, random_state=42)

# Step 4: Convert text to numerical features
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train the sentiment analysis model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Step 7: Save the trained model for future use
# ... Save the model using appropriate methods (e.g., joblib, pickle)

# Step 8: Use the trained model for sentiment analysis
def analyze_sentiment(text):
    vec = vectorizer.transform([text])
    sentiment = model.predict(vec)[0]
    return sentiment

# Step 9: Integrate the sentiment analysis model with your website or application
# ... Use the analyze_sentiment function to analyze the sentiment of user reviews or comments
# ... Display the sentiment or use it for further processing
