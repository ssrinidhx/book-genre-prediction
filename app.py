from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and vectorizer
with open('genre_predictor.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/book', methods=['GET', 'POST'])
def book():
    if request.method == 'POST':
        # Get the description from the form
        description = request.form['description']
        
        # Convert the description into the same format as the training data
        desc_vectorized = vectorizer.transform([description])
        
        # Predict the genre using the trained model
        predicted_genre = model.predict(desc_vectorized)
        
        return render_template('book.html', genre=predicted_genre[0], description=description)
    
    return render_template('book.html')

if __name__ == '__main__':
    app.run(debug=True)
