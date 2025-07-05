# Book Genre Prediction using Description

This project predicts the genre of a book based on its description using a machine learning model trained on a dataset of book summaries and genres.

# Features:

- Predicts the book genre using the description of the story.
- Trained with a machine learning model (`genre_prediction.pkl`).
- Simple web interface using Flask.
- Clean UI with `home.html` and `book.html`.

# Technologies Used:
Python – Programming language used to build the backend logic and machine learning model.

Flask – Lightweight web framework used to create the web application.

Pandas – For handling and processing the dataset.

Scikit-learn – For building and training the machine learning model.

HTML & CSS – For designing the frontend interface (`home.html`, `book.html`, `styles.css`).

Pickle – To save and load the trained machine learning model (`genre_prediction.pkl`).

# Project Structure:
```
BookGenrePrediction/
│
├── static/
│ └── styles.css 
│
├── templates/
│ ├── home.html 
│ └── book.html 
│
├── app.py 
├── Book_Dataset.csv 
├── genre_prediction.pkl 
├── training_model.py 
```
