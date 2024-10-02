# Six Human Emotions Detection App

This project is a web-based application that detects six human emotions—Joy, Fear, Love, Anger, Sadness, and Surprise—based on text input. The app leverages natural language processing (NLP) techniques to analyze user input and predict the corresponding emotion using a Logistic Regression model.

## Features
- **Emotion Prediction**: Detects six emotions from text: Joy, Fear, Love, Anger, Sadness, and Surprise.
- **NLP Pipeline**: Includes text cleaning, stemming, and stopword removal using the `nltk` library.
- **Logistic Regression**: Pre-trained model to classify emotions based on user input.
- **TF-IDF Vectorization**: Converts input text into feature vectors for model prediction.
- **Streamlit Web App**: Provides a user-friendly interface to input text and get predictions.

## How it Works
1. The app processes the input text, cleans it, and removes any unwanted characters.
2. The cleaned text is transformed into a vector using a TF-IDF vectorizer.
3. The Logistic Regression model predicts the emotion based on the vectorized input.
4. The predicted emotion is displayed in the app.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/emotions-detection-app.git
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
3. Download necessary NLTK data:
   ```bash
   import nltk
   nltk.download('stopwords')
4. Run the Streamlit app:
   ```bash
   streamlit run app.py


## USAGE

1. Open the app in your browser.  
2. Enter text in the input field.  
3. Click the "Predict" button to see the predicted emotion.


## FILES

1. app.py: The main Streamlit app for emotion prediction.  
2. logistic_regresion.pkl: Pre-trained Logistic Regression model.  
3. tfidf_vectorizer.pkl: TF-IDF vectorizer used to convert text to feature vectors.  
4. label_encoder.pkl: Label encoder to decode predicted labels back into emotion names.


## Requirements
Python 3.x  
streamlit  
nltk  
scikit-learn  
numpy  
re  
pickle  

## EXAMPLE



   
   
   
   
   
   
