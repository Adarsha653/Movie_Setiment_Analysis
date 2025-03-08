from flask import Flask, request, render_template
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pickle as pk

app = Flask(__name__)

model = pk.load(open('model.pkl', 'rb'))
tfidf = pk.load(open('tfidf.pkl', 'rb'))

# Define function to stem and lemmatize the sentiment texts
def preprocess_text(text):
    
    # Define lemmatizer object
    lemmatizer = WordNetLemmatizer() 
    
    text = text.lower() # Converts all to lowercase
    text = re.sub(r'<br />', r'', text) # Removes HTML tags '<br /'
    text = re.sub(r'\d+', r'', text) # Removes numeric values 
    text = re.sub(r'[^\w\s]', r'', text) # Removes punctuations (keeping spaces)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces    
    
    words = word_tokenize(text) #Splits text into words

    # Define a set of english stopwords
    stop_words = set(stopwords.words('english')) 
    
    # Remove stop words and apply lemmatization
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    cleaned_reviews = ' '.join(filtered_words) # Join words back into a single string

    return cleaned_reviews  # Return cleaned tokens

def predict_sentiment(review):
    processed_review = preprocess_text(review)
    # Apply TF-IDF vectorization
    transformed_review = tfidf.transform([processed_review])
    sentiment = model.predict(transformed_review)
    if sentiment == 1:
        return 'Positive'
    else:
        return 'Negative'
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        sentiment = predict_sentiment(review)
        return render_template('index.html', prediction_text='Sentiment: {}'.format(sentiment))
    
if __name__ == '__main__':
    app.run(debug=True)