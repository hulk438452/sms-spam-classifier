import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import string

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("This Message Is a Spam")
    else:
        st.header("This Message Is Not a Spam")






def remove_punctuation(text):
    """
    Removes punctuation from a string and returns the cleaned text.
    """
    # Create a string of all punctuation characters
    punctuations = string.punctuation

    # Remove punctuation from the input text
    cleaned_text = "".join(char for char in text if char not in punctuations)

    return cleaned_text

# Remove punctuation from the user's input
if input_sms:
    cleaned_input = remove_punctuation(input_sms)

    # Count the number of words in the cleaned input
    word_count = len(cleaned_input.split())

    st.write("Number of words (excluding punctuation):", word_count)
