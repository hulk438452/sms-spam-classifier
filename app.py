import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import string
from collections import Counter
#from wordcloud import WordCloud
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

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


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

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
    st.write("Original Message: {}".format(input_sms))
    st.write("Preprocessed Message: {}".format(transformed_sms))
    if result == 1:
        st.header("This Message Is a Spam")
    else:
        st.header("This Message Is Not a Spam")

    st.write("Number of characters:", len(input_sms))
    st.write("Number of words (excluding punctuation):", len(transformed_sms.split()))

    # Word Cloud
   # wc = WordCloud(background_color="white", width=800, height=400).generate(transformed_sms)
#    st.write("Word Cloud")
    #st.image(wc.to_array())

    # Bar Chart
    words = transformed_sms.split()
    word_counts = Counter(words)
    top_words = word_counts.most_common(10)
    plt.bar([w[0] for w in top_words], [w[1] for w in top_words])
    plt.xticks(rotation=45)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Word Frequency')
    st.pyplot()


if st.button('Reset'):
    input_sms = ''
