import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup

nltk.download('vader_lexicon')

def analyze_sentiment_text(text):
    analyzer = SentimentIntensityAnalyzer()
    nltk_score = analyzer.polarity_scores(text)

    blob = TextBlob(text)
    textblob_score = blob.sentiment.polarity

    return nltk_score, textblob_score 

def analyze_sentiment_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()

    return analyze_sentiment_text(text)

def interpret_sentiment(score):
    if score >= 0.5:
        return "Very Positive"
    elif 0.1 <= score < 0.5:
        return "Positive"
    elif -0.1 <= score < 0.1:
        return "Neutral"
    elif -0.5 <= score < -0.1:
        return "Negative"
    else:
        return "Very Negative"

def main():
    choice = input("Enter 'T' for text or 'U' for URL: ").upper()

    if choice == 'T':
        text = input("Enter the text for sentiment analysis: ")
        nltk_score, textblob_score = analyze_sentiment_text(text)
    elif choice == 'U':
        url = input("Enter the URL for sentiment analysis: ")
        nltk_score, textblob_score = analyze_sentiment_url(url)
    else:
        print("Invalid choice. Please enter 'T' or 'U'.")
        return

    print("\nSentiment Analysis Results:")
    print("NLTK Vader Score:", nltk_score)
    print("NLTK Vader Sentiment:", interpret_sentiment(nltk_score['compound']))
    print("TextBlob Score:", textblob_score)
    print("TextBlob Sentiment:", interpret_sentiment(textblob_score))

if __name__ == "__main__":
    main()
