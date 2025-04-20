import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime

# Streamlit App Title
st.title("Bitcoin Price Prediction with Sentiment Analysis")

# Function to get yesterday's date
def get_yesterday():
    yesterday = datetime.datetime.now() - datetime.timedelta(1)
    return yesterday.date()

# Section 1: Fetch Bitcoin Data
def fetch_bitcoin_data():
    st.subheader("Bitcoin Historical Data")
    yesterday = get_yesterday()
    data = yf.download('BTC-USD', start='2021-01-01', end=yesterday)
    data['Price Change'] = data['Close'].pct_change() * 100
    st.write(data.tail())
    return data

# Section 2: Perform Sentiment Analysis
def fetch_sentiment_data():
    st.subheader("Bitcoin Sentiment Analysis")
    analyzer = SentimentIntensityAnalyzer()
    sample_comments = [
        "Bitcoin is the future of finance!",
        "I'm not sure about Bitcoin's stability.",
        "Bitcoin is a scam!",
        "Crypto is an amazing investment opportunity!",
        "I lost so much money in Bitcoin recently."
    ]
    
    sentiment_scores = [analyzer.polarity_scores(comment)['compound'] for comment in sample_comments]
    sentiment_labels = [1 if score > 0.05 else -1 if score < -0.05 else 0 for score in sentiment_scores]
    sentiment_df = pd.DataFrame({'Comment': sample_comments, 'Sentiment': sentiment_labels})
    st.write(sentiment_df.head())

    # Sentiment distribution bar chart
    st.subheader("Sentiment Distribution")
    sentiment_count = sentiment_df['Sentiment'].value_counts().sort_index()
    st.bar_chart(sentiment_count)
    st.write("X-axis: Sentiment Categories (-1(negative), 0(neutral), 1(positive))")
    st.write("Y-axis: Frequency")
    return sentiment_df

# Section 3: Feature Engineering
def preprocess_data(price_data, sentiment_data):
    sentiment_score = sentiment_data['Sentiment'].mean()
    price_data['Sentiment'] = sentiment_score
    price_data['Target'] = (price_data['Close'].shift(-1) > price_data['Close']).astype(int)
    price_data.dropna(inplace=True)
    return price_data

# Section 4: Train Random Forest Classifier
def train_model(data):
    X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment']]
    y = data['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, scaler

# Section 5: Streamlit Interface
def main():
    st.sidebar.header("Navigation")
    options = ["Bitcoin Data", "Sentiment Analysis", "Prediction"]
    choice = st.sidebar.selectbox("Choose a section:", options)

    if choice == "Bitcoin Data":
        data = fetch_bitcoin_data()
        st.line_chart(data['Close'])
        st.write("X-axis: Date")
        st.write("Y-axis: Closing Price (USD)")

    elif choice == "Sentiment Analysis":
        fetch_sentiment_data()
    
    elif choice == "Prediction":
        st.subheader("Bitcoin Price Prediction")
        price_data = fetch_bitcoin_data()
        sentiment_data = fetch_sentiment_data()
        processed_data = preprocess_data(price_data, sentiment_data)

        model, scaler = train_model(processed_data)

        # Display prediction for the next 7 days
        last_data = processed_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment']].tail(1)
        predictions = []
        predicted_prices = [last_data['Close'].values[0]]
        dates = []

        # Predict the next 7 days
        for i in range(7):
            scaled_data = scaler.transform(last_data)
            prediction = model.predict(scaled_data)
            predictions.append(prediction[0])

            # Simulate price change based on prediction
            predicted_price = predicted_prices[-1] * (1 + (predictions[-1] * 0.01))
            predicted_prices.append(predicted_price)

            last_data['Sentiment'] = sentiment_data['Sentiment'].mean()
            last_data['Close'] = predicted_price
            last_data = last_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment']]

            dates.append(get_yesterday() + datetime.timedelta(days=i+1))

        # Prepare DataFrame for the predicted data
        predicted_data = pd.DataFrame({
            'Date': dates,
            'Predicted Close': predicted_prices[1:]
        })

        # Combine actual and predicted data
        actual_data = price_data[['Open', 'High', 'Low', 'Close']].tail(30)
        actual_data['Date'] = actual_data.index

        # Line Chart for Predicted Prices
        st.subheader("Predicted Bitcoin Prices (Line Chart)")
        plt.figure(figsize=(10, 6))
        plt.plot(actual_data['Date'], actual_data['Close'], label="Actual", color="orange")
        plt.plot(predicted_data['Date'], predicted_data['Predicted Close'], label="Predicted", color="blue", linestyle="--")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.title("Actual vs Predicted Bitcoin Prices")
        plt.legend()
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Candlestick Chart
        st.subheader("Bitcoin Price: Candlestick Chart with Predictions")
        fig, ax = plt.subplots(figsize=(10, 6))
        actual_data['Date'] = actual_data['Date'].apply(mdates.date2num)
        predicted_data['Date'] = predicted_data['Date'].apply(mdates.date2num)
        ohlc = actual_data[['Date', 'Open', 'High', 'Low', 'Close']].values
        candlestick_ohlc(ax, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.8)
        ax.plot(predicted_data['Date'], predicted_data['Predicted Close'], label='Predicted', color='blue', linestyle='--', marker='o')
        ax.xaxis_date()
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.set_title('Bitcoin Price: Candlestick Chart with Predictions')
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
