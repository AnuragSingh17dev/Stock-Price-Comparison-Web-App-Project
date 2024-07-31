import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from stocknews import StockNews
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go


hide_st_style = ''' <style> #MainMenu {visibility: hidden;} footer{visibility: hidden;} </style>'''
st.markdown(hide_st_style, unsafe_allow_html=True)
pd.set_option('display.max_colwidth', 500)

st.title('Stock Price Dashboard')
st.write('---')
st.write('''
**Credits**
- App built by __Group 17__
- Devesh Talreja, Mohsin Panjwani , Anurag Singh
''')
st.write('---')

# Sidebar options
stck1 = st.sidebar.text_input('Stock 1')
stck2 = st.sidebar.text_input('Stock 2')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')
forecast_period = st.sidebar.slider('Forecast period (days)', min_value=1, max_value=365, value=30)

def graph():
    # Download data for stock 1 and stock 2
    data1 = yf.download(stck1, start=start_date, end=end_date)
    data2 = yf.download(stck2, start=start_date, end=end_date)

    # Plot line chart for stock 1 and stock 2
    fig = px.line(data1, x=data1.index, y=data1['Adj Close'], title=f"{stck1} vs {stck2}")
    fig.add_scatter(x=data2.index, y=data2['Adj Close'], mode='lines', name=stck2,line=dict(color='red'))

    st.plotly_chart(fig)

    st.write('---')
    st.markdown('## Pricing data')


    col1, col2 = st.columns([5,5])

    with col1:
        st.write('### ' + stck1)
        data_temp1 = data1
        data_temp1['% Change'] = data1['Adj Close'] / data1['Adj Close'].shift(1) - 1
        data_temp1.dropna(inplace=True)
        st.write(data_temp1)

        annual_return1 = data_temp1['% Change'].mean() * 252 * 100  # excluding weekends
        st.write('According to the trend your annual return is', annual_return1, '%')
        stdev1 = np.std(data_temp1['% Change']) * np.sqrt(252)
        st.write('The Standard Deviation is', stdev1 * 100, '%')
        st.write('Risk Adjusted Return = ', annual_return1 / (stdev1 * 100))

    with col2:
        st.write('### ' + stck2)
        data_temp2 = data2
        data_temp2['% Change'] = data2['Adj Close'] / data2['Adj Close'].shift(1) - 1
        data_temp2.dropna(inplace=True)
        st.write(data_temp2)

        annual_return2 = data_temp2['% Change'].mean() * 252 * 100  # excluding weekends
        st.write('According to the trend your annual return is', annual_return2, '%')
        stdev2 = np.std(data_temp2['% Change']) * np.sqrt(252)
        st.write('The Standard Deviation is', stdev2 * 100, '%')
        st.write('Risk Adjusted Return = ', annual_return2 / (stdev2 * 100))

    st.write('---')
    st.markdown('## News Comparison')
    sn1 = StockNews(stck1, save_news=False)
    df_news1 = sn1.read_rss()
    sn2 = StockNews(stck2, save_news=False)
    df_news2 = sn2.read_rss()

    for i in range(10):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f'{stck1} News {i + 1}')
            st.write(df_news1['published'][i])
            st.write(df_news1['title'][i])
            st.write(df_news1['summary'][i])
            sentiment = df_news1['sentiment_title'][i]
            st.write(f'Title Sentiment {sentiment}')
            news_senti = df_news1['sentiment_summary'][i]
            st.write(f'News Sentiment {news_senti}')

        with col2:
            st.subheader(f'{stck2} News {i + 1}')
            st.write(df_news2['published'][i])
            st.write(df_news2['title'][i])
            st.write(df_news2['summary'][i])
            sentiment = df_news2['sentiment_title'][i]
            st.write(f'Title Sentiment {sentiment}')
            news_senti = df_news2['sentiment_summary'][i]
            st.write(f'News Sentiment {news_senti}')

    st.write('---')
    st.markdown('## Future Comparison')
    data = pd.concat([data1['Adj Close'], data2['Adj Close']], axis=1)
    data.columns = [stck1, stck2]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data.dropna(), data.dropna(), test_size=0.2, shuffle=False)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions for the forecast period
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=forecast_period+1, freq='D')[1:]
    forecast = pd.DataFrame(index=forecast_dates, columns=data.columns)
    forecast.loc[forecast.index[0], :] = data.iloc[-1, :].values
    for i in range(1, len(forecast)):
        forecast.iloc[i, :] = model.predict(forecast.iloc[i-1, :].values.reshape(1, -1))
    # Plot the actual and forecasted values on a line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data[stck1], mode='lines', name=stck1))
    fig.add_trace(go.Scatter(x=data.index, y=data[stck2], mode='lines', name=stck2))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast[stck1], mode='lines', name=f'{stck1} forecast'))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast[stck2], mode='lines', name=f'{stck2} forecast'))
    fig.update_layout(title=f'{stck1} vs {stck2} Stock Prices', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)
        


    

if st.sidebar.button("Compare"):
    if stck1 and stck2:
        graph()
