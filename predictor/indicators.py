import pandas as pd

def moving_average(data, days):
    df = pd.DataFrame(data)
    df['MA'] = df['Close'].rolling(days).mean()
    return df

def bollinger_bands(data, window=20, num_std=2):
    df = pd.DataFrame(data)

    df['Rolling Mean'] = df['Close'].rolling(window).mean()
    df['Standard Deviation'] = df['Close'].rolling(window).std()

    df['Upper Band'] = df['Rolling Mean'] + num_std * df['Standard Deviation']
    df['Lower Band'] = df['Rolling Mean'] - num_std * df['Standard Deviation']

    df['Bollinger Band Coefficient'] = (df['Close'] - df['Lower Band']) / (df['Upper Band'] - df['Lower Band'])

    df.drop(['Rolling Mean', 'Standard Deviation', 'Upper Band', 'Lower Band'], axis=1, inplace=True)

    df = df.dropna()

    return df

def stochastic_oscillator(data, k_period=14, d_period=3):
    df = pd.DataFrame(data)

    df['Lowest Low'] = df['Low'].rolling(window=k_period).min()
    df['Highest High'] = df['High'].rolling(window=k_period).max()

    df['%K'] = (df['Close'] - df['Lowest Low']) / (df['Highest High'] - df['Lowest Low']) * 100
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    
    df.drop(['Lowest Low', 'Highest High'], axis=1, inplace=True)
    
    return df

def fibonacci_retracement(df):
    high = df['High']
    low = df['Low']
    diff = high - low

    df['Fibonacci_0.0'] = high
    df['Fibonacci_0.236'] = high - (diff * 0.236)
    df['Fibonacci_0.382'] = high - (diff * 0.382)
    df['Fibonacci_0.5'] = high - (diff * 0.5)
    df['Fibonacci_0.618'] = high - (diff * 0.618)
    df['Fibonacci_0.764'] = high - (diff * 0.764)
    df['Fibonacci_1.0'] = low

    return df

def difference_indicator(data):
    df = pd.DataFrame(data)
    df['Difference Close - Open'] = df['Close'] - df['Open']

    df.drop(['Open'], axis=1, inplace=True)

    return df

def percentage_difference_indicator(data):
    df = pd.DataFrame(data)
    df['Percentage Difference Close - Open'] = (df['Close'] - df['Open']) / df['Open'] * 100

    df.drop(['Open'], axis=1, inplace=True)

    return df