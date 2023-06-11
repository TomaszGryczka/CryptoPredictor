import pandas as pd

def moving_average(data, days):
    df = pd.DataFrame(data)
    df['MA'] = df['Close'].rolling(days).mean()
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