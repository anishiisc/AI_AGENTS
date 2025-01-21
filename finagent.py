import gradio as gr
from openai import OpenAI
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import io
import base64
import yaml
import re
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpdates

class TickerExtractor:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def extract_from_text(self, text):
        """Extract ticker from text using regex patterns"""
        patterns = [
            r'\$([A-Z]{1,5})\b',  # Matches $AAPL
            r'\b([A-Z]{1,5})(?=\s+(?:stock|shares|equity))',  # Matches AAPL stock
            r'\b([A-Z]{1,5})\b'  # Matches standalone uppercase words
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            if matches:
                for potential_ticker in matches:
                    try:
                        ticker = yf.Ticker(potential_ticker)
                        info = ticker.info
                        if info:
                            return potential_ticker
                    except:
                        continue
        return None

    def get_ticker_from_ai(self, text):
        """Use OpenAI to identify potential stock ticker from text"""
        prompt = f"""Extract the stock ticker symbol from this text. If multiple tickers are mentioned, 
        return the most relevant one. If no clear ticker is found, suggest the most likely company's ticker 
        being discussed. Return ONLY the ticker symbol, nothing else.
        
        Text: {text}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial expert. Extract or suggest stock tickers from text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10
            )
            
            potential_ticker = response.choices[0].message.content.strip().upper()
            
            try:
                ticker = yf.Ticker(potential_ticker)
                info = ticker.info
                if info:
                    return potential_ticker
            except:
                return None
                
        except Exception as e:
            print(f"Error in AI ticker extraction: {str(e)}")
            return None

class FinancialAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.ticker_extractor = TickerExtractor(api_key)
        self.context = """You are a sophisticated financial expert assistant. Analyze the provided metrics and generate insights about:
        1. Current market position based on technical indicators
        2. Volume analysis and its implications
        3. Trend strength and potential reversal points
        4. Risk assessment based on volatility metrics
        5. Specific trading signals from indicators
        
        Format your response with clear sections and bullet points when appropriate."""
        self.conversation_history = []

    def get_stock_data(self, ticker, period="1y"):
        """Fetch stock data using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            return df
        except Exception as e:
            return f"Error fetching data: {str(e)}"

    def create_technical_analysis(self, df):
        """Add technical indicators to dataframe"""
        # Price-based indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'], df['Signal'] = self.calculate_macd(df['Close'])
        
        # Volume analysis
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_SMA_50'] = df['Volume'].rolling(window=50).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # Volatility indicators
        df['ATR'] = self.calculate_atr(df)
        df['Bollinger_Upper'], df['Bollinger_Lower'] = self.calculate_bollinger_bands(df['Close'])
        
        # Price patterns
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        return df

    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices, slow=26, fast=12, signal=9):
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    @staticmethod
    def calculate_atr(df, period=14):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(window=period).mean()

    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        
        return upper_band, lower_band

    def analyze_volume_patterns(self, df):
        """Analyze volume patterns and return insights"""
        analysis = {
            'average_volume': df['Volume'].mean(),
            'volume_trend': 'increasing' if df['Volume_SMA_20'].iloc[-1] > df['Volume_SMA_50'].iloc[-1] else 'decreasing',
            'high_volume_days': df[df['Volume_ratio'] > 2].index.tolist(),
            'volume_price_correlation': df['Volume'].corr(df['Close']),
            'unusual_volume_events': []
        }
        
        volume_mean = df['Volume'].mean()
        volume_std = df['Volume'].std()
        unusual_volume = df[df['Volume'] > (volume_mean + 2 * volume_std)]
        
        # Calculate price changes
        df['price_change'] = df['Close'].pct_change()
        
        for date in unusual_volume.index:
            try:
                analysis['unusual_volume_events'].append({
                    'date': date,
                    'volume': unusual_volume.loc[date, 'Volume'],
                    'price_change': df.loc[date, 'price_change'] if not pd.isna(df.loc[date, 'price_change']) else 0
                })
            except Exception as e:
                continue
        
        return analysis

    def analyze_technical_signals(self, df):
        """Analyze technical indicators and generate trading signals"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = {
            'rsi_signal': 'oversold' if latest['RSI'] < 30 else 'overbought' if latest['RSI'] > 70 else 'neutral',
            'macd_signal': 'bullish' if latest['MACD'] > latest['Signal'] and prev['MACD'] <= prev['Signal'] 
                          else 'bearish' if latest['MACD'] < latest['Signal'] and prev['MACD'] >= prev['Signal'] 
                          else 'neutral',
            'ma_signal': 'bullish' if latest['SMA_20'] > latest['SMA_50'] else 'bearish',
            'volatility_state': 'high' if latest['Volatility'] > df['Volatility'].mean() * 1.5 else 'normal',
            'volume_signal': 'high' if latest['Volume'] > latest['Volume_SMA_20'] * 1.5 else 'normal'
        }
        
        signals['price_to_sma20'] = (latest['Close'] / latest['SMA_20'] - 1) * 100
        signals['bollinger_position'] = (latest['Close'] - latest['Bollinger_Lower']) / (latest['Bollinger_Upper'] - latest['Bollinger_Lower'])
        
        return signals

    def generate_market_insights(self, df, ticker):
        """Generate comprehensive market insights"""
        signals = self.analyze_technical_signals(df)
        volume_analysis = self.analyze_volume_patterns(df)
        
        returns = df['Daily_Return'].iloc[-20:].mean() * 100
        volatility = df['Volatility'].iloc[-1] * 100
        
        insights = f"""Analysis for {ticker}:

Technical Indicators:
- RSI ({df['RSI'].iloc[-1]:.2f}) indicates {signals['rsi_signal']} conditions
- MACD shows {signals['macd_signal']} momentum
- Price is {signals['price_to_sma20']:.2f}% relative to 20-day SMA
- Current position in Bollinger Bands: {signals['bollinger_position']:.2f} (0=lower band, 1=upper band)

Volume Analysis:
- Average Volume: {volume_analysis['average_volume']:,.0f}
- Volume Trend: {volume_analysis['volume_trend']}
- Volume-Price Correlation: {volume_analysis['volume_price_correlation']:.2f}
- Current Volume Signal: {signals['volume_signal']}

Risk Metrics:
- 20-day Rolling Volatility: {volatility:.2f}%
- 20-day Average Return: {returns:.2f}%
- ATR: {df['ATR'].iloc[-1]:.2f}
- Volatility State: {signals['volatility_state']}"""

        return insights

    def get_response(self, message, include_analysis=True):
        """Get response from OpenAI with enhanced technical analysis"""
        if include_analysis and hasattr(self, 'df'):
            insights = self.generate_market_insights(self.df, self.selected_ticker)
            message += f"\n\n{insights}"
            
        self.conversation_history.append({"role": "user", "content": message})
        messages = [
            {"role": "system", "content": self.context},
            *self.conversation_history
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=2000
        )
        
        assistant_message = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message

class FinancialPlotter:
    @staticmethod
    def create_stock_plot(df, ticker):
        """Create a stock price plot with technical indicators"""
        # Ensure datetime format and remove timezones
        df.index = pd.to_datetime(df.index)
        if df.index.tz:
            df.index = df.index.tz_localize(None)
    
        # Add date numbers for plotting
        df['Date_num'] = mpdates.date2num(df.index)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
    
        # Plot candlesticks
        ohlc_data = df[['Date_num', 'Open', 'High', 'Low', 'Close']].values
        candlestick_ohlc(ax, ohlc_data, width=0.6, colorup='green', colordown='red', alpha=0.8)
    
        # Plot moving averages
        ax.plot(df['Date_num'], df['SMA_20'], color='orange', label='SMA 20', linewidth=1.5)
        ax.plot(df['Date_num'], df['SMA_50'], color='blue', label='SMA 50', linewidth=1.5)
    
        # Customize appearance
        ax.set_title(f'{ticker} Stock Price', fontsize=14, pad=20)
        ax.set_ylabel('Price', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
    
        # Format x-axis
        ax.xaxis.set_major_locator(mpdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mpdates.ConciseDateFormatter(mpdates.AutoDateLocator()))
        plt.xticks(rotation=45)
    
        ax.legend(loc='best')
        plt.tight_layout()
    
        return fig
    
    @staticmethod
    def create_technical_indicators_plot(df):
        """Create subplot with RSI and MACD"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1])
        
        # Plot RSI
        ax1.plot(df.index, df['RSI'], color='purple', linewidth=1.5)
        ax1.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax1.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax1.set_title('RSI (14)', fontsize=12)
        ax1.set_ylabel('RSI Value')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.fill_between(df.index, df['RSI'], 30, where=(df['RSI'] <= 30), color='g', alpha=0.3)
        ax1.fill_between(df.index, df['RSI'], 70, where=(df['RSI'] >= 70), color='r', alpha=0.3)
        
        # Plot MACD
        ax2.plot(df.index, df['MACD'], color='blue', label='MACD', linewidth=1.5)
        ax2.plot(df.index, df['Signal'], color='orange', label='Signal', linewidth=1.5)
        ax2.bar(df.index, df['MACD'] - df['Signal'], 
                color=['g' if val >= 0 else 'r' for val in df['MACD'] - df['Signal']],
                alpha=0.3)
        ax2.set_title('MACD', fontsize=12)
        ax2.set_ylabel('Value')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper left')
        
        # Format x-axis dates
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_volume_analysis_plot(df):
        """Create volume analysis plot"""
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        if df.index.tz:
            df.index = df.index.tz_localize(None)
            
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Plot volume bars with safe color assignment
        volume_colors = []
        for close, open_price in zip(df['Close'], df['Open']):
            try:
                volume_colors.append('g' if close >= open_price else 'r')
            except:
                volume_colors.append('b')  # default color if comparison fails
                
        ax.bar(df.index, df['Volume'], color=volume_colors, alpha=0.7)
        
        # Add volume moving average
        volume_ma = df['Volume'].rolling(window=20).mean()
        ax.plot(df.index, volume_ma, color='blue', label='Volume MA (20)', linewidth=1.5)
        
        # Customize appearance
        ax.set_title('Volume Analysis', fontsize=12)
        ax.set_ylabel('Volume')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper left')
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig

def clear_outputs():
    """Function to clear all outputs"""
    return [], None, None, None  # Returns empty list for chatbot and None for each plot

def process_request(message, ticker_input, period_input, history):
    """Process user request with enhanced analysis and visualization"""
    
    # Initialize history if None
    if history is None:
        history = []
    
    history.append([message, None])
    
    try:
        # Load API key
        with open('auth.yaml', 'r') as f:
            config = yaml.safe_load(f)
        apikey = config['openai']['access_key']
    except Exception as e:
        history[-1][1] = f"Error loading API key: {str(e)}"
        return history, None, None, None  # Return all expected outputs
    
    # Initialize OpenAI client
    client = OpenAI(api_key=apikey)
    
    # Step 1: Check if ticker is explicitly provided in input field
    selected_ticker = None
    if ticker_input and ticker_input.strip():
        try:
            ticker = yf.Ticker(ticker_input.strip().upper())
            info = ticker.info
            if info:
                selected_ticker = ticker_input.strip().upper()
                print(f"Using provided ticker: {selected_ticker}")
            else:
                history[-1][1] = f"Error: Invalid ticker symbol {ticker_input}"
                return history, None, None, None
        except Exception as e:
            history[-1][1] = f"Error validating ticker {ticker_input}: {str(e)}"
            return history, None, None, None
    
    # Step 2: If no ticker provided, look for explicit ticker mentions in message
    if not selected_ticker:
        explicit_patterns = [
            r'\$([A-Z]{1,5})\b',  # $AAPL
            r'ticker[:\s]+([A-Z]{1,5})\b',  # ticker: AAPL
            r'symbol[:\s]+([A-Z]{1,5})\b',  # symbol: AAPL
            r'\b([A-Z]{1,5})(?=\s+(?:stock|shares|price|ticker))\b'  # AAPL stock
        ]
        
        for pattern in explicit_patterns:
            matches = re.findall(pattern, message.upper())
            if matches:
                for match in matches:
                    ticker_symbol = match[0] if isinstance(match, tuple) else match
                    try:
                        ticker = yf.Ticker(ticker_symbol)
                        info = ticker.info
                        if info:
                            selected_ticker = ticker_symbol
                            print(f"Found explicit ticker mention: {selected_ticker}")
                            break
                    except:
                        continue
            if selected_ticker:
                break
    
    # If still no ticker found, try AI extraction
    if not selected_ticker:
        try:
            extractor = TickerExtractor(apikey)
            selected_ticker = extractor.get_ticker_from_ai(message)
        except Exception as e:
            print(f"Error in AI ticker extraction: {str(e)}")
    
    # If no ticker found after all attempts, return error message
    if not selected_ticker:
        response = "I couldn't identify a specific stock ticker. Please either:\n" \
                  "1. Use the ticker input field (e.g., AAPL), or\n" \
                  "2. Mention the company name or ticker in your message (e.g., 'Analyze Tesla' or 'Check $TSLA')"
        history[-1][1] = response
        return history, None, None, None
    
    # Process the request with the identified ticker
    try:
        agent = FinancialAgent(api_key=apikey)
        df = agent.get_stock_data(selected_ticker, period_input)
        
        if isinstance(df, pd.DataFrame):
            df = agent.create_technical_analysis(df)
            agent.df = df
            agent.selected_ticker = selected_ticker
            
            # Create plots
            stock_plot = FinancialPlotter.create_stock_plot(df, selected_ticker)
            technical_plot = FinancialPlotter.create_technical_indicators_plot(df)
            volume_analysis_plot = FinancialPlotter.create_volume_analysis_plot(df)
            
            # Generate response incorporating the analysis
            response = agent.get_response(
                f"{message}\nAnalysis for {selected_ticker} stock based on your query:", 
                include_analysis=True
            )
            
            history[-1][1] = f"Using ticker: {selected_ticker}\n\n{response}"
            return history, stock_plot, technical_plot, volume_analysis_plot
            
        else:
            error_msg = f"Error: Unable to fetch data for {selected_ticker}"
            history[-1][1] = error_msg
            return history, None, None, None
            
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        history[-1][1] = error_msg
        return history, None, None, None

# Main Gradio interface setup
demo = gr.Blocks(theme=gr.themes.Base())
with demo:
    gr.Markdown("# Advanced Financial Analysis Assistant")
    
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=600, value=[])
            with gr.Row():
                ticker_input = gr.Textbox(label="Stock Ticker (optional, e.g., AAPL)", scale=1)
                period_input = gr.Dropdown(
                    choices=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                    label="Time Period",
                    value="1y",
                    scale=1
                )
            message = gr.Textbox(label="Type your message here...", scale=2)
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear All", variant="secondary")
        
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("Stock Price"):
                    stock_plot = gr.Plot(label="Stock Price Analysis")
                with gr.Tab("Technical Indicators"):
                    technical_plot = gr.Plot(label="Technical Indicators")
                with gr.Tab("Volume Analysis"):
                    volume_analysis_plot = gr.Plot(label="Volume Analysis")

    submit.click(
        fn=process_request,
        inputs=[message, ticker_input, period_input, chatbot],
        outputs=[chatbot, stock_plot, technical_plot, volume_analysis_plot]
    )
    
    clear.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[chatbot, stock_plot, technical_plot, volume_analysis_plot]
    )

if __name__ == "__main__":
    # Load API key
    OPENAI_CONFIG_FILE = 'auth.yaml'
    
    with open(OPENAI_CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    apikey = config['openai']['access_key']
    
    # Initialize OpenAI client
    client = OpenAI(api_key=apikey)
    
    # Launch the Gradio interface
    demo.launch()