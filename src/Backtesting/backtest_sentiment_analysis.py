# src/Backtesting/backtest_sentiment_analysis_2.py

import backtrader as bt
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
import os
from crewai import Crew  # Ensure CrewAI is properly installed and accessible
import sys
import numpy as np

# Add the parent directory of 'src' to PYTHONPATH to ensure proper module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import necessary modules from the 'src' package
from src.Indicators.fibonacci import FibonacciRetracement
from src.UI.sentiment_analysis import SentimentCrew  # Correct Import Path for SentimentCrew

# Mock Classes to Replace Missing Imports
# If you have actual implementations for these classes, remove the mock definitions below.
class StockAnalysisAgents:
    def financial_analyst(self):
        return FinancialAnalystAgent()

class FinancialAnalystAgent:
    def analyze_fibonacci_levels(self, fib_levels):
        # Implement your analysis logic here
        # For demonstration, we'll return a simple string
        return f"Analyzed Fibonacci levels: {fib_levels}"

class StockAnalysisTasks:
    def fibonacci_analysis(self, agent, fib_levels):
        analysis_result = agent.analyze_fibonacci_levels(fib_levels)
        return analysis_result

# Load environment variables
load_dotenv()

class SentimentAnalysisCrewAIStrategy(bt.Strategy):
    params = dict(
        stock_or_sector='AAPL',
        printlog=True,
        sentiment_signals=None,  # Dictionary containing buy/sell signals mapped to dates
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None

        # Retrieve sentiment signals passed via parameters
        self.sentiment_signals = self.p.sentiment_signals or {}
        self.buy_signals = self.sentiment_signals.get('buy_signals', {})
        self.sell_signals = self.sentiment_signals.get('sell_signals', {})

    def next(self):
        current_datetime = self.datas[0].datetime.datetime(0)
        current_date = current_datetime.date()

        close_price = self.dataclose[0]

        # Check for buy signal
        if current_date in self.buy_signals and not self.position:
            self.order = self.buy()
            if self.p.printlog:
                self.log(f'BUY CREATE, {close_price:.2f}, Date: {current_date}')

        # Check for sell signal
        elif current_date in self.sell_signals and self.position:
            self.order = self.sell()
            if self.p.printlog:
                self.log(f'SELL CREATE, {close_price:.2f}, Date: {current_date}')

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

class FibonacciStrategy(bt.Strategy):
    params = dict(
        data_df=None,  # Add data_df as a parameter
        printlog=True,
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None

        # Use the data_df passed in via params
        data_df = self.p.data_df

        if data_df is None:
            raise ValueError("FibonacciStrategy requires a non-None data_df parameter.")
        else:
            print(f"FibonacciStrategy received data_df with {len(data_df)} rows.")

        # Initialize Fibonacci Retracement levels using the data_df
        try:
            fibonacci = FibonacciRetracement(data_df)
            self.fib_levels = fibonacci.calculate_levels()
        except Exception as e:
            print(f"Error initializing FibonacciRetracement: {e}")
            raise e

    def next(self):
        close_price = self.dataclose[0]
        current_datetime = self.datas[0].datetime.datetime(0)
        current_date = current_datetime.date()

        if close_price <= self.fib_levels.get('61.8%', 0) and not self.position:
            self.order = self.buy()
            if self.p.printlog:
                self.log(f'BUY CREATE, {close_price:.2f}, Date: {current_date}')
        elif close_price >= self.fib_levels.get('38.2%', 0) and self.position:
            self.order = self.sell()
            if self.p.printlog:
                self.log(f'SELL CREATE, {close_price:.2f}, Date: {current_date}')

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

def run_strategy(strategy_class, strategy_name, data_feed, **kwargs):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    # Create the data feed with adjusted column names
    data = bt.feeds.PandasData(
        dataname=data_feed,
        datetime=None,  # Use index as datetime
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=-1
    )
    cerebro.adddata(data)

    # Prepare strategy arguments
    strategy_kwargs = kwargs.copy()

    # Add the strategy with its parameters
    cerebro.addstrategy(strategy_class, **strategy_kwargs)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.NoTimeFrame, _name='timereturn')

    print(f'\nRunning {strategy_name}...')
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    try:
        results = cerebro.run()
    except Exception as e:
        print(f"Error running {strategy_name}: {e}")
        sys.exit(1)

    strat = results[0]
    final_portfolio_value = cerebro.broker.getvalue()
    print(f'Final Portfolio Value: {final_portfolio_value:.2f}')

    # Extract analyzer results
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    timereturn = strat.analyzers.timereturn.get_analysis()

    cumulative_return = (final_portfolio_value / 100000.0) - 1.0
    start_date = data_feed.index[0]
    end_date = data_feed.index[-1]
    num_years = (end_date - start_date).days / 365.25
    annual_return = (1 + cumulative_return) ** (1 / num_years) - 1 if num_years != 0 else 0.0

    print(f'\n{strategy_name} Performance Metrics:')
    print('----------------------------------------')
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
    print(f"Total Return: {cumulative_return * 100:.2f}%")
    print(f"Annual Return: {annual_return * 100:.2f}%")
    print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")

    # Plot the strategy results (comment out if not needed)
    # cerebro.plot(style='candlestick')

    return {
        'strategy_name': strategy_name,
        'sharpe_ratio': sharpe.get('sharperatio', 'N/A'),
        'total_return': cumulative_return * 100,
        'annual_return': annual_return * 100,
        'max_drawdown': drawdown.max.drawdown,
    }

def extract_signals(sentiment_result, first_date):
    """
    Parses the sentiment analysis result to extract buy/sell signals.

    Args:
        sentiment_result (CrewOutput): The sentiment analysis report.
        first_date (datetime.date): The first date of the backtest period.

    Returns:
        dict: A dictionary with 'buy_signals' and 'sell_signals' mapped to dates.
    """
    buy_signals = {}
    sell_signals = {}

    # Convert CrewOutput to string
    sentiment_str = str(sentiment_result)

    # Check for 'BUY' or 'SELL' in the sentiment report
    if "BUY" in sentiment_str.upper():
        buy_signals[first_date] = True
    if "SELL" in sentiment_str.upper():
        sell_signals[first_date] = True

    return {
        'buy_signals': buy_signals,
        'sell_signals': sell_signals
    }

if __name__ == '__main__':
    # Prompt the user to enter a stock ticker symbol
    stock_or_sector = input("Enter the stock ticker symbol (e.g., AAPL): ").strip().upper()

    if not stock_or_sector:
        print("No ticker symbol entered. Exiting.")
        sys.exit()

    # Fetch historical data
    print(f"\nFetching historical data for {stock_or_sector}...")
    data_df = yf.download(stock_or_sector, start='2020-01-01', end='2024-10-30')

    if data_df.empty:
        print(f"No price data found for {stock_or_sector}")
        sys.exit()

    # Flatten the columns if they are MultiIndex
    if isinstance(data_df.columns, pd.MultiIndex):
        data_df.columns = [' '.join(col).strip() for col in data_df.columns.values]
        # Remove the ticker symbol from the column names
        data_df.columns = [col.split(' ')[0] for col in data_df.columns]

    # If you prefer to use 'Adj Close' as 'Close', rename it
    if 'Adj Close' in data_df.columns:
        data_df['Close'] = data_df['Adj Close']

    # Remove any unnecessary columns
    data_df = data_df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Run Sentiment Analysis to obtain buy/sell signals
    print(f"\nRunning Sentiment Analysis for {stock_or_sector}...")
    sentiment_crew = SentimentCrew(stock_or_sector)
    sentiment_result = sentiment_crew.run()

    # Get the first date of the backtest
    first_date = data_df.index[0].date()

    # Parse the sentiment analysis result to extract buy/sell signals
    sentiment_signals = extract_signals(sentiment_result, first_date)

    if not sentiment_signals['buy_signals'] and not sentiment_signals['sell_signals']:
        print("No actionable buy/sell signals were extracted from the sentiment analysis.")
    else:
        print(f"Extracted {len(sentiment_signals['buy_signals'])} buy signal(s) and {len(sentiment_signals['sell_signals'])} sell signal(s).")

    # Run the Sentiment Analysis CrewAI Strategy
    sentiment_metrics_crewai = run_strategy(
        SentimentAnalysisCrewAIStrategy,
        'Sentiment Analysis CrewAI Strategy',
        data_df.copy(),
        stock_or_sector=stock_or_sector,
        sentiment_signals=sentiment_signals,
        printlog=True  # Ensure printlog is enabled
    )

    # Run the Non-CrewAI Fibonacci Strategy
    fibonacci_metrics_noncrewai = run_strategy(
        FibonacciStrategy,
        'Non-CrewAI Fibonacci Strategy',
        data_df.copy(),
        data_df=data_df.copy(),  # Pass data_df as a parameter
        printlog=True  # Ensure printlog is enabled
    )

    # Compare the performance metrics
    print("\nComparison of Strategies:")
    print("-------------------------")
    metrics = ['strategy_name', 'sharpe_ratio', 'total_return', 'annual_return', 'max_drawdown']
    df_metrics = pd.DataFrame([sentiment_metrics_crewai, fibonacci_metrics_noncrewai], columns=metrics)
    print(df_metrics.to_string(index=False))
