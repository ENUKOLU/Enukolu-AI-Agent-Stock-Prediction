import backtrader as bt
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
import os
from crewai import Crew
from src.Agents.Analysis.stock_analysis_agents import StockAnalysisAgents
from src.Agents.Analysis.stock_analysis_tasks import StockAnalysisTasks
import sys
import numpy as np

# Load environment variables
load_dotenv()

class SentimentAnalysisCrewAIStrategy(bt.Strategy):
    params = dict(
        stock_or_sector='AAPL',
        data_df=None,  # Add data_df as a parameter
        printlog=True,
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None

        # Use the data_df passed in via params
        data_df = self.params.data_df

        # Initialize agents and tasks
        agents = StockAnalysisAgents()
        tasks = StockAnalysisTasks()
        self.sentiment_analyst = agents.sentiment_analyst()

        # Simulate sentiment scores over the data
        # For the sake of example, we can simulate sentiment scores as random numbers or based on price changes

        # Let's create a Series to hold the sentiment scores with the same index as data_df
        self.sentiment_scores = pd.Series(index=data_df.index, dtype=float)

        # For simplicity, let's assume that if the close price increased compared to previous day, sentiment is positive
        # If close price decreased, sentiment is negative
        close_prices = data_df['Close']
        self.sentiment_scores.iloc[0] = 0  # Neutral sentiment on the first day
        for i in range(1, len(close_prices)):
            if close_prices.iloc[i] > close_prices.iloc[i -1]:
                self.sentiment_scores.iloc[i] = 1  # Positive sentiment
            elif close_prices.iloc[i] < close_prices.iloc[i -1]:
                self.sentiment_scores.iloc[i] = -1  # Negative sentiment
            else:
                self.sentiment_scores.iloc[i] = 0  # Neutral sentiment

        # Alternatively, you could simulate sentiment scores with random numbers
        # self.sentiment_scores = pd.Series(np.random.uniform(-1, 1, len(data_df)), index=data_df.index)

    def next(self):
        # Get the current datetime as a pandas Timestamp
        current_datetime = self.datas[0].datetime.datetime(0)
        current_date = pd.Timestamp(current_datetime).normalize()

        close_price = self.dataclose[0]

        # Get the sentiment score for the current date
        sentiment_score = self.sentiment_scores.loc[current_date]

        # Trading logic based on sentiment score
        if sentiment_score > 0 and not self.position:
            self.order = self.buy()
            if self.params.printlog:
                self.log(f'BUY CREATE, {close_price:.2f}, Sentiment Score: {sentiment_score}')
        elif sentiment_score < 0 and self.position:
            self.order = self.sell()
            if self.params.printlog:
                self.log(f'SELL CREATE, {close_price:.2f}, Sentiment Score: {sentiment_score}')

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

# New Buy and Hold Strategy
class BuyAndHoldStrategy(bt.Strategy):
    params = dict(
        printlog=True,
    )

    def __init__(self):
        self.order = None
        self.dataclose = self.datas[0].close

    def next(self):
        # Buy on the first day
        if len(self) == 1:
            size = int(self.broker.getcash() / self.dataclose[0])
            self.order = self.buy(size=size)
            if self.params.printlog:
                self.log(f'BUY CREATE, {self.dataclose[0]:.2f}, Size: {size}')

    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
        self.order = None

def run_strategy(strategy_class, strategy_name, data_df, **kwargs):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    # Create the data feed with adjusted column names
    data = bt.feeds.PandasData(
        dataname=data_df,
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
    if hasattr(strategy_class.params, 'data_df'):
        # Strategy expects 'data_df', pass it
        strategy_kwargs['data_df'] = data_df
    else:
        # Strategy does not expect 'data_df', remove it if present
        strategy_kwargs.pop('data_df', None)

    cerebro.addstrategy(strategy_class, **strategy_kwargs)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.NoTimeFrame, _name='timereturn')

    print(f'\nRunning {strategy_name}...')
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    results = cerebro.run()
    strat = results[0]
    final_portfolio_value = cerebro.broker.getvalue()
    print(f'Final Portfolio Value: {final_portfolio_value:.2f}')

    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    timereturn = strat.analyzers.timereturn.get_analysis()

    cumulative_return = (final_portfolio_value / 100000.0) - 1.0
    start_date = data_df.index[0]
    end_date = data_df.index[-1]
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

if __name__ == '__main__':
    stock_or_sector = 'AAPL'
    data_df = yf.download(stock_or_sector, start='2020-01-01', end='2023-10-30')

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

    # Run the Sentiment Analysis CrewAI Strategy
    sentiment_metrics_crewai = run_strategy(
        SentimentAnalysisCrewAIStrategy,
        'Sentiment Analysis CrewAI Strategy',
        data_df.copy(),
        stock_or_sector=stock_or_sector
    )

    # Run the Non-CrewAI Buy and Hold Strategy
    buy_and_hold_metrics = run_strategy(
        BuyAndHoldStrategy,
        'Non-CrewAI Buy and Hold Strategy',
        data_df.copy()
    )

    # Compare the performance metrics
    print("\nComparison of Strategies:")
    print("-------------------------")
    metrics = ['strategy_name', 'sharpe_ratio', 'total_return', 'annual_return', 'max_drawdown']
    df_metrics = pd.DataFrame([sentiment_metrics_crewai, buy_and_hold_metrics], columns=metrics)
    print(df_metrics.to_string(index=False))
