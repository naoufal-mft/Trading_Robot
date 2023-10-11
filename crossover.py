import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG
import pandas_ta as ta

#df = pd.read_csv("data.csv")
#df = pd.read_csv("daily.csv")
df = pd.read_csv("test4.csv")
#df = pd.read_csv("weekly_IBM.csv")
#df = pd.read_csv("test3.csv")
print(df)
#df = df.rename(columns={'date': 'Timestamp', '1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'})
df.head()
df = df.rename(columns={'timestamp': 'Timestamp', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
df=df[df['Volume']!=0]
df.isna().sum()
df.reset_index(drop=True, inplace=True)
df.tail()
class Crossover(Strategy):
  def init(self):
    price = self.data.Close
    self.ma1 = self.I(SMA, price, 50)
    self.ma2 = self.I(SMA, price, 200)
    self.rsi = ta.rsi(price, length=14)
    #self.rsi = rsi.fillna(0).values
  
  def next(self):
   # rsi = ta.rsi(self.data.Close, length=14)  # Calculate RSI with a period of 14
        
    if crossover(self.ma1, self.ma2) and (self.rsi is not None and self.rsi > 30):  # Check if rsi is not None or NaN
        self.buy()
    elif crossover(self.ma2, self.ma1) or (self.rsi is not None and self.rsi < 70):  # Check if rsi is not None or NaN
        self.sell()

backtest = Backtest(df, Crossover, commission=.002, exclusive_orders=True)

stats = backtest.run()
print(stats)
#backtest.plot()'''