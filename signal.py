# APO(Absolute Price Oscillator) : 절대 가격 이격도 전략
  # EMA(Exponential Moving Average) 중 Fast EMA와 Slow EMA간 이격도를 바탕으로 시그널 생성
  # Fast EMA : 최근 가격에 더 반응, Slow EMA : 과거 추세에 더 반응하는 Slow EMA 와의 이격정도를 바탕으로 시그널 생성

# MACD(Moving Average Convergence Divergence) : 이동평균 수렴 발산 전략
  # MACD = Fast EMA - Slow EMA 생성 후, 이에 대한 Exponential Moving Average 값을 이용하여 시그널 생성

# RSI(Relative Strength Indicator) : 상대강도지표 전략
  # MACD = Fast EMA - Slow EMA 에 대한 Exponential Moving Average 값을 이용함


# 과매수 구간 : 기초지수 Long / 인버스지수 Short
# 과매도 구간 : 인버스지수 Long / 기초지수 Short

# 0. 기초 prameter 입력
# 1. 시그널 생성하기
# 2. 수익률 생성하기
import pandas as pd
from pandas_datareader import data
from datetime import datetime
from numpy import inf

#. 0. parameter 입력하기
start_date = '2015-07-01'
end_date = datetime.today()
num_periods_fast = 5                 # time period for the fast EMA
K_fast = 2 / (num_periods_fast + 1) # smoothing factor for fast EMA
num_periods_slow = 20               # time period for slow EMA
K_slow = 2 / (num_periods_slow + 1) # smoothing factor for slow EMA
TICKER = ['SPY', 'SH']
df_total = data.DataReader(TICKER, 'yahoo', start_date, end_date)['Adj Close']
num_periods_macd = 20 # MACD EMA time period
K_macd = 2 / (num_periods_macd + 1) # MACD EMA smoothing factor


#. 1. 데이터 불러오기
df =  data.DataReader(TICKER[0], 'yahoo', start_date, end_date)
close = df['Adj Close']


# 2. 시그널 생성하기
def get_APO_signal(close, K_fast, K_slow) :
  ema_fast = 0
  ema_slow = 0
  ema_fast_values = []
  ema_slow_values = []
  apo_values = []
  for close_price in close:
      if (ema_fast == 0): # first observation
          ema_fast = close_price
          ema_slow = close_price
      else:
          ema_fast = (close_price - ema_fast) * K_fast + ema_fast
          ema_slow = (close_price - ema_slow) * K_slow + ema_slow
      ema_fast_values.append(ema_fast)
      ema_slow_values.append(ema_slow)
      apo_values.append(ema_fast - ema_slow)
  df_result = df_total.assign(F_EMA=pd.Series(ema_fast_values, index=df.index))
  df_result = df_result.assign(S_EMA=pd.Series(ema_slow_values, index=df.index))
  df_result = df_result.assign(APO=pd.Series(apo_values, index=df_result.index))
  df_result[TICKER[0]]  =  df_result['APO'].apply(lambda x : '1' if x>0 else '0')
  df_result[TICKER[1]]  =  df_result['APO'].apply(lambda x : '0' if x>0 else '1')
  df_signal = df_result[TICKER]
  return df_signal, df_result

df_signal_apo = get_APO_signal(close, K_fast, K_slow)[0]
df_result_apo = get_APO_signal(close, K_fast, K_slow)[1]

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize = (15,10))
# ax1 = fig.add_subplot(211, ylabel='Google price in $')
# df_total[TICKER[0]].plot(ax=ax1, color='g', lw=2., legend=True)
# df_result_apo['F_EMA'].plot(ax=ax1, color='b', lw=2., legend=True)
# df_result_apo['S_EMA'].plot(ax=ax1, color='r', lw=2., legend=True)
# ax2 = fig.add_subplot(212, ylabel='APO')
# df_result_apo['APO'].plot(ax=ax2, color='black', lw=2., legend=True)
# ax2.plot(df_result_apo.index, df_result_apo['APO']==0, '.', markersize = 5, color = 'r')
# plt.show()

def get_MACD_signal(close, K_fast, K_slow, K_macd) :
  ema_fast = 0
  ema_slow = 0
  ema_macd = 0
  ema_fast_values = []
  ema_slow_values = []
  macd_values = []
  macd_signal_values = []
  macd_histogram_values = []
  for close_price in close:
      if (ema_fast == 0): # first observation
          ema_fast = close_price
          ema_slow = close_price
      else:
          ema_fast = (close_price - ema_fast) * K_fast + ema_fast
          ema_slow = (close_price - ema_slow) * K_slow + ema_slow
      ema_fast_values.append(ema_fast)
      ema_slow_values.append(ema_slow)
      macd = ema_fast - ema_slow  # MACD is fast_MA - slow_EMA
      if ema_macd == 0:
          ema_macd = macd
      else:
          ema_macd = (macd - ema_macd) * K_macd + ema_macd  # signal is EMA of MACD values
      macd_values.append(macd)
      macd_signal_values.append(ema_macd)
      macd_histogram_values.append(macd - ema_macd)

  df_result = df_total.assign(F_EMA=pd.Series(ema_fast_values, index=df.index))
  df_result = df_result.assign(S_EMA=pd.Series(ema_slow_values, index=df.index))
  df_result = df_result.assign(MACD=pd.Series(macd_values, index=df_result.index))
  df_result = df_result.assign(MA_MACD=pd.Series(macd_signal_values, index=df_result.index))
  df_result = df_result.assign(HIS_MACD=pd.Series(macd_histogram_values, index=df_result.index))
  df_result[TICKER[0]]  =  df_result['MACD'].apply(lambda x : '1' if x>0 else '0')
  df_result[TICKER[1]]  =  df_result['MACD'].apply(lambda x : '0' if x>0 else '1')
  df_signal = df_result[TICKER]
  return df_signal, df_result

df_signal_macd = get_MACD_signal(close, K_fast, K_slow, K_macd)[0]
df_result_macd = get_MACD_signal(close, K_fast, K_slow, K_macd)[1]

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize = (15,10))
# ax1 = fig.add_subplot(311, ylabel='Google price in $')
# df_total[TICKER[0]].plot(ax=ax1, color='g', lw=2., legend=True)
# df_result_macd['F_EMA'].plot(ax=ax1, color='b', lw=2., legend=True)
# df_result_macd['S_EMA'].plot(ax=ax1, color='r', lw=2., legend=True)
# ax2 = fig.add_subplot(312, ylabel='MACD')
# df_result_macd['MACD'].plot(ax=ax2, color='black', lw=2., legend=True)
# ax2.plot(df_result_macd.index, df_result_macd['MACD']==0, '.', markersize = 5, color = 'r')
# ax3 = fig.add_subplot(313, ylabel='MACD')
# df_result_macd['HIS_MACD'].plot(ax=ax3, color='r', kind='bar', legend=True, use_index=False)
# plt.show()

import statistics as stats

def get_RSI_signal(close) :
  time_period = 20  # look back period to compute gains & losses
  gain_history = []  # history of gains over look back period (0 if no gain, magnitude of gain if gain)
  loss_history = []  # history of losses over look back period (0 if no loss, magnitude of loss if loss)
  avg_gain_values = []  # track avg gains for visualization purposes
  avg_loss_values = []  # track avg losses for visualization purposes
  rsi_values = []  # track computed RSI values
  last_price = 0  # current_price - last_price > 0 => gain. current_price - last_price < 0 => loss.
  for close_price in close:
      if last_price == 0:
          last_price = close_price

      gain_history.append(max(0, close_price - last_price))
      loss_history.append(max(0, last_price - close_price))
      last_price = close_price

      if len(gain_history) > time_period:  # maximum observations is equal to lookback period
          del (gain_history[0])
          del (loss_history[0])

      avg_gain = stats.mean(gain_history)  # average gain over lookback period
      avg_loss = stats.mean(loss_history)  # average loss over lookback period

      avg_gain_values.append(avg_gain)
      avg_loss_values.append(avg_loss)

      rs = 0
      if avg_loss > 0:  # to avoid division by 0, which is undefined
          rs = avg_gain / avg_loss

      rsi = 100 - (100 / (1 + rs))
      rsi_values.append(rsi)

  df_result = df_total.assign(RSI_Gain=pd.Series(avg_gain_values, index=df.index))
  df_result = df_result.assign(RSI_Loss=pd.Series(avg_loss_values, index=df.index))
  df_result = df_result.assign(RSI=pd.Series(rsi_values, index=df_result.index))
  df_result[TICKER[0]]  =  df_result['RSI'].apply(lambda x : '1' if x>50 else '0')
  df_result[TICKER[1]]  =  df_result['RSI'].apply(lambda x : '0' if x>50 else '1')
  df_signal = df_result[TICKER]
  return df_signal, df_result

df_signal_rsi = get_RSI_signal(close)[0]
df_result_rsi = get_RSI_signal(close)[1]

import matplotlib.pyplot as plt
fig = plt.figure(figsize = (15,10))
ax1 = fig.add_subplot(411, ylabel='Price in $')
df_total[TICKER[0]].plot(ax=ax1, color='g', lw=2., legend=True)
df_result_apo['F_EMA'].plot(ax=ax1, color='b', lw=2., legend=True)
df_result_apo['S_EMA'].plot(ax=ax1, color='r', lw=2., legend=True)
ax2 = fig.add_subplot(412, ylabel='RS')
df_result_rsi['RSI_Gain'].plot(ax=ax2, color='g', lw=2., legend=True)
df_result_rsi['RSI_Loss'].plot(ax=ax2, color='r', lw=2., legend=True)
ax3 = fig.add_subplot(413, ylabel='RSI')
df_result_rsi['RSI'].plot(ax=ax3, color='b', lw=2., legend=True)
ax3.axhline(y=50, color='r', linewidth=1)
ax4 = fig.add_subplot(414, ylabel='MACD')
df_result_macd['MACD'].plot(ax=ax4, color='black', lw=2., legend=True)
ax4.axhline(y=0, color='r', linewidth=1)
plt.show()


# 3. 수익률 생성하기
def get_return(df_total, df_signal):
  df_total = df_total.astype(float)
  df_signal = df_signal.astype(float)
  dff  = pd.DataFrame((df_signal * df_total))
  dff = dff.pct_change().fillna(0)
  dff['total'] = dff[TICKER[0]] + dff[TICKER[1]]
  dff[dff == inf] = 0
  return dff

dff = get_return(df_total, df_signal_macd)

plt.figure(figsize=(17,7))
plt.title('Long Short Strategy by Absolute Price Osciliator')
plt.plot((1 + dff['total']).cumprod() - 1, label = 'L/S by Signal')
plt.plot((1+ df_total[TICKER[0]].pct_change().fillna(0)).cumprod() -1 , label = TICKER[0])
plt.legend()
plt.show()