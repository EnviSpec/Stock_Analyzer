import pandas as pd
from yahooquery import Ticker


import numpy as np
# from pandas_datareader import data as web
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import datetime as dt
import mplfinance as mpf

import seaborn as sns

import yfinance as yf

import plotly.graph_objects as go
from statsmodels.tsa.ar_model import AutoReg, ar_select_order


import requests
import pandas as pd
import yfinance as yf
import json

import numpy as np
import plotly.express as px

import streamlit as st

""" 01 : Stock Trend Prediction Using ARIMA """

def stock_information(symbol):
  "Generates stock dataframe ready for ARIMA"

  df = yf.download(symbol)
  df = df.asfreq('d')
  df = df.fillna(method = 'ffill')

  df['Date'] = df.index
  df['Date'] = pd.to_datetime(df['Date'])
  df.set_index('Date', inplace=True)
  df.index.freq = 'D'  # Set the frequency to daily

  df = df.drop(columns=['Open','Adj Close','High', 'Volume','Low'])
  return df


def stock_trend_prediction(df):

  eigthy_percent = int(0.8*len(df))
  twenty_percent = len(df) - eigthy_percent

  lags = ar_select_order(df, maxlag = 30)
  model = AutoReg(df['Close'], lags.ar_lags)
  model_fit = model.fit()


  train_df = df.iloc[:eigthy_percent]
  test_df = df.iloc[eigthy_percent:]
  train_model = AutoReg(df, 500).fit(cov_type='HC0')
  start = len(train_df)
  end = len(train_df) + len(test_df) - 1

  prediction = train_model.predict(start = start, end = end, dynamic = True)

  forecast = (train_model.predict(start = end,
                               end = end + 300,
                               dynamic = True))

  # Set the style and figure size
  sns.set_style('darkgrid')
  pd.plotting.register_matplotlib_converters()
  sns.mpl.rc('figure', figsize=(19, 12))

  # Create the plotly figure
  fig = go.Figure()

  # Add the train_df, test_df, prediction, and forecast traces to the figure
  fig.add_trace(go.Scatter(x=train_df.index, y=train_df['Close'], name='Train'))
  fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Close'], name='Test'))
  fig.add_trace(go.Scatter(x=prediction.index, y=prediction, name='Prediction'))
  fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name='Forecast'))

  # Update the layout
  fig.update_layout(
      title='Stock Price Prediction',
      xaxis_title='Date',
      yaxis_title='Price',
      showlegend=True,
      legend=dict(x=0, y=1),
      autosize=True,
      margin=dict(l=50, r=50, t=50, b=50),
   )

  # Show the plotly figure
  return fig


""" 02 : Companies Evaluation using Comparable Analysis """

def fundamentals_information(tickers ,balance_sheet = True, cash_flow = False, income_statement = False, quarterly=False):
   """
   Input : ticker and fundamental sheet to be downloaded
   Output : Provides the fundamental data required of the given ticker
   """
   nosuccess = []
   income_statement_df = pd.DataFrame()
   balance_sheet_df = pd.DataFrame()
   cash_flow_df = pd.DataFrame()

   for ticker in tickers:
      print("Working on ",ticker)

      try:
         dfprice = Ticker(ticker)
         if (balance_sheet):

            if(quarterly):
               balance_sheet_df = pd.DataFrame(dfprice.balance_sheet('q'))
               balance_sheet_df.to_csv(f'Data/Balance_Sheet/balance_sheet_{ticker[:-3]}.csv')

            else:
               balance_sheet_df = pd.DataFrame(dfprice.balance_sheet())
               balance_sheet_df.to_csv(f'Data/Balance_Sheet/balance_sheet_{ticker[:-3]}.csv')
               
         elif(cash_flow):

            if (quarterly):
               cash_flow_df = pd.DataFrame(dfprice.cash_flow('q'))
               cash_flow_df.to_csv(f'Data/Cash_Flow/cash_flow_{ticker[:-3]}.csv')
            else:
               cash_flow_df = pd.DataFrame(dfprice.cash_flow())
               cash_flow_df.to_csv(f'Data/Cash_Flow/cash_flow_{ticker[:-3]}.csv')
               
            
         elif(income_statement):
            if(quarterly):
               income_statement_df = pd.DataFrame(dfprice.income_statement('q'))
               income_statement_df.to_csv(f'Data/Income_Statement/income_statement_{ticker[:-3]}.csv')
            else:
               income_statement_df = pd.DataFrame(dfprice.income_statement())
               income_statement_df.to_csv(f'Data/Income_Statement/income_statement_{ticker[:-3]}.csv')
               
      except:
         nosuccess.append(ticker)
         print("Found no information for : "+ ticker)
         continue


  
def overall_fundamental_information(tickers):

   fundamentals_information(tickers,quarterly=True)                  # For Balance Sheet
   fundamentals_information(tickers,False,True,quarterly=True)       # CashFlow
   fundamentals_information(tickers,False,False,True,quarterly=True) # Income Statement   

def company_information_df(tickers,finance_companies = True):
    "Generate Company Information Matrix of Symbols Provided"
    stock_fundametals_for_evaluation0 = {
    'STOCK' : 'Name',
    'Market Cap': 100,
    'Enterprise Value': 100,
    'Revenue' : 100,
    'Earnings' : 100,
    'EV/Revenue': 100,
    'P / E' : 100
    }

    company_information_data = pd.DataFrame(stock_fundametals_for_evaluation0,index=[])
    
    for ticker in tickers:
        
        balance = pd.read_csv(f'Data/Balance_Sheet/balance_sheet_{ticker[:-3]}.csv').replace(np.nan,0)
        income = pd.read_csv(f'Data/Income_Statement/income_statement_{ticker[:-3]}.csv').replace(np.nan,0)
        cash = pd.read_csv(f'Data/Cash_Flow/cash_flow_{ticker[:-3]}.csv').replace(np.nan,0)

        if 'NetDebt' not in balance.columns:
            if 'TotalDebt' in balance.columns:
                balance['NetDebt'] = balance['TotalDebt']
            else:
                balance['NetDebt'] = [0 for x in balance['TotalAssets']]
        if finance_companies:
            stock_fundametals_for_evaluation = {
            'STOCK' : ticker,
            'Market Cap': yf.Ticker(ticker).info['marketCap'],
            'Enterprise Value': yf.Ticker(ticker).info['marketCap'] + float(balance['NetDebt'].iloc[-1]*1000),
            'Revenue' : income['TotalRevenue'].iloc[-1]*1000,
            'Earnings' : income['NetIncomeCommonStockholders'].iloc[-1]*1000,
            'EV/Revenue': (yf.Ticker(ticker).info['marketCap'] + balance['NetDebt'].iloc[-1]*1000) / (income['TotalRevenue'].iloc[-1]*1000),
            'P / E' : (yf.Ticker(ticker).info['marketCap']) / (income['NetIncomeCommonStockholders'].iloc[-1]*1000)
    
            }
        
        else :
            stock_fundametals_for_evaluation = {
            'STOCK' : ticker,
            'Market Cap': yf.Ticker(ticker).info['marketCap'],
            'Enterprise Value': yf.Ticker(ticker).info['marketCap'] + balance['NetDebt'].iloc[-1]*1000,
            'Revenue' : income['TotalRevenue'].iloc[-1]*1000,
            'Earnings' : income['NetIncomeCommonStockholders'].iloc[-1]*1000,
            'EV/Revenue': (yf.Ticker(ticker).info['marketCap'] + balance['NetDebt'].iloc[-1]*1000) / (income['TotalRevenue'].iloc[-1]*1000),
            'EV/EBITDA': (yf.Ticker(ticker).info['marketCap'] + balance['NetDebt'].iloc[-1]*1000) / (income['EBITDA'].iloc[-1]*1000),
            'EV/EBIT': (yf.Ticker(ticker).info['marketCap'] + balance['NetDebt'].iloc[-1]*1000) / (income['EBIT'].iloc[-1]*1000),
            'P / E' : (yf.Ticker(ticker).info['marketCap']) / (income['NetIncomeCommonStockholders'].iloc[-1]*1000)
    
            }

        company_information_data = company_information_data._append(stock_fundametals_for_evaluation,ignore_index=True)

    return company_information_data


def price(symbol):

    ticker = yf.Ticker(symbol)

    # Get the historical data for the past day
    today_data = ticker.history(period="1d")

    # Retrieve the current stock price
    current_price = today_data["Close"].iloc[-1]

    # Print the current stock price
    return current_price

def fundamental_stock_value(symbol,company_evaluation_metrics,finance_companies = True):
    """
    Input : Symbol of company to evaluated
    Process : Stock Price is evaluated by comparing fundamentals
              of companies present in the same sector
    Output : An average stock value based on fundamentals
    """
    df = company_evaluation_metrics
    stock_current_price = price(symbol)

    if df['P / E'][:1].values > 0:
        df = df[df['P / E'] > 0]
        
        if finance_companies:
            average_ev_revenue = df['EV/Revenue'][1:].mean()
            average_pe = df['P / E'][1:].mean()

            difference_ev_revenue = df['EV/Revenue'][:1].values/average_ev_revenue
            difference_pe = df['P / E'][:1].values/average_pe

            average_stock_value = stock_current_price * ((1/difference_ev_revenue) + (1/difference_pe))
        else:
            average_ev_revenue = df['EV/Revenue'][1:].mean()
            average_pe = df['P / E'][1:].mean()
            average_ebitda = df['EV/EBITDA'][1:].mean()
            average_ebit = df['EV/EBIT'][1:].mean()
        
            difference_ev_revenue = df['EV/Revenue'][:1].values/average_ev_revenue
            difference_pe = df['P / E'][:1].values/average_pe
            difference_ebitda = df['EV/EBITDA'][:1].values/average_ebitda
            difference_ebit = df['EV/EBIT'][:1].values/average_ebit

            average_stock_value = stock_current_price * ((1/difference_ev_revenue) + (1/difference_pe) + (1/difference_ebitda) + (1/difference_ebit))
    
    
    
    elif df['P / E'][:1].values < 0:
        average_stock_value = "Company is in loss better to analyze after next quarter results"
    
    return average_stock_value


def company_information():
    nifty_50 = pd.read_html('https://en.wikipedia.org/wiki/Nifty_50')[2]
    nifty_50['Symbol'] = [i + '.NS' for i in nifty_50['Symbol']]
    nifty_info = pd.DataFrame(nifty_50['Symbol'])
    nifty_info['Sector'] = nifty_50['Sector[18]']

    return nifty_info

def company_information_200():
    nifty_200 = pd.read_csv('Data/ind_nifty200list.csv')
    nifty_200['Symbol'] = [x + '.NS' for x in nifty_200['Symbol']]
    nifty_200 = nifty_200.rename(columns={'Industry':'Sector'})
    return nifty_200

def company_industry(symbol,company_information):

    df = company_information
    sector = df[df['Symbol']==symbol]['Sector'].to_list()
    list_of_companies = df[df['Sector']==sector[0]]['Symbol'].to_list()
    list_of_companies.remove(symbol)
    list_of_companies.insert(0,symbol)
    return list_of_companies

""" 03 : Option-Chain Analysis """
import requests
import json

# new_url = 'https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY'
# headers = {'User-Agent': 'Mozilla/5.0'}
# page = requests.get(new_url,headers=headers)
# dajs = json.loads(page.text)


def fetch_oi(expiry_dt,symbol):

    new_url = f'https://www.nseindia.com/api/option-chain-indices?symbol={symbol}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    page = requests.get(new_url,headers=headers)
    dajs = json.loads(page.text)

    ce_values = [data['CE'] for data in dajs['records']['data'] if "CE" in data and data['expiryDate'] == expiry_dt]
    pe_values = [data['PE'] for data in dajs['records']['data'] if "PE" in data and data['expiryDate'] == expiry_dt]

    ce_dt = pd.DataFrame(ce_values).sort_values(['strikePrice'])
    pe_dt = pd.DataFrame(pe_values).sort_values(['strikePrice'])
    
    return ce_dt, pe_dt

columns_names = {

    'strikePrice':'STRIKE',
    'expiryDate':'EXPIRY',
    'underlying':'INDEX/STOCK',
    'identifier':'IDENTIFIER',
    'openInterest':'OI',
    'changeinOpenInterest':'CHNG IN OI',
    'pchangeinOpenInterest':'PCHNG IN OI',
    'totalTradedVolume':'VOLUME',
    'impliedVolatility':'IV',
    'lastPrice':'LTP',
    'change':'CHNG',
    'pChange':'PCHNG',
    'totalBuyQuantity':'TOTAL BUY QTY',
    'totalSellQuantity':'TOTAL SELL QTY',
    'bidQty': 'BID QTY',
    'bidprice':'BID',
    'askQty': 'ASK QTY',
    'askPrice':'ASK',
    'underlyingValue':'CURRENT VALUE'

}


def call_modification(trimmed_oi_data):
    
    call_options = trimmed_oi_data[['STRIKE','BID QTY','BID','ASK','ASK QTY','CHNG','LTP','IV','VOLUME','CHNG IN OI','OI', 'TOTAL BUY QTY', 'TOTAL SELL QTY']]
    call_options["STRIKE"] = call_options["STRIKE"].astype(int).astype(str)
    call_options['CE/PE'] = ['CE' for i in call_options['STRIKE']]
    call_options['STRIKE'] = call_options['STRIKE'] + call_options['CE/PE']

    return call_options
    
def put_modification(trimmed_oi_data):
    
    put_options = trimmed_oi_data[['STRIKE','BID QTY','BID','ASK','ASK QTY','CHNG','LTP','IV','VOLUME','CHNG IN OI','OI', 'TOTAL BUY QTY', 'TOTAL SELL QTY']]
    put_options["STRIKE"] = put_options["STRIKE"].astype(int).astype(str)
    put_options['CE/PE'] = ['PE' for i in put_options['STRIKE']]
    put_options['STRIKE'] = put_options['STRIKE'] + put_options['CE/PE']

    return put_options

def get_theta(raw_oi_dataframe,noOfDays,current_close):

    theta = []
    
    for i in range(len(raw_oi_dataframe)):

        strike = raw_oi_dataframe['STRIKE'][i]
        price = raw_oi_dataframe['BID'][i]

        EstimatedPrice = (current_close - strike)
        if EstimatedPrice >= 0:
            ChangeInPrice = price - EstimatedPrice
            ThetaDecay = ChangeInPrice/noOfDays
        else :
            ThetaDecay = price/noOfDays
            
        theta._append(ThetaDecay.round(2))
    
    raw_oi_dataframe['Theta'] = theta
    return raw_oi_dataframe

def oi_retriever(ce_dt,pe_dt,symbol):
    "Converts Raw Web OI data to useful dataframe for analysis"
    
    #raw_oi_dataframe = raw_oi_dataframe.replace('-',0)
    
    # type_to_change = {'BID QTY':float,
    #               'BID': float,
    #               'ASK QTY':float,
    #               'ASK': float,
    #               'CHNG':float,
    #               'LTP': float,
    #               'IV':float,
    #               'VOLUME': float,
    #               'CHNG IN OI':float,
    #               'OI': float,

    #               'BID QTY.1':float,
    #               'BID.1': float,
    #               'ASK QTY.1':float,
    #               'ASK.1': float,
    #               'CHNG.1':float,
    #               'LTP.1': float,
    #               'IV.1':float,
    #               'VOLUME.1': float,
    #               'CHNG IN OI.1':float,
    #               'OI.1': float,}
    #raw_oi_dataframe = raw_oi_dataframe.astype(type_to_change)
    # Getting the latest close
    latest = yf.download(symbol)
    last_close = latest['Close'].iloc[-1].round()
    last_close = round(last_close/100)*100
    


    current_index = np.where(ce_dt == last_close)[0][0]
    ce_oi_data = ce_dt.iloc[current_index-10:current_index+11]
    pe_oi_data = pe_dt.iloc[current_index-10:current_index+11]

    call_oi = call_modification(ce_oi_data)
    put_oi = put_modification(pe_oi_data)
    
    final_dataset = call_oi._append(put_oi,ignore_index=True)
    final_dataset['BUY PERCENTAGE'] = ((final_dataset['TOTAL BUY QTY']  /  (final_dataset['TOTAL BUY QTY'] + final_dataset['TOTAL SELL QTY']))*100).round(2)
    final_dataset['SELL PERCENTAGE'] = ((final_dataset['TOTAL SELL QTY']  /  (final_dataset['TOTAL BUY QTY'] + final_dataset['TOTAL SELL QTY']))*100).round(2)
    return final_dataset

def overall_information(option_contracts):

    """Input: Processed Option Contracts Data"""

    overall_info = pd.DataFrame()
    overall_info['Volume'] = (option_contracts.groupby(['CE/PE'])['VOLUME'].sum().values)
    overall_info['CE/PE'] = ['CE','PE']

    # OVERALL BID vs. ASK
    df = option_contracts[['STRIKE','CE/PE','BID QTY','ASK QTY']]
    df = df.reset_index()
    df = pd.melt(df, id_vars=['STRIKE','CE/PE'], value_vars=['BID QTY','ASK QTY'])
    df1 = df
    df1['Contracts'] = df['CE/PE'] + ' ' + df['variable']
    ce_pe_bidask = px.pie(df1, values='value', names='Contracts', color='Contracts',color_discrete_map={'CE BID QTY':'#097969','PE ASK QTY':'#088F8F', 'PE BID QTY': '#DC143C','CE ASK QTY':'#D2042D'} ,title='CE/PE : BID vs ASK')


    # Vizualisation of Overall CE/PE
    fig = px.pie(overall_info, values='Volume', names='CE/PE', color='CE/PE' ,color_discrete_map={'CE':'#79EA86','PE':'#e75757'},title='Overall Market Direction')
    st.plotly_chart(fig)
    st.plotly_chart(ce_pe_bidask)


def option_peformance(option_contracts,call_or_put):

    # CE/PE PERFORMANCE
    options = option_contracts[option_contracts['CE/PE'] == call_or_put]

    # CE/PE BID vs. ASK
    cnp_info = pd.DataFrame()
    cnp_info['Action'] = ['BID','ASK']
    cnp_info['Quantity'] = [options['BID QTY'].sum(),options['ASK QTY'].sum()]

    # STRIKE vs. CHNG IN PRICE
    price_info = pd.DataFrame()
    price_info['STRIKE'] = options['STRIKE']
    price_info['PCHNG'] = (options['CHNG']/(options['LTP'] - options['CHNG']))*100

    # INDIVIDUAL BID vs. ASK
    df = options[['STRIKE','CE/PE','BID QTY','ASK QTY']]
    df = df.reset_index()
    df = pd.melt(df, id_vars=['STRIKE','CE/PE'], value_vars=['BID QTY','ASK QTY'])

    # INDIVIDUAL BUY vs. SELL
    df2 = options[['STRIKE','BUY PERCENTAGE','SELL PERCENTAGE']]
    df2 = df2.reset_index()
    df2 = pd.melt(df2, id_vars='STRIKE', value_vars=['BUY PERCENTAGE','SELL PERCENTAGE'])



    # Visualization
    overall_bidask = px.pie(cnp_info, values='Quantity', names='Action', color='Action' ,color_discrete_map={'BID':'#79EA86','ASK':'#e75757'},title=call_or_put+' Bid vs. ASK')
    oi_change = px.bar(options,x='STRIKE',y='CHNG IN OI', title=call_or_put+' Change In OI')
    price_change = px.bar(price_info,x='STRIKE',y='PCHNG', title=call_or_put+' Percentage Change')
    individual_bidask = px.bar(df,x='STRIKE',y='value',color='variable',color_discrete_map={'BID QTY':'#79EA86','ASK QTY': '#e75757'},barmode='group',title=call_or_put+' Individual: BID vs. ASK')
    individual_buysell = px.bar(df2,x='STRIKE',y='value',color='variable',color_discrete_map={'BUY PERCENTAGE':'#79EA86','SELL PERCENTAGE': '#e75757'},barmode='group',title=call_or_put+' Individual: BUY vs. SELL') 

    st.plotly_chart(overall_bidask)
    st.plotly_chart(oi_change)
    st.plotly_chart(individual_bidask)
    st.plotly_chart(individual_buysell)
    st.plotly_chart(price_change)

