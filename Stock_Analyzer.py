from yahooquery import Ticker

# from pandas_datareader import data as web
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import datetime as dt
import mplfinance as mpf

import seaborn as sns

import plotly.graph_objects as go
from statsmodels.tsa.ar_model import AutoReg, ar_select_order


import requests
import pandas as pd
import yfinance as yf
import json

import numpy as np
import plotly.express as px

# Module Containing Major Project Functions
from importlib.machinery import SourceFileLoader
analyze = SourceFileLoader('project-module','project-module.py').load_module()

from importlib.machinery import SourceFileLoader
oal = SourceFileLoader('option_analysis_library','option_analysis_library.py').load_module()


import streamlit as st

# Tab titles
tabs = ["Trend Prediction", "Fundamental Value of a Stock", "Option Chain Analysis"]
selected_tab = st.sidebar.radio("Select Tab", tabs)

# ARIMA Predictions tab
if selected_tab == "Trend Prediction":
    st.title("Trend Prediction")
    # Add your ARIMA prediction code here
    stock_ticker = str(st.text_input("Enter Stock Ticker"))

    if len(stock_ticker) > 0 :
        stock = analyze.stock_information(stock_ticker)

        fig,train_df,test_df,prediction,forecast = analyze.stock_trend_prediction(stock)
        st.dataframe(train_df)
        st.plotly_chart(fig)

# Input Excel and Graphs tab
elif selected_tab == "Option Chain Analysis":
    st.title("Option Chain Analysis")

    
    option = st.radio("Select an option", ["Excel", "API"])

    if option == "Excel":
        stock_ticker = str(st.text_input("Enter Index Ticker"))
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            raw_oi_dataframe = pd.read_csv(uploaded_file)
            processed_oi_data = oal.oi_retriever(raw_oi_dataframe,stock_ticker)
            # st.write(processed_oi_data)
            oal.overall_information(option_contracts=processed_oi_data)
            oal.option_peformance(processed_oi_data,'CE')
            oal.option_peformance(processed_oi_data,'PE')

    else:
        stock_ticker = str(st.text_input("Enter Index Ticker"))
        if len(stock_ticker) > 0:
            expiry_date = str(st.text_input("Enter Expiry Date"))
        # if len(stock_ticker) > 0:
            ce_dt,pe_dt = analyze.fetch_oi(expiry_date,stock_ticker)

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

            ce_dt = ce_dt.rename(columns=columns_names)
            pe_dt = pe_dt.rename(columns=columns_names)

            if stock_ticker == "BANKNIFTY":

                option_contracts = analyze.oi_retriever(ce_dt,pe_dt,'^NSEBANK')
        
            elif stock_ticker == "NIFTY":
                option_contracts = analyze.oi_retriever(ce_dt,pe_dt,'^NSEI')

            analyze.overall_information(option_contracts)
            analyze.option_peformance(option_contracts,'CE')
            analyze.option_peformance(option_contracts,'PE')


elif selected_tab == "Fundamental Value of a Stock":
    st.title("Fundamental Value of a Stock")
    stock_ticker = str(st.text_input("Enter Stock Ticker"))

    if len(stock_ticker) > 0:
        symbol = stock_ticker
        #tickers = analyze.company_industry(symbol,analyze.company_information())
        tickers = analyze.company_industry(symbol,analyze.company_information_200()) # Testing of Nifty 200

        #finance_company = ['HDFCBANK.NS','AXISBANK.NS','ICICIBANK.NS','INDUSINDBK.NS','KOTAKBANK.NS','SBIN.NS','HDFCLIFE.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'SBILIFE.NS']
        nifty_50 = analyze.company_information_200()
        finance_company = nifty_50[nifty_50['Sector'] == 'Financial Services']['Symbol'].to_list()

        if len(tickers)>=2:
            if symbol in finance_company:
                df = analyze.company_information_df(tickers,finance_companies=True)
                st.dataframe(df)

                stock_value = analyze.fundamental_stock_value(symbol,df,finance_companies=True)
                if str(stock_value) != '[nan]':
                    st.write("Current Price :",analyze.price(symbol))
                    st.write("\nValue of Stock Using Comparable Analysis :",(stock_value))
                else:
                    st.write("Unable to Process..")
    
            else:
                df = analyze.company_information_df(tickers,finance_companies=False)
                st.dataframe(df)

                stock_value = analyze.fundamental_stock_value(symbol,df,finance_companies=False)
                if str(stock_value) != '[nan]':
                    st.write("Current Price :",analyze.price(symbol))
                    st.write("\nValue of Stock Using Comparable Analysis :",(stock_value))
                else:
                    st.write("Unable to Process..")

        else:
            stock_value = analyze.price(symbol)
            if str(stock_value) != '[nan]':
                st.write("Current Price :",analyze.price(symbol))
                st.write("\nValue of Stock Using Comparable Analysis :",(stock_value))
            else:
                st.write("Unable to Process..")
