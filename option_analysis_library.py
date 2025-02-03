import pandas as pd
import numpy as np
import yfinance as yf


import plotly as pt
import plotly.express as px

import streamlit as st

""" 01 : Functions to prepare NSE Options Data For Analysis. """
def call_modification(trimmed_oi_data):
    
    call_options = trimmed_oi_data[['STRIKE','BID QTY','BID','ASK','ASK QTY','CHNG','LTP','IV','VOLUME','CHNG IN OI','OI']]
    call_options["STRIKE"] = call_options["STRIKE"].astype(int).astype(str)
    call_options['CE/PE'] = ['CE' for i in call_options['STRIKE']]
    call_options['STRIKE'] = call_options['STRIKE'] + call_options['CE/PE']

    return call_options
    
def put_modification(trimmed_oi_data):
    
    put_options = trimmed_oi_data[['STRIKE', 'BID QTY.1', 'BID.1', 'ASK.1', 'ASK QTY.1','CHNG.1', 'LTP.1', 'IV.1', 'VOLUME.1', 'CHNG IN OI.1', 'OI.1']]
    put_options.columns = ['STRIKE','BID QTY','BID','ASK','ASK QTY','CHNG','LTP','IV','VOLUME','CHNG IN OI','OI']
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

def oi_retriever(raw_oi_dataframe,symbol):
    "Converts Raw Web OI data to useful dataframe for analysis"
    
    raw_oi_dataframe = raw_oi_dataframe.replace('-',0)
    
    type_to_change = {'BID QTY':float,
                  'BID': float,
                  'ASK QTY':float,
                  'ASK': float,
                  'CHNG':float,
                  'LTP': float,
                  'IV':float,
                  'VOLUME': float,
                  'CHNG IN OI':float,
                  'OI': float,

                  'BID QTY.1':float,
                  'BID.1': float,
                  'ASK QTY.1':float,
                  'ASK.1': float,
                  'CHNG.1':float,
                  'LTP.1': float,
                  'IV.1':float,
                  'VOLUME.1': float,
                  'CHNG IN OI.1':float,
                  'OI.1': float,}
    raw_oi_dataframe = raw_oi_dataframe.astype(type_to_change)
    # Getting the latest close
    latest = yf.download(symbol)
    last_close = latest['Close'].iloc[-1].round()
    last_close = round(last_close/100)*100
    


    current_index = np.where(raw_oi_dataframe == last_close)[0][0]
    trimmed_oi_data = raw_oi_dataframe.iloc[current_index-10:current_index+11]

    call_oi = call_modification(trimmed_oi_data)
    put_oi = put_modification(trimmed_oi_data)
    
    final_dataset = call_oi._append(put_oi,ignore_index=True)
    return final_dataset

""" 02 : Functions for Options Analysis Purpose """

""" ** Overall Market Trend """
def overall_information(option_contracts):

    """Input: Processed Option Contracts Data"""

    overall_info = pd.DataFrame()
    overall_info['Volume'] = (option_contracts.groupby(['CE/PE'])['VOLUME'].sum().values)
    overall_info['CE/PE'] = ['CE','PE']

    df = option_contracts[['STRIKE','CE/PE','BID QTY','ASK QTY']]
    df = df.reset_index()
    df = pd.melt(df, id_vars=['STRIKE','CE/PE'], value_vars=['BID QTY','ASK QTY'])
    df1 = df
    df1['Contracts'] = df['CE/PE'] + ' ' + df['variable']
    ce_pe_bidask = px.pie(df1, values='value', names='Contracts', color='Contracts' ,title='CE/PE : BID vs ASK')


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
    df = options[['STRIKE','BID QTY','ASK QTY']]
    df = df.reset_index()
    df = pd.melt(df, id_vars='STRIKE', value_vars=['BID QTY','ASK QTY'])

    # Visualization
    overall_bidask = px.pie(cnp_info, values='Quantity', names='Action', color='Action' ,color_discrete_map={'BID':'#79EA86','ASK':'#e75757'},title=call_or_put+' Bid vs. ASK')
    oi_change = px.bar(options,x='STRIKE',y='CHNG IN OI', title=call_or_put+' Change In OI')
    price_change = px.bar(price_info,x='STRIKE',y='PCHNG', title=call_or_put+' Percentage Change')
    individual_bidask = px.bar(df,x='STRIKE',y='value',color='variable',color_discrete_map={'BID QTY':'#79EA86','ASK QTY': '#e75757'},barmode='group',title=call_or_put+' Individual: BID vs. ASK') 

    st.plotly_chart(overall_bidask)
    st.plotly_chart(oi_change)
    st.plotly_chart(price_change)
    st.plotly_chart(individual_bidask)



""" 03 : Functions for Weekly Analytics """

def oi_retriever_weekly(raw_oi_dataframe,symbol):
    "Converts Raw Web OI data to useful dataframe for analysis"
    
    call_oi = call_modification(raw_oi_dataframe)
    put_oi = put_modification(raw_oi_dataframe)
    
    final_dataset = call_oi._append(put_oi,ignore_index=True)
    return final_dataset

def get_dataset(available_date,expiry,symbol):

    file_read_name = 'Data/Feb/BN_' + available_date + '_Expiry_' + expiry + '.csv'
    web_oi_data = pd.read_csv(file_read_name)

    return oi_retriever_weekly(web_oi_data,symbol)

def individual_weekly_performance(final_dataset,contract_name):
    """ Shows performance of specific option contract on weekly basis """
    
    # Bid vs. Ask
    df1 = final_dataset[final_dataset['STRIKE'] == contract_name]

    df = df1[['Date','BID QTY','ASK QTY']]
    df = df.reset_index()
    df = pd.melt(df, id_vars='Date', value_vars=['BID QTY','ASK QTY'])
    #
    bid_vs_ask_weekly = px.bar(df,x='Date',y='value',color='variable',color_discrete_map={'BID QTY':'#79EA86','ASK QTY': '#e75757'},barmode='group',title=contract_name+': BID vs. ASK')
    
    # OI
    df1 = final_dataset[final_dataset['STRIKE'] == contract_name]
    oi = px.bar(df1,x='Date',y='OI', title=contract_name+' Change In OI')

    # Percentage Change
    price_info = pd.DataFrame()
    price_info['Date'] = df1['Date']
    price_info['PCHNG'] = (df1['CHNG']/(df1['LTP'] - df1['CHNG']))*100
    price_change = px.bar(price_info,x='Date',y='PCHNG', title= contract_name+' Percentage Change')
    
    oi.show()
    price_change.show()
    bid_vs_ask_weekly.show()



""" ** OPTION EXIT LIBRARY ** """

# Stock Manage

def options_tax(buy_price, quantity, sell_price):
    """
    Returns The Tax associated with options
    """
    exchange_charges = .0005*quantity*(buy_price + sell_price)
    sebi_turnover = quantity*(buy_price + sell_price)/10000000
    stt = sell_price * quantity * .000625
    gst = .18 * (40 + exchange_charges + sebi_turnover)
    stamp_duty = .00003 * (buy_price * quantity)

    total_charges = {
        'Total Charges': exchange_charges + sebi_turnover + stt + gst + stamp_duty + 40,
        'Broker Charge':40,
        'Exchange Charges': exchange_charges,
        'SEBI Turnover Fees': sebi_turnover,
        'Securities Transaction Tax': stt,
        'GST': gst,
        'Stamp Duty': stamp_duty,
    }

    return total_charges
def buy(available_balance, current_price,quantity_per_lot):
    """
    It gives the summary of exit points based on targets and percentage
    """
    qpl = quantity_per_lot
    no_of_lots = int(available_balance/(current_price*qpl))
    target = [10000,15000,20000,25000,30000,40000,50000]
    points_target = []
    for x in target:
        points_target_required = x/(no_of_lots * qpl)
        sell_price = points_target_required + current_price
        tax = options_tax(current_price,no_of_lots*qpl,sell_price)['Total Charges']
        points_target_required = x*(tax/x + .05 + 1)/(no_of_lots * qpl)
        points_target.append(points_target_required + current_price)
    
    # Exit points based on targets
    stock_target = {
        'Current Price':current_price,
        'No of Lots': no_of_lots,
        'Quantity': no_of_lots * qpl,
        '10000/1.0%': points_target[0],
        '15000/1.5%': points_target[1],
        '20000/2.0%': points_target[2],
        '25000/2.5%': points_target[3],
        '30000/3.0%': points_target[4],
        '40000/4.0%': points_target[5],
        '50000/5.0%': points_target[6],
    }
    
    # Exit points based on percentage
    percentage = [.01, .015, .02, .025, .03, .04, .05]
    points_percentage = []
    for percent in percentage:
        
        points = current_price * percent
        sell_price = points + current_price

        targeted_profit = no_of_lots * qpl * points

        tax = options_tax(current_price,no_of_lots*qpl,sell_price)['Total Charges']
        points_target_required = targeted_profit*(tax/targeted_profit + .05 + 1)/(no_of_lots * qpl)
        points_percentage.append(points_target_required + current_price)
        

    stock_percentage = {
        'Current Price':current_price,
        'No of Lots': no_of_lots,
        'Quantity': no_of_lots * qpl,
        '10000/1.0%': points_percentage[0],
        '15000/1.5%': points_percentage[1],
        '20000/2.0%': points_percentage[2],
        '25000/2.5%': points_percentage[3],
        '30000/3.0%': points_percentage[4],
        '40000/4.0%': points_percentage[5],
        '50000/5.0%': points_percentage[6]
    }

    index = ['Target']

    summary = pd.DataFrame(stock_target,index=index)
    summary = summary.transpose()
    summary['Percentage'] = stock_percentage.values()
    return summary

def selling_margin(strike_price, quantity, price):
    span = (strike_price * quantity)*.1
    premium = price*100
    extreme = span * .05
    volatility = 5000
    return span + premium + extreme + volatility


def compound(returns,period):

    return (1 + returns)**period - 1

def monthly_stat_ideal():

    monthly_option_return = (60 * 300.65)/selling_margin(45900, 60, 300.65)
    weekly_option_return = (41.15*100)/selling_margin(45600,100,41.15)
    daily_option_return = 3000/200000

    mor_alone = monthly_option_return
    wor_monthly = compound(weekly_option_return,4)
    dor_monthly = compound(daily_option_return,20)
    stats = {
        "Monthly Expiry Return":mor_alone,
        'Weekly Expiry Return': wor_monthly,
        'Daily Buy Return': dor_monthly,
        'Total Monthly Return': (0.0556*324235 + 0.034*487774 + 0.346*200000)/1000000
    }
    return stats
