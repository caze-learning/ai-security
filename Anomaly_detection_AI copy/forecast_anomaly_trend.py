import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
import streamlit as st
from streamlit_utils import add_logo
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
plt.style.use('fivethirtyeight')

cluster_df = pd.DataFrame()
namespace_df = pd.DataFrame()

def select_csv_path():
    if st.session_state.data_option == 'Cluster':
        csv_path = './new_data/Cluster_level_Cost_data.csv'
    elif st.session_state.data_option == 'Namespace':
        csv_path = './new_data/Namespace_level_Cost_data.csv'
    return csv_path

def get_namespaces_and_cloumn(csv_path):
    namespace_df = pd.read_csv(csv_path)
    namespace_list = list(namespace_df.Namespace.unique())
    print(namespace_list)
    column_list = list(namespace_df.columns)
    print(column_list)
    return namespace_list, column_list

def get_cluster_column(csv_path):
    cluster_df = pd.read_csv(csv_path)
    column_list = list(cluster_df.columns)
    return column_list



def predict_forecasting(col1,  periods, Frequency, Frequency_type):
    col1.subheader(f"Forecasting the Cost")
    col1.markdown(f"Anticipating resource usage and cost over a future period of time. This enables expectations to be managed for existing and future projects.")
    col1.subheader(f"", divider='blue')
    freq_dict = {"Month": 'M', "Week": 'W', "Day": 'D', "Hour": 'H' , "Minute": 'min', "Second": 'S'}
    freq_str = str(Frequency) + freq_dict[Frequency_type]
    csv_path = select_csv_path()
    data_df = pd.read_csv(csv_path)
    # convert the 'Date' column to datetime format
    data_df['Time']= pd.to_datetime(data_df['Time'], format = "%d/%m/%y %H:%M")
    data_df.rename(columns={'Time' : 'DateTime'}, inplace=True)
    disp_str =""
    if st.session_state.data_option == 'Namespace':
        data_df = data_df.loc[data_df[st.session_state.data_option] == st.session_state.namespace_option]
        disp_str = disp_str + st.session_state.cost_option + " Cost forecast for  " + st.session_state.namespace_option + " Namespace for next " + str(periods * Frequency) + " " + Frequency_type + "'s with an intervel of " + str(Frequency) + " " + Frequency_type
    else:
        disp_str = disp_str + st.session_state.cost_option + " Cost forecast for the cluster for next " + str(periods * Frequency) + " " + Frequency_type + "'s with an intervel of " + str(Frequency) + " " + Frequency_type
        # print(data_df.shape)
    data_df = data_df.set_index('DateTime')
    analysis_df = data_df[[st.session_state.cost_option]]
    analysis_prophet = analysis_df.reset_index().rename(columns={'DateTime' : 'ds',st.session_state.cost_option : 'y'})
    analysis_prophet = analysis_prophet.reset_index().dropna(subset=['y'])
    if analysis_prophet.shape[0] != 0:
        model = Prophet()
        model.fit(analysis_prophet)
        future_data = model.make_future_dataframe(periods=periods,freq=freq_str, include_history=False)
        future_data_fcst = model.predict(future_data)
        future_data_df = future_data_fcst.reset_index().rename(columns={'ds' : 'DateTime', 'yhat' : 'Expected', 'yhat_upper' : 'Expected between(upper value)', 'yhat_lower' : 'Expected between(lower value)'})
        future_data_df= future_data_df[['DateTime', 'Expected', 'Expected between(upper value)', 'Expected between(lower value)']]
        # print(future_data_df)
        formatted_df = future_data_df.style.format({"Expected": "{:.6f}".format})

        fig,ax = plt.subplots(figsize =(30,5))
        fig = model.plot(future_data_fcst, ax=ax)
        ax.set_title("prophet forecast")
        col1.markdown(f"")
        col1.markdown(f":red[{disp_str}]")
        col1.markdown(f"")
        col1.pyplot(fig)
        col1.markdown(f"")
        col1.dataframe(formatted_df)
    else:
        for i in range (3):
            col1.markdown("")
        col1.subheader(f""" :red[No valid data to analyze, please check the data] """)

def predict_trend(col1):
    col1.subheader(f"Trend and Seasonality analysis of Cost Data")
    col1.markdown(f"Shows the general tendency of the data to increase or decrease during a long period of time.")
    col1.subheader(f"", divider='blue')
    csv_path = select_csv_path()
    data_df = pd.read_csv(csv_path)
    # convert the 'Date' column to datetime format
    data_df['Time']= pd.to_datetime(data_df['Time'], format = "%d/%m/%y %H:%M")
    data_df.rename(columns={'Time' : 'DateTime'}, inplace=True)
    disp_str =""
    if st.session_state.data_option == 'Namespace':
        data_df = data_df.loc[data_df[st.session_state.data_option] == st.session_state.namespace_option]
        disp_str = disp_str + st.session_state.cost_option + " Cost Trend of " + st.session_state.namespace_option + " Namespace"
    else:
        disp_str = disp_str + st.session_state.cost_option + " Cost Trend" +" of the cluster"
        # print(data_df.shape)
    data_df = data_df.set_index('DateTime')
    analysis_df = data_df[[st.session_state.cost_option]]
    analysis_prophet = analysis_df.reset_index().rename(columns={'DateTime' : 'ds',st.session_state.cost_option : 'y'})
    analysis_prophet = analysis_prophet.reset_index().dropna(subset=['y'])
    if analysis_prophet.shape[0] != 0:
        model = Prophet()
        model.fit(analysis_prophet)
        analysis_fcst = model.predict(analysis_prophet)
        print(type(analysis_fcst))
        print(analysis_fcst.shape)
        print(analysis_fcst.columns)
        # analysis_fcst.to_csv("./trend_values.csv")
        print(type(model))
        fig2,ax2 = plt.subplots(figsize =(30,5))
        fig2 = model.plot_components(analysis_fcst, figsize=(30,10))
        fig2.savefig("trend.png")
        print(type(fig2))
        print(fig2)
        col1.markdown(f"")
        col1.markdown(f":red[{disp_str}]")
        col1.markdown(f"")
        ax2.set_title("Data Trend")
        col1.pyplot(fig2)
    else:
        for i in range (3):
            col1.markdown("")
        col1.subheader(f""" :red[No valid data to analyze, please check the data] """)
  

def predict_anomaly(col1):
    col1.subheader(f"Unexpected cost behaviour (Anomaly)")
    col1.markdown("Based on your usage and billing, wherever there is an unexpected cost behaviour due to your resource usage, they are captured for your reference. This along with cost saving recommendations, trend and forecasting, you can build a comprehensive and optimzied budget")
    col1.subheader(f"", divider='blue')
    csv_path = select_csv_path()
    data_df = pd.read_csv(csv_path)
    # convert the 'Date' column to datetime format
    data_df['Time']= pd.to_datetime(data_df['Time'], format = "%d/%m/%y %H:%M")
    data_df.rename(columns={'Time' : 'DateTime'}, inplace=True)
    disp_str = ""
    if st.session_state.data_option == 'Namespace':
        data_df = data_df.loc[data_df[st.session_state.data_option] == st.session_state.namespace_option]
        disp_str = disp_str + "Anomalies detected for "+ st.session_state.namespace_option + " Namespace "+ st.session_state.cost_option + " Cost"
    else:
        disp_str = disp_str + "Anomalies detected for " + st.session_state.cost_option + " Cost" +" of the cluster"
        # print(data_df.shape)
    data_df = data_df.set_index('DateTime')
    analysis_df = data_df[[st.session_state.cost_option]]
    analysis_prophet = analysis_df.reset_index().rename(columns={'DateTime' : 'ds',st.session_state.cost_option : 'y'})
    analysis_prophet = analysis_prophet.reset_index().dropna(subset=['y'])
    if analysis_prophet.shape[0] != 0:
        model = Prophet()
        model.fit(analysis_prophet)
        analysis_fcst = model.predict(analysis_prophet)
        analysis_fcst['fact'] = analysis_prophet['y']
        analysis_st_df = analysis_fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper','fact']].copy()
        analysis_st_df['anomaly'] = 0
        analysis_st_df.loc[analysis_st_df['fact'] > analysis_st_df['yhat_upper'], 'anomaly'] = 1
        analysis_st_df.loc[analysis_st_df['fact'] < analysis_st_df['yhat_lower'], 'anomaly'] = -1
        #anomaly importances
        analysis_st_df['importance'] = 0
        analysis_st_df.loc[analysis_st_df['anomaly'] ==1, 'importance'] = (analysis_st_df['fact'] - analysis_st_df['yhat_upper'])/analysis_st_df['fact']
        analysis_st_df.loc[analysis_st_df['anomaly'] ==-1, 'importance'] = (analysis_st_df['yhat_lower'] - analysis_st_df['fact'])/analysis_st_df['fact']
        analysis_st_df = analysis_st_df.reset_index().rename(columns={'ds' : 'DateTime', 'yhat' : 'Expected', 
                                                                        'yhat_upper' : 'Expected between(upper value)', 
                                                                        'yhat_lower' : 'Expected between(lower value)',
                                                                        'fact' : 'Current value'})
        anomaly_df = analysis_st_df.loc[analysis_st_df['anomaly'] != 0]
        anomaly_df= anomaly_df[['DateTime', 'Expected', 'Expected between(upper value)', 'Expected between(lower value)', 'Current value', 'importance']]
        anomaly_df = anomaly_df.style.format({"Expected": "{:.6f}".format})

        fig3,ax1 = plt.subplots(figsize =(30,5))
        ax1.scatter(analysis_df.index, analysis_df[st.session_state.cost_option],color = "r")
        fig3 = model.plot(analysis_fcst, ax=ax1)
        ax1.set_title("Data Anomaly")
        col1.markdown(f"")
        col1.markdown(f":red[{disp_str}]")
        col1.markdown(f"")
        col1.pyplot(fig3)
        col1.markdown(f"")
        col1.dataframe(anomaly_df)
    else:
        for i in range (3):
            col1.markdown("")
        col1.subheader(f""" :red[No valid data to analyze, please check the data] """)


st.set_page_config(
    page_title="CazeLabs Intelligent-Insights", page_icon=":chart:", layout="wide"
)

# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 2rem;
                    padding-right: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

st.header(
    """ :blue[CazeLabs Data Analysis and Prediction]""", divider='rainbow'
)

my_logo = add_logo(logo_path='./media/cazelabs_logo2.jpeg', width=250, height=70)
st.sidebar.image(my_logo)
st.session_state.no_submit = True
col_ = st.columns(1)
col_1 = col_[0]

col = st.columns(1)
col1 = col[0]

with st.sidebar:
    #url_links = st.text_area("Enter Youtube URL")
    st.caption(f"Select Options and Submit. :arrow_down:  ")
    for i in range(1):
        st.write(f"")
    data_options = ("Cluster Level", "Namespace Level")
    data_index = st.radio(
        "Choose the Data", range(len(data_options)), format_func=lambda x: data_options[x]
    )
    st.session_state.data_option = data_options[0]
    # cluster_options = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"]
    # st.session_state.cluster = st.selectbox("Select the cluster", cluster_options)
    if data_index == 0:
        st.session_state.data_option = 'Cluster'
        csv_path = select_csv_path()
        # print(csv_path)
        column_list = get_cluster_column(csv_path)
        for i in range(1):
            st.write(f"")
        st.session_state.cost_option = st.selectbox("Cost to be analyzed", column_list[1:])
    elif data_index == 1:
        st.session_state.data_option = 'Namespace'
        csv_path = select_csv_path()
        # print(csv_path)
        namespace_list, column_list =  get_namespaces_and_cloumn(csv_path)
        for i in range(1):
            st.write(f"")
        st.session_state.namespace_option = st.selectbox("Which namespace to be analyzed", namespace_list)
        for i in range(1):
            st.write(f"")
        st.session_state.cost_option = st.selectbox("Cost to be analyzed", column_list[2:])

    for i in range(1):
            st.write(f"")
    options = ("Trend", "Anomaly" , "Forecasting")
    index = st.radio(
        "Choose the Option", range(len(options)), format_func=lambda x: options[x]
    )
    st.session_state.option = options[0]
    # print(index)
    if index == 0:
        st.session_state.option = 'Trend'
    elif index == 1:
        st.session_state.option = 'Anomaly'
    elif index == 2:
        st.session_state.option = 'Forecasting'
        periods = st.number_input('Number of periods to forecast', min_value=1, value = 10)
        Frequency_types = ["Month", "Week", "Day", "Hour" , "Minute", "Second"]
        Frequency_type  = st.selectbox("Frequency Type :", Frequency_types)
        Frequency = st.number_input('Frequency Intervel :', min_value=1)
        # print(freq_str)
    if st.button("Submit"):
        # print(st.session_state.option)
        if st.session_state.option == "Trend":
            predict_trend(col1)
        elif st.session_state.option == "Anomaly":
            # print("Inside Anomaly")
            predict_anomaly(col1)
        elif st.session_state.option == "Forecasting":
            predict_forecasting(col1, periods, Frequency, Frequency_type)
    
    
