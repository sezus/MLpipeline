   
import streamlit as st
import altair as alt
import plotly_express as px
import matplotlib as plt
import sys

class EDA():
    def show_power_wind_charts(self,Features_df,option,select):     
        count=0    
        #Show selected Turbine Plots:
        #select=st.sidebar.selectbox("Choose Plot : ", ('Show Gen.Operation Data Plots','Show Gearbox Data Plots','Show Main Bearing Data Plots','Show Pitch System Data Plots'))
        #b1=st.sidebar.button('Show Gen.Operation Data Plots', key="1")
        #b2=st.sidebar.button('Show Gearbox Data Plots', key="2")
        #b3=st.sidebar.button('Show Main Bearing Data Plots', key="3")
        #b4=st.sidebar.button('Show Pitch System Data Plots', key="4")
        if select=='Show Gen.Operation Data Plots': 
            print(Features_df[(Features_df['TURBINE_NUMBER']==option)].head())
            fig_Nacelle_Position = px.scatter(Features_df[(Features_df['TURBINE_NUMBER']==option)], x ='Windspeed_m_s',y='Power_kW',color='Nacelle_Position')# Plot!
            fign=st.plotly_chart(fig_Nacelle_Position)

        if select=='Show Gearbox Data Plots': 
            print(Features_df[(Features_df['TURBINE_NUMBER']==option)].head())
            fig_Nacelle_Position = px.scatter(Features_df[(Features_df['TURBINE_NUMBER']==option)], x ='Windspeed_m_s',y='Power_kW',color='Nacelle_Position')# Plot!
            return(fig_Nacelle_Position)

        if select=='Show Main Bearing Data Plots':  
            print(Features_df[(Features_df['TURBINE_NUMBER']==option)].head())
            fig_Nacelle_Position = px.scatter(Features_df[(Features_df['TURBINE_NUMBER']==option)], x ='Windspeed_m_s',y='Power_kW',color='Nacelle_Position')# Plot!
            fign=st.plotly_chart(fig_Nacelle_Position)

        if select=='Show Pitch System Data Plots':        
            print(Features_df[(Features_df['TURBINE_NUMBER']==option)].head())
            fig_blade2 = px.scatter(Features_df[(Features_df['TURBINE_NUMBER']==option)], x ='Windspeed_m_s',y='Power_kW',color='Blade2_act_val_A_degree')# Plot!
            figx=st.plotly_chart(fig_blade2)

            fig_blade3 = px.scatter(Features_df[(Features_df['TURBINE_NUMBER']==option)], x ='Windspeed_m_s',y='Power_kW',color='Blade3_act_val_A_degree')# Plot!
            figy=st.plotly_chart(fig_blade3)

            fig_blade1 = px.scatter(Features_df[(Features_df['TURBINE_NUMBER']==option)], x ='Windspeed_m_s',y='Power_kW',color='Blade1_act_val_degree')# Plot!
            figz=st.plotly_chart(fig_blade1)

            fig_blade1_set = px.scatter(Features_df[(Features_df['TURBINE_NUMBER']==option)], x ='Windspeed_m_s',y='Power_kW',color='Blade1_set_val_degree')# Plot!
            figt=st.plotly_chart(fig_blade1_set)

