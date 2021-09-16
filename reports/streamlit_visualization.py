import streamlit as st
import sys
sys.path.append('..')
from src.utils.utils_p import Upload_Download_Pickle, Config_Paths, Model_Configs
import altair as alt
import plotly_express as px
# Your imports goes below
def load_data():
    comb_path=Config_Paths().get_combineddatasets_path()
    Features_df=Upload_Download_Pickle().download_pickle(comb_path, 'Features_df')
    AllLabels_df=Upload_Download_Pickle().download_pickle(comb_path, 'AllLabels_df')
    return Features_df,AllLabels_df

def main():
    st.title("Smart PDM")
    st.markdown("DATA ANALYSIS")

    # Your code goes below
    Features_df,AllLabels_df=load_data() 

    #Show all Turbines Power & Windspeed:
    fig = px.scatter(Features_df, x ='Windspeed_m_s',y='Power_kW',color='TURBINE_NUMBER')# Plot!
    fig1=st.plotly_chart(fig)
    option=st.selectbox('Which Turbine Data?',
     Features_df['TURBINE_NUMBER'].unique()) 
    print(option)
    st.write('You selected:', option)

    checks = ["Dataframe", "Show Gen.Operation Data Plots","Show Gearbox Data Plots","Show Main Bearing Data Plots","Show Pitch System Data Plots"]
    check_boxes = [st.sidebar.checkbox(check, key=check) for check in checks]

    #Show selected Turbine Plots:
    if st.button('Show dataframe'):
        st.write(Features_df[(Features_df['TURBINE_NUMBER']==option)].head())
    if st.button('Show Gen.Operation Data Plots'): 
        print(Features_df[(Features_df['TURBINE_NUMBER']==option)].head())
        fig_Nacelle_Position = px.scatter(Features_df[(Features_df['TURBINE_NUMBER']==option)], x ='Windspeed_m_s',y='Power_kW',color='Nacelle_Position')# Plot!
        fign=st.plotly_chart(fig_Nacelle_Position)
    if st.button('Show Gearbox Data Plots'): 
        print(Features_df[(Features_df['TURBINE_NUMBER']==option)].head())
        fig_Nacelle_Position = px.scatter(Features_df[(Features_df['TURBINE_NUMBER']==option)], x ='Windspeed_m_s',y='Power_kW',color='Nacelle_Position')# Plot!
        fign=st.plotly_chart(fig_Nacelle_Position)
    if st.button('Show Main Bearing Data Plots'): 
        print(Features_df[(Features_df['TURBINE_NUMBER']==option)].head())
        fig_Nacelle_Position = px.scatter(Features_df[(Features_df['TURBINE_NUMBER']==option)], x ='Windspeed_m_s',y='Power_kW',color='Nacelle_Position')# Plot!
        fign=st.plotly_chart(fig_Nacelle_Position)
    if st.button('Show Pitch System Data Plots'):          
        print(Features_df[(Features_df['TURBINE_NUMBER']==option)].head())
        fig_blade2 = px.scatter(Features_df[(Features_df['TURBINE_NUMBER']==option)], x ='Windspeed_m_s',y='Power_kW',color='Blade2_act_val_A_degree')# Plot!
        figx=st.plotly_chart(fig_blade2)

        fig_blade3 = px.scatter(Features_df[(Features_df['TURBINE_NUMBER']==option)], x ='Windspeed_m_s',y='Power_kW',color='Blade3_act_val_A_degree')# Plot!
        figy=st.plotly_chart(fig_blade3)

        fig_blade1 = px.scatter(Features_df[(Features_df['TURBINE_NUMBER']==option)], x ='Windspeed_m_s',y='Power_kW',color='Blade1_act_val_degree')# Plot!
        figz=st.plotly_chart(fig_blade1)

        fig_blade1_set = px.scatter(Features_df[(Features_df['TURBINE_NUMBER']==option)], x ='Windspeed_m_s',y='Power_kW',color='Blade1_set_val_degree')# Plot!
        figt=st.plotly_chart(fig_blade1_set)

    c = alt.Chart(Features_df).mark_circle().encode(x='Power_kW', y='Windspeed_m_s', size='c', color='c')

if __name__ == "__main__":
    main()