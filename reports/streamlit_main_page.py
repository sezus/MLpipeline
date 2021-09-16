import streamlit as st
import sys
sys.path.append('..')
from src.utils.utils_p import Upload_Download_Pickle, Config_Paths, Model_Configs
import altair as alt
import plotly_express as px
import plotly.graph_objects as go
from src.featureProcessing.SplitDataset.Split_Train_Test import *
from src.featureProcessing.Feature_Selection.Wrapper import *
from src.featureProcessing.Feature_Selection.Hybrid import *
from reports.streamlit_EDA import *


@st.cache
def load_data():
        comb_path=Config_Paths().get_combineddatasets_path()
        Features_df=Upload_Download_Pickle().download_pickle(comb_path, 'Features_df')
        AllLabels_df=Upload_Download_Pickle().download_pickle(comb_path, 'AllLabels_df')
        return Features_df,AllLabels_df

def aboutSection():
    st.markdown('''
            # About
            This web-app was created using [streamlit](https://streamlit.io), [matplotlib](https://matplotlib.org), 
            [sci-kit learn](https://scikit-learn.org/stable/), [pandas](https://pandas.pydata.org/) and [NumPy](
            https://numpy.org). It is currently a WIP with plans to add more features soon!
            [Git Repository here!](https://github.com/eric-li18/modelweave)
            ''')

if __name__ == "__main__":
    st.sidebar.subheader("Choose Analysis Steps:")
    bars=["EDA", "Feature Selection", "Classification","Clustering","Remaining Useful Life Analysis"]

    option = st.sidebar.selectbox("Choose Analysis Step ",
                                  bars)
    Features_df,AllLabels_df = load_data()
    print(Features_df.shape)
    if option == "EDA":
        st.markdown('''
            # Visualize Data 
            This simple web-app allows you to see Smart PDM Data Analysis 
            ''')
        st.markdown('''
            ## All Turbine data             
            ''')
        st.write(Features_df.describe())
        #Show selected Turbine Plots:
        option=st.sidebar.selectbox('Which Turbine Data?',
            Features_df['TURBINE_NUMBER'].unique()) 
        if st.sidebar.button('Show dataframe'):
            st.markdown('''
            ## Selected Turbine Data 
            Selected Turbine Data
            ''')
            st.write(Features_df[(Features_df['TURBINE_NUMBER']==option)].describe())
        if st.sidebar.button('Show Power Plots'):
            #checks = ["Dataframe", "Show Gen.Operation Data Plots","Show Gearbox Data Plots","Show Main Bearing Data Plots","Show Pitch System Data Plots"]
            #check_boxes = [st.sidebar.checkbox(check, key=check) for check in checks]
            #if st.sidebar.button('Show Gen.Operation Data Plots'):
            #    print(Features_df.head())
            #    fig_Nacelle_Position = px.scatter(Features_df[(Features_df['TURBINE_NUMBER']==option)], x ='Windspeed_m_s',y='Power_kW',color='Nacelle_Position')# Plot!
            #    print(fig_Nacelle_Position)
            #    fign= st.plotly_chart(fig_Nacelle_Position)
            print(Features_df.columns)
            #EDA().show_power_wind_charts(Features_df,option)
            select=st.sidebar.selectbox("Choose Plot : ", ('Show Gen.Operation Data Plots','Show Gearbox Data Plots','Show Main Bearing Data Plots','Show Pitch System Data Plots'))
        
            plt=EDA().show_power_wind_charts(Features_df,option,select)
            st.plotly_chart(plt)

    elif option == "Feature Selection":
        st.markdown('''
            # Select Important features 
            This feature allows you to see Smart PDM Data Feature Selection Results
            ''')
        #Split_Train_Test().split_train_testset()
        #Hybrid().featureSelection()

    elif option == "Classification":
        st.markdown('''
            # Diagnosis for Fault Detection 
            This feature allows you to see Smart PDM Data Feature Selection Results
            ''')
        models=["XGBoost", "Catboost", "RandomForest","LightGBM","SVM"]

        option = st.sidebar.selectbox("Choose Classification Model ",
                                  models)
        #Model Hiper parametreleri
        #Validation set 
        #Optimization Metric
        #Training time
        #Metric details
        #Confusion_Matrix
        #Learning Curve
        #Permutation-based Importance
        #SHAP Importance
        #SHAP Dependence plots
        #SHAP Decision plots
        metric_path=Config_Paths().get_modelsOutputs_path()
        recall_score_each=Upload_Download_Pickle().download_pickle(metric_path,'bests_LightGBM_recall_score_list.pckl')
        recall_score_all=Upload_Download_Pickle().download_pickle(metric_path,'bests_LightGBM_recall_score_df.pckl')
        print(recall_score_each['recall_score_seperated'])
        print(recall_score_all['recall_score_combined'])
        recall_score=recall_score_each['recall_score_seperated'].append(recall_score_all['recall_score_combined'])
        print(recall_score)
        fig = go.Figure(data=[go.Table(header=dict(values=['Metric','T02 Scores', 'T03 Scores','T27 Scores','Combined']),
                 cells=dict(values=['recall_score',recall_score_each['recall_score_seperated'][0],recall_score_each['recall_score_seperated'][1],recall_score_each['recall_score_seperated'][2],recall_score_all['recall_score_combined']]))
                     ])
        fig.show()
    elif option == "Clustering":
        st.markdown('''
            # Diagnosis for Fault Detection 
            This feature allows you to see Smart PDM Data Feature Selection Results
            ''')
        #Split_Train_Test().split_train_testset()
        #Hybrid().featureSelection()
    elif option == "Remaining Useful Life Analysis":
        st.markdown('''
            # Remaining Useful Life Analysis
            This feature allows you to see Smart PDM Data Feature Selection Results
            ''')
        #Split_Train_Test().split_train_testset()
        #Hybrid().featureSelection()

