## Ia and APP Create by Autexier Nicolas : nicolas.atx@gmx.fr
## Supervised By Lyon CHU

import streamlit as canvas
import pandas as pd 
from sklearn import svm
from lib import consume_validation
from lib.data_clean import DataClean
from lib.anomaly_detection import Anomaly_nature
from joblib import load

######################### 

canvas.set_option('deprecation.showfileUploaderEncoding', False) ## Warming from futur version of streamlit ?
canvas.set_option('deprecation.showPyplotGlobalUse', False)
canvas.set_page_config(page_title="Leukemia APL CLF", layout="wide", page_icon='ressources/cord17a36.png')

canvas.markdown('<style>.css-1aumxhk {background: linear-gradient(to right, #ffffff, #1F618D);}</style>',unsafe_allow_html=True)
canvas.markdown('<style>.reportview-container{background: linear-gradient(to left, #ffffff, #ffffff);}</style>',unsafe_allow_html=True)
canvas.markdown('<style>.css-145kmo2 {background: linear-gradient(to right, #ffffff, #1F618D);}</style>',unsafe_allow_html=True)
canvas.markdown('<style>h2{background: linear-gradient(to right, #ffffff, #1F618D);}</style>',unsafe_allow_html=True)
canvas.markdown('<style>.css-2trqyj {background: linear-gradient(to right, #ffffff, #1F618D);}</style>',unsafe_allow_html=True)

######################### 

## IA MODEL
global MODEL
MODEL = load("./Model/clf_model_2021-03-05_15_13_20_270634.joblib") 

## Databrick for prevent data leak
global DATABRICK
DATABRICK = load("./Model/databrick_model_2021-02-21_14_37_04_740893.joblib")

## Databrick for anomalies detection
global DATABRICK_ANO
DATABRICK_ANO = load("./Model/databrick_anomalies.joblib")

## 1.0 prediction become label
global LABEL_POSITIF
LABEL_POSITIF = "APL"

## 0.0 prediction become label
global LABEL_NEGATIF
LABEL_NEGATIF = "Non-APL"

## width of truster images
#global truster_width
#truster_width = 90

## Global trust warning
global TRUST_THRESHOLD
TRUST_THRESHOLD = 99
##########################   


def convert_pred_to_label(df_pred):
    if df_pred["Predictions"] == 1:
        return LABEL_POSITIF
    return LABEL_NEGATIF

def set_df(i_age,i_Fibrinogene_,i_Leucocytes_,i_VGM_,
            i_CCMH_,i_PNN_,i_Lymphos_,i_TP_):
    df = pd.DataFrame()
    df.at[0,'Patient_numbers'] = 12
    df.at[0,"Age"] = i_age
    df.at[0,"Leucocytes_(G_L)"] = i_Leucocytes_
    df.at[0,"VGM_(fL)"] = i_VGM_
    df.at[0,"CCMH_(g_L)"] = i_CCMH_
    df.at[0,"PNN_(G_L)"] = i_PNN_
    df.at[0,"Lymphos_(%)"] = i_Lymphos_
    df.at[0,"TP_(%)"] = i_TP_
    df.at[0,"Fibrinogene_(g_L)"] = i_Fibrinogene_
    return df

def get_cell_values(col,key,ordered=True):
    if ordered:
        value = col.number_input(f"{key}", value=0.0, key=f"{key}")
    if col.checkbox(f"Missing {key}"):
        value = None
    col.text("")
    col.text("")
    col.text("")
    return value

def AI_Job(df,single_pred=False):

    try:
        colA, colB = canvas.columns(2)
        my_bar = colA.progress(0)

        try:
            df = df.drop(columns=['target'])
        except:
            print('pass')
        my_bar.progress(5)

        x_valid = df

        if len(x_valid) < 2:
            single_pred = True
        
        colA.header("Data Selected")
        if single_pred:
            x_valid_bis = x_valid.drop(columns=['Patient_numbers'])
            colA.dataframe(x_valid_bis)
        else:
            colA.dataframe(x_valid)

        my_bar.progress(10)
        cleandata = DataClean(x_valid)
        my_bar.progress(30)
        x_valid_clean = cleandata.prod_clean_job(DATABRICK)
        my_bar.progress(40)
        m_val_step = consume_validation.ConsumeModelClf(x_valid_clean,MODEL,validation=False)
        my_bar.progress(50)
        m_val_step.predict_job()
        my_bar.progress(60)
        Final_return = m_val_step.show_result()
        my_bar.progress(70)
        colA.header("Prediction for datas")
        Final_return = Final_return.drop(columns=['Id'])
        Final_return['Predictions'] = Final_return.apply(convert_pred_to_label, axis=1)
        if not single_pred:
            Final_return['Patient_numbers'] = x_valid['Patient_numbers']
        display_predictions(colA, Final_return)
    except Exception as e:
        canvas.warning(e)
        return
    if single_pred:
        try:
            colA.header("Anomaly Dectection")
            fig, ano_pred = Anomalie_job(x_valid_clean) 
        except Exception as e:
            canvas.warning(e)
            return
        try:
            visual_truster(colA, Final_return.loc[0,'Trust%'], ano_pred)
            colA.pyplot(fig, clear_figure=True)
        except Exception as e:
            canvas.warning(e)
            return
        my_bar.progress(80)
    try:
        fig_expl = m_val_step.explain()
        colA.header("Explanation of prediction")
        colA.pyplot(fig_expl)
        my_bar.progress(90)
    except Exception as e:
        canvas.warning(e)
    my_bar.progress(100)

def display_predictions(col, Final_return):
    trust_value = min(Final_return['Trust%'])
    if  trust_value < TRUST_THRESHOLD:
        col.error(f"The minimal trust value {trust_value}% is below the {TRUST_THRESHOLD}%."+
        " You should be aware the results are outside the zone of trust for the predictions.")
    col.dataframe(Final_return)

def Anomalie_job(X_input):
    model_svm = svm.OneClassSVM(degree=1, nu=0.05, kernel='rbf', gamma='scale')
    AN = Anomaly_nature(model_svm, DATABRICK_ANO, None, -2.5,3, reduce_dim=True)
    _, fig, anomalie_prediction = AN.build_anomalies_model(X_input=X_input)
    return fig, anomalie_prediction

def visual_truster(col, trust_value, ano_pred):
    if int(ano_pred) == -1:
        col.error("Anomaly in data detected, check the data entry otherwise these values are out of the zone observed during the training of the AI")
        return
    else:
        col.succes("Data are in zone observed during the training of the AI")

def input_manager():

    my_expander = canvas.expander('Manual Entries', expanded=True)
    with my_expander:
        colA, colB, colC, colD = canvas.columns(4)
        i_age = get_cell_values(colA,"Age")
        i_Fibrinogene_ = get_cell_values(colA,"Fibrinogene_(g_L)")
        i_Leucocytes_ = get_cell_values(colA,"Leucocytes_(G_L)")
        i_VGM_ = get_cell_values(colB,"VGM_(fL)")
        i_CCMH_ = get_cell_values(colB,"CCMH_(g_L)")
        i_PNN_ = get_cell_values(colB,"PNN_(G_L)")
        # i_PNE_ = get_cell_values(colC,"PNE_(%)")
        i_Lymphos_ = get_cell_values(colC,"Lymphos_(%)")
        # i_Mono_ = get_cell_values(colC,"Mono_(G_L)")
        i_TP_ = get_cell_values(colC,"TP_(%)")

    df = set_df(i_age,i_Fibrinogene_,i_Leucocytes_,i_VGM_,
                i_CCMH_,i_PNN_,i_Lymphos_,i_TP_)
    return df

def show_privacy_policy():
    back = canvas.button("Back to the app")
    if back:
        privacy = False
    canvas.write("""App doesn't collect any data, in any form.
    
    This software is provided "as is," without warranty of any kind, express or implied. 
    In no event shall creator or its contributors be held liable for any direct, indirect, incidental, special or consequential damages 
    arising out of the use of or inability to use this software.

    Project is open source under BSD-3-Clause License see more details :
    """)
    canvas.write("BSD-3-Clause [link](https://github.com/Nico-Facto/Leukemia-Apl-Classification/blob/main/LICENSE)")

def ui_manager():
    """Store Ui elements and return var for button action """

    canvas.sidebar.title('Leukemia APL AI')
    

    privacy = canvas.sidebar.button("Privacy Policies")
    if privacy :
        show_privacy_policy()
        return

    canvas.sidebar.header("Batch Entries")
    uploaded_file = canvas.sidebar.file_uploader("",type="csv")

    if uploaded_file is not None:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        canvas.sidebar.write("unselect the csv file to return to manual entry")
        selec_patient = int(canvas.sidebar.number_input("Patient Number", value=0))
        if selec_patient != 0:
            df = df[df['Patient_numbers'].isin([selec_patient])]
            if len(df) <1:
                canvas.warning("Error in Id selected, is it present in the file ?")
                return
    else:
        canvas.sidebar.write("Csv example [link](https://github.com/Nico-Facto/Leukemia-Apl-Classification/blob/main/templates_bach_pediction/templates_bach_pediction.csv)")
        df = input_manager()

    if canvas.button("AI Prediction"):
        if uploaded_file is not None:
            AI_Job(df)
        else:
            AI_Job(df,single_pred=True)

if __name__ == "__main__":
    ui_manager()