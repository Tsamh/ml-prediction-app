# Application Streamlit de déploiement des 4 modèles du devoir :
#   - Taxi_Reg        : prix d'un trajet de taxi (régression)
#   - Car_Class       : voiture Venant / D'occasion (classification)
#   - Car_Reg         : prix de revente d'une voiture (régression)
#   - Telephone_Class : état d'un téléphone (classification)
import os
import base64
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Dossier de l'application (les chemins marchent quel que soit le dossier de lancement)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration de la page
st.set_page_config(page_title="ML Prediction App", page_icon="🤖", layout="wide")

# Charger le fichier de style (style.css)
def load_css(file_name):
    with open(os.path.join(BASE_DIR, file_name), encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css('style.css')


# Afficher une carte (image + titre + description)
def card(image, title, text, badge, color, bg):
    with open(os.path.join(BASE_DIR, 'images', image), 'rb') as f:
        img = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <div class="card">
      <div class="card-img" style="background:{bg}"><img src="data:image/svg+xml;base64,{img}"/></div>
      <div class="card-body">
        <span class="badge" style="background:{color}">{badge}</span>
        <h3>{title}</h3>
        <p>{text}</p>
      </div>
    </div>""", unsafe_allow_html=True)


# Bannière d'accueil
def banner():
    st.markdown("""
    <div class="hero">
      <div class="hero-title">🤖 Machine Learning Prediction App</div>
      <div class="hero-sub">Devoir ML : Samba Hama TRAORE</div>
      <div class="hero-stats">
        <div><b>4</b><span>modèles déployés</span></div>
        <div><b>2</b><span>régressions</span></div>
        <div><b>2</b><span>classifications</span></div>
      </div>
    </div>""", unsafe_allow_html=True)


# Carte de résultat d'une prédiction simple
def result_card(label, value, color):
    st.markdown(f"""
    <div class="result" style="--accent:{color}">
      <span class="result-label">{label}</span>
      <span class="result-value">{value}</span>
    </div>""", unsafe_allow_html=True)


# Colonnes attendues pour la prédiction multiple
def cols_hint(cols):
    tags = ''.join(f'<span>{c}</span>' for c in cols)
    st.markdown(f'Colonnes attendues :<div class="cols-hint">{tags}</div>', unsafe_allow_html=True)


# Pied de page
def footer():
    st.markdown('<div class="footer">Devoir Machine Learning · Samba Hama TRAORE · '
                'Streamlit & scikit-learn</div>', unsafe_allow_html=True)


# Choisir l'interface simple ou multiple
def choose_mode(key):
    mode = st.segmented_control('Interface', ['Simple Prédiction', 'Prédiction multiple'],
                                default='Simple Prédiction', key=key)
    return mode if mode is not None else 'Simple Prédiction'


# Bouton de téléchargement d'un fichier csv d'exemple
def sample_button(file_name, key):
    path = os.path.join(BASE_DIR, 'samples', file_name)
    with open(path, 'rb') as f:
        st.download_button('Download a sample csv file', data=f.read(), file_name=file_name,
                           mime='text/csv', key=key)


# Charger les objets sauvegardés (une seule fois grâce au cache)
@st.cache_resource
def load_objects(folder):
    folder = os.path.join(BASE_DIR, folder)
    # importer les encodeurs
    encoders = joblib.load(os.path.join(folder, 'encoders.joblib'))
    # importer le best modèle
    best_model = joblib.load(os.path.join(folder, 'best_model.joblib'))
    # importer les valeurs uniques
    uniques = joblib.load(os.path.join(folder, 'uniques.joblib'))
    # importer le scaler
    scaler = joblib.load(os.path.join(folder, 'scaler.joblib'))
    return encoders, best_model, uniques, scaler


# Afficher les résultats de la prédiction multiple et le bouton de téléchargement
def show_csv_predictions(df_pred, key):
    st.write("Résultats des prédictions :")
    st.dataframe(df_pred)
    # Télécharger le fichier des prédictions
    st.download_button(label='Download a csv file',
                       data=df_pred.to_csv(index=False).encode('utf-8'),
                       file_name='predictions.csv',
                       mime='text/csv', key=key)


# =====================================================================
# 1. TAXI : prix d'un trajet de taxi (régression)
# =====================================================================
taxi_encoders, taxi_model, taxi_uniques, taxi_scaler = load_objects('taxi_reg')


# Fonction de prédiction
def taxi_Pred_func(Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                   Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes):
    # Encoder les valeurs des Time_of_Day, Day_of_Week, Traffic_Conditions et Weather
    Time_of_Day = taxi_encoders[0].transform([Time_of_Day])[0]
    Day_of_Week = taxi_encoders[1].transform([Day_of_Week])[0]
    Traffic_Conditions = taxi_encoders[2].transform([Traffic_Conditions])[0]
    Weather = taxi_encoders[3].transform([Weather])[0]
    # vecteur des valeurs numériques
    x_new = np.array([Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                      Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes], dtype=float)
    x_new = x_new.reshape(1, -1)  # convertir en un 2D array
    # normaliser
    x_new = taxi_scaler.transform(x_new)
    # Prédire
    y_pred = taxi_model.predict(x_new)
    # Arrondir
    return round(float(y_pred[0]), 2)


# Fonction de prédiction multiple
def taxi_Pred_func_csv(df):
    predictions = []
    # Boucle sur les lignes du dataframe
    for row in df.iloc[:, :10].values:
        y_pred = taxi_Pred_func(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9])
        predictions.append(y_pred)
    df['Trip_Price_Prediction'] = predictions
    return df


# =====================================================================
# 2. CAR CLASS : voiture Venant / D'occasion (classification)
# =====================================================================
# encoders / uniques : [Marque, Boite_Vitesse, Venant_Occasion, Quartier]
cc_encoders, cc_model, cc_uniques, cc_scaler = load_objects('car_class')


# Fonction de prédiction
def cc_Pred_func(Marque, Annee, Boite_Vitesse, Prix, Quartier):
    # Encoder les valeurs de Marque, Boite_Vitesse et Quartier
    Marque = cc_encoders[0].transform([Marque])[0]
    Boite_Vitesse = cc_encoders[1].transform([Boite_Vitesse])[0]
    Quartier = cc_encoders[3].transform([Quartier])[0]
    # vecteur des valeurs numériques (même ordre que dans le notebook)
    x_new = np.array([Marque, Annee, Boite_Vitesse, Prix, Quartier], dtype=float)
    x_new = x_new.reshape(1, -1)  # convertir en un 2D array
    # normaliser
    x_new = cc_scaler.transform(x_new)
    # Prédire
    y_pred = cc_model.predict(x_new)
    # Décoder la classe prédite
    return cc_encoders[2].inverse_transform([int(y_pred[0])])[0]


# Fonction de prédiction multiple (colonnes : Marque, Année, Boite_Vitesse, Adresse, Prix)
def cc_Pred_func_csv(df):
    predictions = []
    # Boucle sur les lignes du dataframe
    for row in df.iloc[:, :5].values:
        # séparer l'Adresse 'Quartier,Region' et garder le Quartier
        Quartier = str(row[3]).split(',')[0].strip()
        y_pred = cc_Pred_func(row[0], row[1], row[2], row[4], Quartier)
        predictions.append(y_pred)
    df['Venant_Occasion_Prediction'] = predictions
    return df


# =====================================================================
# 3. CAR REG : prix de revente d'une voiture (régression)
# =====================================================================
# encoders / uniques : [Fuel_Type, Seller_Type, Transmission]
cr_encoders, cr_model, cr_uniques, cr_scaler = load_objects('car_reg')


# Fonction de prédiction
def cr_Pred_func(Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age):
    # Encoder les valeurs de Fuel_Type, Seller_Type et Transmission
    Fuel_Type = cr_encoders[0].transform([Fuel_Type])[0]
    Seller_Type = cr_encoders[1].transform([Seller_Type])[0]
    Transmission = cr_encoders[2].transform([Transmission])[0]
    # vecteur des valeurs numériques
    x_new = np.array([Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age], dtype=float)
    x_new = x_new.reshape(1, -1)  # convertir en un 2D array
    # normaliser
    x_new = cr_scaler.transform(x_new)
    # Prédire
    y_pred = cr_model.predict(x_new)
    # Arrondir
    return round(float(y_pred[0]), 2)


# Fonction de prédiction multiple
def cr_Pred_func_csv(df):
    predictions = []
    # Boucle sur les lignes du dataframe
    for row in df.iloc[:, :6].values:
        y_pred = cr_Pred_func(row[0], row[1], row[2], row[3], row[4], row[5])
        predictions.append(y_pred)
    df['Selling_Price_Prediction'] = predictions
    return df


# =====================================================================
# 4. TELEPHONE CLASS : état d'un téléphone (classification)
# =====================================================================
# encoders / uniques : [adresse, marque, etat]
tel_encoders, tel_model, tel_uniques, tel_scaler = load_objects('telephone_class')


# Fonction de prédiction
def tel_Pred_func(prix, adresse, marque, dim_ecr, ram, stockage):
    # Encoder les valeurs de adresse et marque
    adresse = tel_encoders[0].transform([adresse])[0]
    marque = tel_encoders[1].transform([marque])[0]
    # vecteur des valeurs numériques
    x_new = np.array([prix, adresse, marque, dim_ecr, ram, stockage], dtype=float)
    x_new = x_new.reshape(1, -1)  # convertir en un 2D array
    # normaliser
    x_new = tel_scaler.transform(x_new)
    # Prédire
    y_pred = tel_model.predict(x_new)
    # Décoder la classe prédite
    return tel_encoders[2].inverse_transform([int(y_pred[0])])[0]


# Fonction de prédiction multiple
def tel_Pred_func_csv(df):
    predictions = []
    # Boucle sur les lignes du dataframe
    for row in df.iloc[:, :6].values:
        y_pred = tel_Pred_func(row[0], row[1], row[2], row[3], row[4], row[5])
        predictions.append(y_pred)
    df['etat_Prediction'] = predictions
    return df


# =====================================================================
# Interface
# =====================================================================
# Titre de l'application
banner()

# Créer les onglets (un par modèle)
tab_tel, tab_taxi, tab_cc, tab_cr = st.tabs(['📱 Telephone', '🚕 Taxi Price', '🚗 Car Class', '💰 Car Price'])

# ---------------- Onglet 2 : Taxi ----------------
with tab_taxi:
    card('taxi.svg', "Taxi Trip Price Prediction",
         """This machine learning model allows us to predict the price of a taxi trip
         from the trip distance, time of day, day of week, passenger count, traffic conditions, weather,
         base fare, per km rate, per minute rate and trip duration.""",
         'Régression', '#d4a106', 'linear-gradient(#1e2a4a, #3b4f7d)')
    mode = choose_mode('taxi_mode')

    # prédiction simple
    if mode == 'Simple Prédiction':
        with st.form('taxi_form_simple'):
            col1, col2 = st.columns(2)
            with col1:
                Trip_Distance_km = st.number_input('Trip Distance (km)', min_value=0.0, value=25.0, step=0.5)
                Time_of_Day = st.selectbox('Time of Day', options=taxi_uniques[0])
                Day_of_Week = st.selectbox('Day of Week', options=taxi_uniques[1])
                Passenger_Count = st.number_input('Passenger Count', min_value=1, max_value=4, value=2, step=1)
                Traffic_Conditions = st.selectbox('Traffic Conditions', options=taxi_uniques[2])
            with col2:
                Weather = st.selectbox('Weather', options=taxi_uniques[3])
                Base_Fare = st.number_input('Base Fare', min_value=0.0, value=3.5, step=0.1)
                Per_Km_Rate = st.number_input('Per Km Rate', min_value=0.0, value=1.2, step=0.05)
                Per_Minute_Rate = st.number_input('Per Minute Rate', min_value=0.0, value=0.3, step=0.01)
                Trip_Duration_Minutes = st.number_input('Trip Duration (minutes)', min_value=0.0, value=60.0, step=1.0)
            submit = st.form_submit_button('Predict')

        if submit:
            with st.spinner('Prédiction en cours...'):
                y_pred = taxi_Pred_func(Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                                        Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes)
            result_card("Trip Price", f"{y_pred} $", '#d4a106')
            st.toast('Prédiction terminée', icon='✅')

    # prédiction multiple
    else:
        cols_hint(['Trip_Distance_km', 'Time_of_Day', 'Day_of_Week', 'Passenger_Count', 'Traffic_Conditions', 'Weather', 'Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes'])
        sample_button('taxi_sample.csv', 'taxi_sample')
        file = st.file_uploader('Upload a csv file', type=['csv'], key='taxi_file')

        if file is not None:
            # Lire le fichier csv
            df = pd.read_csv(file)
            st.write("Aperçu des données chargées :")
            st.dataframe(df.head())

            if st.button('Predict', key='taxi_predict'):
                try:
                    with st.spinner('Prédictions en cours...'):
                        df_pred = taxi_Pred_func_csv(df)
                    show_csv_predictions(df_pred, 'taxi_download')
                    st.toast('Prédictions terminées', icon='✅')
                except Exception as e:
                    st.error(f"Le fichier ne respecte pas le format attendu : {e}")

# ---------------- Onglet 3 : Car Class ----------------
with tab_cc:
    card('car_class.svg', "Car Class Prediction (Venant / D'occasion)",
         """This machine learning model allows us to predict if a car sold in Dakar is "Venant"
         (imported) or "D'occasion" (used locally) from its brand, year, gearbox, district and price.""",
         'Classification', '#3b82c8', '#e8f1f8')
    mode = choose_mode('cc_mode')

    # prédiction simple
    if mode == 'Simple Prédiction':
        with st.form('cc_form_simple'):
            col1, col2 = st.columns(2)
            with col1:
                Marque = st.selectbox('Marque', options=cc_uniques[0])
                Annee = st.number_input('Année', min_value=1990, max_value=2026, value=2015, step=1)
                Boite_Vitesse = st.selectbox('Boite de vitesse', options=cc_uniques[1])
            with col2:
                Quartier = st.selectbox('Quartier (Dakar)', options=cc_uniques[3])
                Prix = st.number_input('Prix (FCFA)', min_value=0, value=7800000, step=100000)
            submit = st.form_submit_button('Predict')

        if submit:
            with st.spinner('Prédiction en cours...'):
                y_pred = cc_Pred_func(Marque, Annee, Boite_Vitesse, Prix, Quartier)
            result_card("Classe prédite", y_pred, '#3b82c8')
            st.toast('Prédiction terminée', icon='✅')

    # prédiction multiple
    else:
        cols_hint(['Marque', 'Année', 'Boite_Vitesse', 'Adresse (Quartier,Region)', 'Prix'])
        sample_button('car_class_sample.csv', 'cc_sample')
        file = st.file_uploader('Upload a csv file', type=['csv'], key='cc_file')

        if file is not None:
            # Lire le fichier csv
            df = pd.read_csv(file)
            st.write("Aperçu des données chargées :")
            st.dataframe(df.head())

            if st.button('Predict', key='cc_predict'):
                try:
                    with st.spinner('Prédictions en cours...'):
                        df_pred = cc_Pred_func_csv(df)
                    show_csv_predictions(df_pred, 'cc_download')
                    st.toast('Prédictions terminées', icon='✅')
                except Exception as e:
                    st.error(f"Le fichier ne respecte pas le format attendu : {e}")

# ---------------- Onglet 4 : Car Reg ----------------
with tab_cr:
    card('car_reg.svg', "Car Selling Price Prediction",
         """This machine learning model allows us to predict the selling price of a car
         from the kms driven, present price, fuel type, seller type, transmission and age.""",
         'Régression', '#2e8b57', '#eef6ee')
    mode = choose_mode('cr_mode')

    # prédiction simple
    if mode == 'Simple Prédiction':
        with st.form('cr_form_simple'):
            col1, col2 = st.columns(2)
            with col1:
                Kms_Driven = st.number_input('Kms Driven', min_value=0, value=70000, step=1000)
                Present_Price = st.number_input('Present Price', min_value=0.0, value=15.0, step=0.5)
                Fuel_Type = st.selectbox('Fuel Type', options=cr_uniques[0])
            with col2:
                Seller_Type = st.selectbox('Seller Type', options=cr_uniques[1])
                Transmission = st.selectbox('Transmission', options=cr_uniques[2])
                Age = st.number_input('Age (years)', min_value=0, max_value=40, value=8, step=1)
            submit = st.form_submit_button('Predict')

        if submit:
            with st.spinner('Prédiction en cours...'):
                y_pred = cr_Pred_func(Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age)
            result_card("Selling Price", y_pred, '#2e8b57')
            st.toast('Prédiction terminée', icon='✅')

    # prédiction multiple
    else:
        cols_hint(['Kms_Driven', 'Present_Price', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Age'])
        sample_button('car_reg_sample.csv', 'cr_sample')
        file = st.file_uploader('Upload a csv file', type=['csv'], key='cr_file')

        if file is not None:
            # Lire le fichier csv
            df = pd.read_csv(file)
            st.write("Aperçu des données chargées :")
            st.dataframe(df.head())

            if st.button('Predict', key='cr_predict'):
                try:
                    with st.spinner('Prédictions en cours...'):
                        df_pred = cr_Pred_func_csv(df)
                    show_csv_predictions(df_pred, 'cr_download')
                    st.toast('Prédictions terminées', icon='✅')
                except Exception as e:
                    st.error(f"Le fichier ne respecte pas le format attendu : {e}")

# ---------------- Onglet 1 : Telephone ----------------
with tab_tel:
    card('telephone.svg', "Telephone State Prediction",
         """This machine learning model allows us to predict the state of a phone
         (Neuf, D'occasion, Reconditionné or Venant) from its price, address, brand,
         screen size, RAM and storage.""",
         'Classification', '#7b5cd6', '#f2eef9')
    mode = choose_mode('tel_mode')

    # prédiction simple
    if mode == 'Simple Prédiction':
        with st.form('tel_form_simple'):
            col1, col2 = st.columns(2)
            with col1:
                prix = st.number_input('Prix (FCFA)', min_value=0, value=360000, step=5000)
                adresse = st.selectbox('Adresse', options=tel_uniques[0])
                marque = st.selectbox('Marque', options=tel_uniques[1])
            with col2:
                dim_ecr = st.number_input("Dimension de l'écran (pouces)", min_value=0.0, value=6.0, step=0.1)
                ram = st.number_input('RAM (Go)', min_value=0, value=12, step=1)
                stockage = st.number_input('Stockage (Go)', min_value=0, value=128, step=1)
            submit = st.form_submit_button('Predict')

        if submit:
            with st.spinner('Prédiction en cours...'):
                y_pred = tel_Pred_func(prix, adresse, marque, dim_ecr, ram, stockage)
            result_card("État prédit", y_pred, '#7c3aed')
            st.toast('Prédiction terminée', icon='✅')

    # prédiction multiple
    else:
        cols_hint(['prix', 'adresse', 'marque', 'dim_ecr', 'ram', 'stockage'])
        sample_button('telephone_sample.csv', 'tel_sample')
        file = st.file_uploader('Upload a csv file', type=['csv'], key='tel_file')

        if file is not None:
            # Lire le fichier csv
            df = pd.read_csv(file)
            st.write("Aperçu des données chargées :")
            st.dataframe(df.head())

            if st.button('Predict', key='tel_predict'):
                try:
                    with st.spinner('Prédictions en cours...'):
                        df_pred = tel_Pred_func_csv(df)
                    show_csv_predictions(df_pred, 'tel_download')
                    st.toast('Prédictions terminées', icon='✅')
                except Exception as e:
                    st.error(f"Le fichier ne respecte pas le format attendu : {e}")

# Pied de page
footer()
