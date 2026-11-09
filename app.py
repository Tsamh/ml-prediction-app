# Application Streamlit de déploiement des 4 modèles du devoir :
#   - Taxi_Reg        : prix d'un trajet de taxi (régression)
#   - Car_Class       : voiture Venant / D'occasion (classification)
#   - Car_Reg         : prix de revente d'une voiture (régression)
#   - Telephone_Class : état d'un téléphone (classification)
# Interface en français par défaut, avec option anglais.
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


# =====================================================================
# Langue : français par défaut, anglais en option
# =====================================================================
TEXTS = {
    'fr': {
        'hero_sub': "Devoir ML : Samba Hama TRAORE",
        'stat_models': "modèles déployés", 'stat_reg': "régressions", 'stat_clf': "classifications",
        'tab_tel': "📱 Téléphone", 'tab_taxi': "🚕 Prix Taxi", 'tab_cc': "🚗 Classe Voiture", 'tab_cr': "💰 Prix Voiture",
        'mode_label': "Interface", 'simple': "Prédiction simple", 'multiple': "Prédiction multiple",
        'badge_reg': "Régression", 'badge_clf': "Classification",
        'cols_expected': "Colonnes attendues :",
        'sample_btn': "Télécharger un fichier CSV d'exemple",
        'upload': "Charger un fichier CSV",
        'preview': "Aperçu des données chargées :",
        'predict': "Prédire",
        'results': "Résultats des prédictions :",
        'download': "Télécharger les prédictions (CSV)",
        'spinner': "Prédiction en cours...", 'spinner_multi': "Prédictions en cours...",
        'toast': "Prédiction terminée", 'toast_multi': "Prédictions terminées",
        'error': "Le fichier ne respecte pas le format attendu : ",
        'footer': "Devoir Machine Learning · Samba Hama TRAORE · Streamlit & scikit-learn",
        # Taxi
        'taxi_title': "Prédiction du prix d'un trajet de taxi",
        'taxi_text': """Ce modèle de machine learning prédit le prix d'un trajet de taxi à partir de la distance,
            du moment de la journée, du type de jour, du nombre de passagers, des conditions de trafic, de la météo,
            du tarif de base, du tarif par km, du tarif par minute et de la durée du trajet.""",
        'taxi_distance': "Distance du trajet (km)", 'taxi_time': "Moment de la journée", 'taxi_day': "Type de jour",
        'taxi_passengers': "Nombre de passagers", 'taxi_traffic': "Conditions de trafic", 'taxi_weather': "Météo",
        'taxi_base': "Tarif de base", 'taxi_km': "Tarif par km", 'taxi_min': "Tarif par minute",
        'taxi_duration': "Durée du trajet (minutes)", 'taxi_result': "Prix du trajet",
        # Car Class
        'cc_title': "Classification d'une voiture (Venant / D'occasion)",
        'cc_text': """Ce modèle de machine learning prédit si une voiture vendue à Dakar est « Venant » (importée)
            ou « D'occasion » (déjà utilisée localement) à partir de sa marque, son année, sa boîte de vitesse,
            son quartier et son prix.""",
        'cc_brand': "Marque", 'cc_year': "Année", 'cc_gearbox': "Boîte de vitesse",
        'cc_district': "Quartier (Dakar)", 'cc_price': "Prix (FCFA)", 'cc_result': "Classe prédite",
        # Car Reg
        'cr_title': "Prédiction du prix de revente d'une voiture",
        'cr_text': """Ce modèle de machine learning prédit le prix de revente d'une voiture à partir de son kilométrage,
            de son prix actuel à l'état neuf, du type de carburant, du type de vendeur, de la transmission et de son âge.""",
        'cr_kms': "Kilométrage (km)", 'cr_present': "Prix actuel du neuf", 'cr_fuel': "Carburant",
        'cr_seller': "Type de vendeur", 'cr_trans': "Transmission", 'cr_age': "Âge (années)",
        'cr_result': "Prix de revente",
        # Telephone
        'tel_title': "Prédiction de l'état d'un téléphone",
        'tel_text': """Ce modèle de machine learning prédit l'état d'un téléphone (Neuf, D'occasion, Réconditionné
            ou Venant) à partir de son prix, son adresse, sa marque, la taille de l'écran, la RAM et le stockage.""",
        'tel_price': "Prix (FCFA)", 'tel_address': "Adresse", 'tel_brand': "Marque",
        'tel_screen': "Taille de l'écran (pouces)", 'tel_ram': "RAM (Go)", 'tel_storage': "Stockage (Go)",
        'tel_result': "État prédit",
    },
    'en': {
        'hero_sub': "ML assignment: Samba Hama TRAORE",
        'stat_models': "deployed models", 'stat_reg': "regressions", 'stat_clf': "classifications",
        'tab_tel': "📱 Phone", 'tab_taxi': "🚕 Taxi Price", 'tab_cc': "🚗 Car Class", 'tab_cr': "💰 Car Price",
        'mode_label': "Interface", 'simple': "Single prediction", 'multiple': "Multiple prediction",
        'badge_reg': "Regression", 'badge_clf': "Classification",
        'cols_expected': "Expected columns:",
        'sample_btn': "Download a sample CSV file",
        'upload': "Upload a CSV file",
        'preview': "Preview of the uploaded data:",
        'predict': "Predict",
        'results': "Prediction results:",
        'download': "Download the predictions (CSV)",
        'spinner': "Predicting...", 'spinner_multi': "Running predictions...",
        'toast': "Prediction done", 'toast_multi': "Predictions done",
        'error': "The file does not match the expected format: ",
        'footer': "Machine Learning assignment · Samba Hama TRAORE · Streamlit & scikit-learn",
        # Taxi
        'taxi_title': "Taxi Trip Price Prediction",
        'taxi_text': """This machine learning model predicts the price of a taxi trip from the trip distance,
            time of day, day of week, passenger count, traffic conditions, weather, base fare, per km rate,
            per minute rate and trip duration.""",
        'taxi_distance': "Trip Distance (km)", 'taxi_time': "Time of Day", 'taxi_day': "Day of Week",
        'taxi_passengers': "Passenger Count", 'taxi_traffic': "Traffic Conditions", 'taxi_weather': "Weather",
        'taxi_base': "Base Fare", 'taxi_km': "Per Km Rate", 'taxi_min': "Per Minute Rate",
        'taxi_duration': "Trip Duration (minutes)", 'taxi_result': "Trip Price",
        # Car Class
        'cc_title': "Car Class Prediction (Imported / Used)",
        'cc_text': """This machine learning model predicts whether a car sold in Dakar is "Venant" (imported)
            or "D'occasion" (already used locally) from its brand, year, gearbox, district and price.""",
        'cc_brand': "Brand", 'cc_year': "Year", 'cc_gearbox': "Gearbox",
        'cc_district': "District (Dakar)", 'cc_price': "Price (FCFA)", 'cc_result': "Predicted class",
        # Car Reg
        'cr_title': "Car Selling Price Prediction",
        'cr_text': """This machine learning model predicts the selling price of a car from the kms driven,
            its present price when new, fuel type, seller type, transmission and age.""",
        'cr_kms': "Kms Driven", 'cr_present': "Present Price (new)", 'cr_fuel': "Fuel Type",
        'cr_seller': "Seller Type", 'cr_trans': "Transmission", 'cr_age': "Age (years)",
        'cr_result': "Selling Price",
        # Telephone
        'tel_title': "Phone State Prediction",
        'tel_text': """This machine learning model predicts the state of a phone (New, Used, Refurbished
            or Imported) from its price, address, brand, screen size, RAM and storage.""",
        'tel_price': "Price (FCFA)", 'tel_address': "Address", 'tel_brand': "Brand",
        'tel_screen': "Screen size (inches)", 'tel_ram': "RAM (GB)", 'tel_storage': "Storage (GB)",
        'tel_result': "Predicted state",
    },
}

# Traduction des classes prédites (affichage en anglais)
CLASSES_EN = {"Venant": "Imported (Venant)", "D'occasion": "Used (D'occasion)",
              "Neuf": "New (Neuf)", "Réconditionné": "Refurbished (Réconditionné)"}

# Sélecteurs de langue et de thème (en haut à droite)
_, col_lang, col_theme = st.columns([5.6, 1.4, 1], vertical_alignment='center')
with col_lang:
    st.segmented_control('Langue / Language', ['fr', 'en'], default='fr', key='lang',
                         format_func=lambda l: {'fr': 'Français', 'en': 'English'}[l],
                         label_visibility='collapsed')
with col_theme:
    st.segmented_control('Thème / Theme', ['light', 'dark'], default='light', key='theme',
                         format_func=lambda m: {'light': '☀️', 'dark': '🌙'}[m],
                         label_visibility='collapsed')
LANG = st.session_state.get('lang') or 'fr'

# Mode sombre : on ajoute le fichier dark.css par-dessus style.css
if st.session_state.get('theme') == 'dark':
    load_css('dark.css')


# Texte dans la langue choisie
def t(key):
    return TEXTS[LANG][key]


# Classe prédite dans la langue choisie
def t_class(label):
    return CLASSES_EN.get(label, label) if LANG == 'en' else label


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
    st.markdown(f"""
    <div class="hero">
      <div class="hero-title">🤖 Machine Learning Prediction App</div>
      <div class="hero-sub">{t('hero_sub')}</div>
      <div class="hero-stats">
        <div><b>4</b><span>{t('stat_models')}</span></div>
        <div><b>2</b><span>{t('stat_reg')}</span></div>
        <div><b>2</b><span>{t('stat_clf')}</span></div>
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
    st.markdown(f'{t("cols_expected")}<div class="cols-hint">{tags}</div>', unsafe_allow_html=True)


# Pied de page
def footer():
    st.markdown(f'<div class="footer">{t("footer")}</div>', unsafe_allow_html=True)


# Choisir l'interface simple ou multiple
def choose_mode(key):
    mode = st.segmented_control(t('mode_label'), ['simple', 'multiple'], default='simple',
                                format_func=t, key=key)
    return mode if mode is not None else 'simple'


# Bouton de téléchargement d'un fichier csv d'exemple
def sample_button(file_name, key):
    path = os.path.join(BASE_DIR, 'samples', file_name)
    with open(path, 'rb') as f:
        st.download_button(t('sample_btn'), data=f.read(), file_name=file_name,
                           mime='text/csv', key=key)


# Afficher les résultats de la prédiction multiple et le bouton de téléchargement
def show_csv_predictions(df_pred, key):
    st.write(t('results'))
    st.dataframe(df_pred)
    # Télécharger le fichier des prédictions
    st.download_button(label=t('download'),
                       data=df_pred.to_csv(index=False).encode('utf-8'),
                       file_name='predictions.csv',
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
tab_tel, tab_taxi, tab_cc, tab_cr = st.tabs([t('tab_tel'), t('tab_taxi'), t('tab_cc'), t('tab_cr')])

# ---------------- Onglet 1 : Telephone ----------------
with tab_tel:
    card('telephone.svg', t('tel_title'), t('tel_text'), t('badge_clf'), '#7b5cd6', '#f2eef9')
    mode = choose_mode('tel_mode')

    # prédiction simple
    if mode == 'simple':
        with st.form('tel_form_simple'):
            col1, col2 = st.columns(2)
            with col1:
                prix = st.number_input(t('tel_price'), min_value=0, value=360000, step=5000)
                adresse = st.selectbox(t('tel_address'), options=tel_uniques[0])
                marque = st.selectbox(t('tel_brand'), options=tel_uniques[1])
            with col2:
                dim_ecr = st.number_input(t('tel_screen'), min_value=0.0, value=6.0, step=0.1)
                ram = st.number_input(t('tel_ram'), min_value=0, value=12, step=1)
                stockage = st.number_input(t('tel_storage'), min_value=0, value=128, step=1)
            submit = st.form_submit_button(t('predict'))

        if submit:
            with st.spinner(t('spinner')):
                y_pred = tel_Pred_func(prix, adresse, marque, dim_ecr, ram, stockage)
            result_card(t('tel_result'), t_class(y_pred), '#7c3aed')
            st.toast(t('toast'), icon='✅')

    # prédiction multiple
    else:
        cols_hint(['prix', 'adresse', 'marque', 'dim_ecr', 'ram', 'stockage'])
        sample_button('telephone_sample.csv', 'tel_sample')
        file = st.file_uploader(t('upload'), type=['csv'], key='tel_file')

        if file is not None:
            # Lire le fichier csv
            df = pd.read_csv(file)
            st.write(t('preview'))
            st.dataframe(df.head())

            if st.button(t('predict'), key='tel_predict'):
                try:
                    with st.spinner(t('spinner_multi')):
                        df_pred = tel_Pred_func_csv(df)
                    show_csv_predictions(df_pred, 'tel_download')
                    st.toast(t('toast_multi'), icon='✅')
                except Exception as e:
                    st.error(f"{t('error')}{e}")

# ---------------- Onglet 2 : Taxi ----------------
with tab_taxi:
    card('taxi.svg', t('taxi_title'), t('taxi_text'), t('badge_reg'), '#d4a106',
         'linear-gradient(#1e2a4a, #3b4f7d)')
    mode = choose_mode('taxi_mode')

    # prédiction simple
    if mode == 'simple':
        with st.form('taxi_form_simple'):
            col1, col2 = st.columns(2)
            with col1:
                Trip_Distance_km = st.number_input(t('taxi_distance'), min_value=0.0, value=25.0, step=0.5)
                Time_of_Day = st.selectbox(t('taxi_time'), options=taxi_uniques[0])
                Day_of_Week = st.selectbox(t('taxi_day'), options=taxi_uniques[1])
                Passenger_Count = st.number_input(t('taxi_passengers'), min_value=1, max_value=4, value=2, step=1)
                Traffic_Conditions = st.selectbox(t('taxi_traffic'), options=taxi_uniques[2])
            with col2:
                Weather = st.selectbox(t('taxi_weather'), options=taxi_uniques[3])
                Base_Fare = st.number_input(t('taxi_base'), min_value=0.0, value=3.5, step=0.1)
                Per_Km_Rate = st.number_input(t('taxi_km'), min_value=0.0, value=1.2, step=0.05)
                Per_Minute_Rate = st.number_input(t('taxi_min'), min_value=0.0, value=0.3, step=0.01)
                Trip_Duration_Minutes = st.number_input(t('taxi_duration'), min_value=0.0, value=60.0, step=1.0)
            submit = st.form_submit_button(t('predict'))

        if submit:
            with st.spinner(t('spinner')):
                y_pred = taxi_Pred_func(Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                                        Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes)
            result_card(t('taxi_result'), f"{y_pred} $", '#d4a106')
            st.toast(t('toast'), icon='✅')

    # prédiction multiple
    else:
        cols_hint(['Trip_Distance_km', 'Time_of_Day', 'Day_of_Week', 'Passenger_Count', 'Traffic_Conditions',
                   'Weather', 'Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes'])
        sample_button('taxi_sample.csv', 'taxi_sample')
        file = st.file_uploader(t('upload'), type=['csv'], key='taxi_file')

        if file is not None:
            # Lire le fichier csv
            df = pd.read_csv(file)
            st.write(t('preview'))
            st.dataframe(df.head())

            if st.button(t('predict'), key='taxi_predict'):
                try:
                    with st.spinner(t('spinner_multi')):
                        df_pred = taxi_Pred_func_csv(df)
                    show_csv_predictions(df_pred, 'taxi_download')
                    st.toast(t('toast_multi'), icon='✅')
                except Exception as e:
                    st.error(f"{t('error')}{e}")

# ---------------- Onglet 3 : Car Class ----------------
with tab_cc:
    card('car_class.svg', t('cc_title'), t('cc_text'), t('badge_clf'), '#3b82c8', '#e8f1f8')
    mode = choose_mode('cc_mode')

    # prédiction simple
    if mode == 'simple':
        with st.form('cc_form_simple'):
            col1, col2 = st.columns(2)
            with col1:
                Marque = st.selectbox(t('cc_brand'), options=cc_uniques[0])
                Annee = st.number_input(t('cc_year'), min_value=1990, max_value=2026, value=2015, step=1)
                Boite_Vitesse = st.selectbox(t('cc_gearbox'), options=cc_uniques[1])
            with col2:
                Quartier = st.selectbox(t('cc_district'), options=cc_uniques[3])
                Prix = st.number_input(t('cc_price'), min_value=0, value=7800000, step=100000)
            submit = st.form_submit_button(t('predict'))

        if submit:
            with st.spinner(t('spinner')):
                y_pred = cc_Pred_func(Marque, Annee, Boite_Vitesse, Prix, Quartier)
            result_card(t('cc_result'), t_class(y_pred), '#3b82c8')
            st.toast(t('toast'), icon='✅')

    # prédiction multiple
    else:
        cols_hint(['Marque', 'Année', 'Boite_Vitesse', 'Adresse (Quartier,Region)', 'Prix'])
        sample_button('car_class_sample.csv', 'cc_sample')
        file = st.file_uploader(t('upload'), type=['csv'], key='cc_file')

        if file is not None:
            # Lire le fichier csv
            df = pd.read_csv(file)
            st.write(t('preview'))
            st.dataframe(df.head())

            if st.button(t('predict'), key='cc_predict'):
                try:
                    with st.spinner(t('spinner_multi')):
                        df_pred = cc_Pred_func_csv(df)
                    show_csv_predictions(df_pred, 'cc_download')
                    st.toast(t('toast_multi'), icon='✅')
                except Exception as e:
                    st.error(f"{t('error')}{e}")

# ---------------- Onglet 4 : Car Reg ----------------
with tab_cr:
    card('car_reg.svg', t('cr_title'), t('cr_text'), t('badge_reg'), '#2e8b57', '#eef6ee')
    mode = choose_mode('cr_mode')

    # prédiction simple
    if mode == 'simple':
        with st.form('cr_form_simple'):
            col1, col2 = st.columns(2)
            with col1:
                Kms_Driven = st.number_input(t('cr_kms'), min_value=0, value=70000, step=1000)
                Present_Price = st.number_input(t('cr_present'), min_value=0.0, value=15.0, step=0.5)
                Fuel_Type = st.selectbox(t('cr_fuel'), options=cr_uniques[0])
            with col2:
                Seller_Type = st.selectbox(t('cr_seller'), options=cr_uniques[1])
                Transmission = st.selectbox(t('cr_trans'), options=cr_uniques[2])
                Age = st.number_input(t('cr_age'), min_value=0, max_value=40, value=8, step=1)
            submit = st.form_submit_button(t('predict'))

        if submit:
            with st.spinner(t('spinner')):
                y_pred = cr_Pred_func(Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age)
            result_card(t('cr_result'), y_pred, '#2e8b57')
            st.toast(t('toast'), icon='✅')

    # prédiction multiple
    else:
        cols_hint(['Kms_Driven', 'Present_Price', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Age'])
        sample_button('car_reg_sample.csv', 'cr_sample')
        file = st.file_uploader(t('upload'), type=['csv'], key='cr_file')

        if file is not None:
            # Lire le fichier csv
            df = pd.read_csv(file)
            st.write(t('preview'))
            st.dataframe(df.head())

            if st.button(t('predict'), key='cr_predict'):
                try:
                    with st.spinner(t('spinner_multi')):
                        df_pred = cr_Pred_func_csv(df)
                    show_csv_predictions(df_pred, 'cr_download')
                    st.toast(t('toast_multi'), icon='✅')
                except Exception as e:
                    st.error(f"{t('error')}{e}")

# Pied de page
footer()
