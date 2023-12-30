import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Charger les données
data = pd.read_csv('Expresso_churn_dataset (1).csv')

# Exploration de base
st.write("Affichage des premières lignes du DataFrame :")
st.write(data.head())

st.write("Informations générales sur le DataFrame :")
st.write(data.info())

st.write("Description statistique du DataFrame :")
st.write(data.describe())

# Gérer les valeurs manquantes
data = data.dropna()

# Supprimer les doublons
data = data.drop_duplicates()
data = data.drop('MRG', axis=1)
data = data.drop('user_id', axis=1)

# Encodage des variables catégorielles
data_encoded = pd.get_dummies(data, columns=['REGION', 'TENURE', 'TOP_PACK'])


features = data_encoded.drop('CHURN', axis=1)
target = data_encoded['CHURN']

# Divisez les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialisez un modèle,
model = RandomForestClassifier()

# Entraînez le modèle
model.fit(X_train, y_train)

# Faites des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluez les performances du modèle
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy}")

# Fonction pour prédire une nouvelle observation
def predict(input_data):

    input_data_encoded = pd.get_dummies(pd.DataFrame(input_data), columns=['REGION', 'TENURE',  'TOP_PACK'])
    prediction = model.predict(input_data_encoded)[0]
    return prediction

# Titre de l'application
st.title("Votre Application de Prédiction")

# Ajouter des champs de saisie pour les fonctionnalités
feature1 = st.slider("Feature 1", min_value=0, max_value=100)
feature2 = st.slider("Feature 2", min_value=0, max_value=100)
feature_cat = st.selectbox("Feature Catégorielle", options=["num1", "num2", "num3"])

# Bouton de validation
if st.button("Faire une prédiction"):
    # Créer un dictionnaire avec les valeurs d'entrée
    input_data = {'feature1': feature1, 'feature2': feature2, 'REGION_FATICK': 0, 'REGION_DAKAR': 0, 'REGION_LOUGA': 0, 'REGION_SAINT-LOUIS': 0, 'TENURE_D-12': 0, 'TENURE_D-6': 0, 'TENURE_D-7': 0, 'TENURE_D-8': 0, 'TENURE_D-9': 0, 'TENURE_D-11': 0, 'TENURE_D-10': 0, 'TENURE_D-5': 0, 'TENURE_D-4': 0, 'TENURE_D-3': 0, 'TENURE_D-2': 0, 'TENURE_D-1': 0, 'MRG_REGULAR': 0, 'MRG_YES': 0, 'MRG_SUSPENDED': 0, 'TOP_PACK_Data:2997': 0, 'TOP_PACK_ONNET_OTHER': 0, 'TOP_PACK_ON_NET': 0, 'TOP_PACK_OFF_NET': 0, 'TOP_PACK_Data:799': 0, 'TOP_PACK_Internet:490': 0, 'TOP_PACK_SUPERMAGIK_100F': 0, 'TOP_PACK_Jokko_Daily': 0, 'TOP_PACK_SUPERMAGIK_1000F': 0, 'TOP_PACK_Jokko_Weekly': 0, 'TOP_PACK_NEON_400F': 0, 'TOP_PACK_NEON_1000F': 0, 'TOP_PACK_SUPERMAGIK_5000F': 0, 'TOP_PACK_Jokko_Monthly': 0}

    if feature_cat == "num1":
        input_data['REGION_FATICK'] = 1
    elif feature_cat == "num2":
        input_data['REGION_DAKAR'] = 1
    elif feature_cat == "num3":
        input_data['REGION_LOUGA'] = 1
    else:
        input_data['REGION_SAINT-LOUIS'] = 1

    # Faire une prédiction
    prediction = predict(input_data)

    # Afficher la prédiction
    st.success(f"La prédiction est: {prediction}")
