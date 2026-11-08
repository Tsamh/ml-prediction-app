# 🤖 Machine Learning Prediction App

Application Streamlit qui déploie les 4 modèles du devoir de Machine Learning (Samba Hama TRAORE).

| Onglet | Problème | Cible | Meilleur modèle |
|---|---|---|---|
| 📱 Telephone | Classification | État du téléphone (Neuf, D'occasion, Réconditionné, Venant) | Random Forest (accuracy ≈ 0,86) |
| 🚕 Taxi Price | Régression | Prix d'un trajet de taxi | Random Forest (R² ≈ 0,92) |
| 🚗 Car Class | Classification | Voiture Venant / D'occasion (Dakar) | Random Forest (accuracy ≈ 0,82) |
| 💰 Car Price | Régression | Prix de revente d'une voiture | Random Forest (R² ≈ 0,87) |

Chaque onglet propose :
- **Simple Prédiction** : un formulaire pour prédire une seule observation ;
- **Prédiction multiple** : l'upload d'un fichier CSV (un exemple est téléchargeable dans l'app, voir aussi le dossier `samples/`), avec téléchargement des prédictions.

## Structure

```
app.py               # application Streamlit
style.css            # style de l'interface
.streamlit/config.toml  # thème
requirements.txt
images/              # illustrations des cartes
samples/             # fichiers CSV d'exemple pour la prédiction multiple
telephone_class/  taxi_reg/  car_class/  car_reg/
    best_model.joblib  encoders.joblib  scaler.joblib  uniques.joblib
```

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Déployer sur Streamlit Community Cloud

1. Aller sur [share.streamlit.io](https://share.streamlit.io) → **Create app** → **Deploy a public app from GitHub**.
2. Choisir ce dépôt, la branche `main` et le fichier principal `app.py`.
3. Cliquer sur **Deploy**.

## Licence

Distribué sous licence [Apache 2.0](LICENSE).
