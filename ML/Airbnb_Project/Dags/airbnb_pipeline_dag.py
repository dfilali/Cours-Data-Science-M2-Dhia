from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
import os

############################
#Ingestion des données
############################

def ingest_and_merge_data():
    RAW_DATA_PATH = "./data/raw"
    PROCESSED_DATA_PATH = "./data/processed"
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    train = pd.read_csv(f"{RAW_DATA_PATH}/airbnb_train.csv")
    test = pd.read_csv(f"{RAW_DATA_PATH}/airbnb_test.csv")
    preds = pd.read_csv(f"{RAW_DATA_PATH}/prediction_example.csv", index_col=0)
    preds.reset_index(inplace=True)
    preds.columns = ['id', 'log_price_pred']

    test_with_preds = test.copy()
    test_with_preds['id'] = test_with_preds.index.astype(int)
    df_final = test_with_preds.merge(preds, on='id', how='left')

    df_final['predicted_price'] = np.exp(df_final['log_price_pred'])

    df_final.to_parquet(f"{PROCESSED_DATA_PATH}/data_ingested.parquet", index=False)
    print("✅ Données ingérées et sauvegardées.")

############################
#Prétraitement
############################

def preprocess_data():
    PROCESSED_DATA_PATH = "./data/processed"
    df = pd.read_parquet(f"{PROCESSED_DATA_PATH}/data_ingested.parquet")

    df['predicted_price'] = df['predicted_price'].fillna(df['predicted_price'].median())
    df = df.dropna(subset=['latitude', 'longitude'])

    if 'instant_bookable' in df.columns:
        df['instant_bookable'] = df['instant_bookable'].map({'t': 1, 'f': 0})

    df.to_parquet(f"{PROCESSED_DATA_PATH}/data_preprocessed.parquet", index=False)
    print("✅ Données prétraitées et sauvegardées.")

# Définition du DAG
default_args = {
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

with DAG(
    dag_id='airbnb_ingestion_dag',
    default_args=default_args,
    description='DAG pour ingérer les données Airbnb et sauvegarder les résultats fusionnés',
    schedule_interval="@hourly",  # Remplacer par `schedule='@hourly'` si tu veux éviter le warning
    catchup=False,
    tags=['airbnb', 'ingestion'],
) as dag:

    ingestion_task = PythonOperator(
        task_id='ingest_and_merge_data',
        python_callable=ingest_and_merge_data
    )

    preprocessing_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )

    ingestion_task >> preprocessing_task
