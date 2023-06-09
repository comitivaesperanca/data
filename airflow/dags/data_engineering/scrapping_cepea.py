from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow import Dataset

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from common.tasks import export_cepea, classification_cepea
import logging


dag_id = 'Comitiva_Esperanca-CEPEA-Scrapping'

default_args = {
    'owner': 'Comitiva Esperança',
    "email": ["maycon.mota@ufms.br"],
    'description': 'DAG para realizar scrapping do informativo do CEPEA',
    "email_on_failure": True,
    "email_on_retry": False,
    'retries': 4,
    'catchup': False
}


with DAG(
    start_date=datetime(2021, 12, 18),
    dag_id=dag_id,
    default_args=default_args,
    max_active_runs=1,
    schedule_interval='0 5 * * *',
    concurrency=12,
    catchup=False
) as dag:

    begin_extract = DummyOperator(
        task_id="begin_extract",
        dag=dag
    )
    end_extract = DummyOperator(
        task_id="end_extract",
        dag=dag,
        outlets=[Dataset('data/ingestion/RAW/noticias_cepea_incremental.csv')]
            )

    for i in range(3):
        task_extract_cepea = PythonOperator(
            task_id=f"export_noticias_cepea_{i}",
            python_callable= export_cepea.export_noticias_cepea,
            op_kwargs={"page_number": i},
            dag=dag
        )
        begin_extract >> task_extract_cepea >> end_extract

with DAG(
    schedule=[Dataset("data/ingestion/RAW/noticias_cepea_incremental.csv")],
    dag_id="Classificacao_Publicacao_Noticias",
    start_date=datetime(2021, 12, 18),
    default_args=default_args,
    max_active_runs=1,
    concurrency=12,
    catchup=False
):
    classificacao_publicacao = PythonOperator(
        task_id="classificacao_publicacao",
        python_callable=classification_cepea.classification_publish,
        op_kwargs={"model": "rede_neural"},
    )

    begin_classificacao = DummyOperator(
        task_id="begin_classificacao",
    )
    end_classificacao = DummyOperator(
        task_id="end_classificacao",
    )

    begin_classificacao >> classificacao_publicacao >> end_classificacao