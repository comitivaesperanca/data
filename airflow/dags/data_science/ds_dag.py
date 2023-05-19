from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
import logging
import os
import sys

# Import tasks folder
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.append('/opt/airflow/dags')


dag_id = 'Comitiva_Esperanca-TreinamentoDoModelo'

default_args = {
    'owner': 'Comitiva Esperança',
    "email": ["maycon.mota@ufms.br"],
    'description': 'DAG para aprendizado sobre Airflow',
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
    concurrency=16,
    schedule_interval="0 0/6 * * *",
    catchup=False
) as dag:

    begin_extract = DummyOperator(
        task_id="begin_extract",
        dag=dag
    )

    begin_extract

