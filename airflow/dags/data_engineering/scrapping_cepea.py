from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from common.tasks import export_cepea
import logging
import os
import sys

# Import tasks folder
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.append('/opt/airflow/dags')


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
    concurrency=12,
    schedule_interval="* * * * *",
    catchup=False
) as dag:

    begin_extract = DummyOperator(
        task_id="begin_extract",
        dag=dag
    )

    for i in range(815):
        task_extract_cepea = PythonOperator(
            task_id=f"export_noticias_cepea_{i}",
            python_callable= export_cepea.export_noticias_cepea,
            op_kwargs={"page_number": i},
            dag=dag
        )
        begin_extract >> task_extract_cepea

