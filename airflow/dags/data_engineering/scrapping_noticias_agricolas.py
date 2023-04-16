from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from common.tasks import export_noticias_agricolas
import logging
import os
import sys


dag_id = 'Comitiva_Esperanca-NoticiasAgricolas-Scrapping'

default_args = {
    'owner': 'Comitiva Esperança',
    "email": ["maycon.mota@ufms.br"],
    'description': 'DAG para realizar scrapping do site Noticas Agricolas',
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

    for i in range(1, 3000):
        task_extract_noticias_agricolas = PythonOperator(
            task_id=f"export_noticias_agricola_page_{i}",
            python_callable= export_noticias_agricolas.export_noticias_agricolas,
            op_kwargs={"page_number": i},
            dag=dag
        )
        begin_extract >> task_extract_noticias_agricolas

