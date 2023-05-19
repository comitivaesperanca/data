### Apache Airflow
Apache Airflow é uma plataforma para criar, programar e monitorar fluxos de trabalho. É uma ferramenta de código aberto para automatizar fluxos de trabalho complexos e gerenciar tarefas de forma programática. <br>
No projeto, será utilizada para orquestrar o scrapping de dados e o treinamento dos modelos de Machine Learning. <br>
<br>
No Airflow, cada fluxo é chamado de Dag (Directed Acyclic Graph). Uma Dag é composta por tarefas, que são executadas em sequência ou em paralelo. Cada tarefa é executada por um operador, que é responsável por executar a tarefa. <br>
Os operadores são responsáveis por executar as tarefas, que podem ser qualquer coisa, desde um comando bash até um script Python. <br>

#### Como criar novas tarefas?
Na pasta Dags do Airflow, há duas subpastas: <br>
- **data_engineering**: contém os arquivos para scrapping de textos e dados 
- **data_science**: contém os arquivos para treinamento dos modelos de Machine Learning

Dentro de cada uma das pastas, há um arquivo chamado **dag.py**. Esse arquivo é responsável por criar a Dag e adicionar as tarefas. <br>
Para criar uma nova tarefa, basta criar um arquivo em Python (Ex. task_example.py) no diretório **common/tasks**, como por exemplo:
```python
def task_example():
    print("Hello World!")
```
Em seguida, importar o arquivo no arquivo **dag.py** e adicionar a tarefa na Dag:
```python

from common.tasks.task_example import task_example
task_example = PythonOperator(
    task_id='task_example',
    python_callable=task_example,
    dag=dag
)
```
Em seguida, adicionar a tarefa na sequência de execução da Dag:
```python
task_example >> task_2
```

A partir disso, sua task será exibida na lista de Dags do Airflow.