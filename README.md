# 🎲 Data
Repositório para armazenar todos os componentes referentes as atividades de Machine Learning do projeto [pantanal.dev](<Pantanal.dev>) da [Comitiva Esperança](<https://github.com/comitivaesperanca>).

## 🆘 Como executar a plataforma de Machine Learning?
A plataforma é composta por 2 componentes principais:
- [Apache Airflow](<https://airflow.apache.org/>)
- [Jupyter](<https://jupyter.org/>)

Para executar a plataforma, é necessário ter o [Docker](<https://www.docker.com/>) instalado na máquina. <br>

Inicialmente, clone o repositório para sua máquina, seguindo os passos abaixos:
```bash
git clone
```
Em seguida, execute o comando:
```bash
docker-compose up -d --build
```

Após a execução do comando, a plataforma estará disponível para uso. <br>
- Para acessar o *Apache Airflow*, acesse o endereço: http://localhost:8080
- Para acessar o *Jupyter*, acesse o endereço: http://localhost:8888

## 💻 Tecnologias

### Docker
O Docker foi utilizado no projeto com o objetivo de facilitar a execução da plataforma de Machine Learning. <br>
Com o Docker, é possível executar a plataforma em qualquer sistema operacional, sem a necessidade de instalar as dependências necessárias para a execução do projeto. <br>

### Apache Airflow
Apache Airflow é uma plataforma para criar, programar e monitorar fluxos de trabalho. É uma ferramenta de código aberto para automatizar fluxos de trabalho complexos e gerenciar tarefas de forma programática.
No projeto, será utilizada para orquestrar o scrapping de dados e o treinamento dos modelos de Machine Learning. <br>
No Airflow, cada fluxo é chamado de Dag (Directed Acyclic Graph). Uma Dag é composta por tarefas, que são executadas em sequência ou em paralelo. Cada tarefa é executada por um operador, que é responsável por executar a tarefa. Os operadores são responsáveis por executar as tarefas, que podem ser qualquer coisa, desde um comando bash até um script Python.
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



### Jupyter 
Jupyter é uma aplicação web que permite criar e compartilhar documentos que contém código, equações, visualizações e texto explicativo. Os documentos Jupyter são chamados de notebooks. Os notebooks são executados em um servidor e são acessados por meio de um navegador web. O Jupyter Notebook é uma aplicação web de código aberto que permite criar e compartilhar documentos que contém código, equações, visualizações e texto explicativo. Os documentos Jupyter são chamados de notebooks. Os notebooks são executados em um servidor e são acessados por meio de um navegador web.

#### Como criar novos notebooks?
Basta abrir o Jupyter através do link http://localhost:8888 e criar um novo notebook. <br>
Para criar um novo notebook, basta clicar no botão **New** e selecionar a opção **Python 3**.

#### Como instalar novas bibliotecas?
Para instalar novas bibliotecas, basta adicionar no arquivo **requirements.txt** e executar o comando, em seu terminal:
```bash
docker-compose up -d --build
```