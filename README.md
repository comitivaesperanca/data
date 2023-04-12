# 🎲 Data
Repositório para armazenar todos os componentes referentes as atividades de Machine Learning do projeto [pantanal.dev](<Pantanal.dev>) da [Comitiva Esperança](<https://github.com/comitivaesperanca>).

## 🆘 Como executar a plataforma?
A plataforma é composta por 2 componentes principais:
- [Apache Airflow](<https://airflow.apache.org/>)
- [Jupyter](<https://jupyter.org/>)

Para executar a plataforma, é necessário ter o [Docker](<https://www.docker.com/>) instalado na máquina. <br>
Inicialmente, clone o repositório:
```bash
git clone
```
Em seguida, execute o comando:
```bash
docker-compose up
```
Após a execução do comando, acesse o [Jupyter](<http://localhost:8888/>), utilizando a senha `pantanal.dev`. <br>
Você pode também acessar o [Apache Airflow](https://airflow.apache.org/) através do endereço [http://localhost:8080/](<http://localhost:8080/>).

## 💻 Tecnologias

### Apache Airflow
Apache Airflow é uma plataforma para criar, programar e monitorar fluxos de trabalho. É uma ferramenta de código aberto para automatizar fluxos de trabalho complexos e gerenciar tarefas de forma programática.
No projeto, será utilizada para orquestrar o scrapping de dados e o treinamento dos modelos de Machine Learning.

### Jupyter 
Jupyter é uma aplicação web que permite criar e compartilhar documentos que contém código, equações, visualizações e texto explicativo. Os documentos Jupyter são chamados de notebooks. Os notebooks são executados em um servidor e são acessados por meio de um navegador web. O Jupyter Notebook é uma aplicação web de código aberto que permite criar e compartilhar documentos que contém código, equações, visualizações e texto explicativo. Os documentos Jupyter são chamados de notebooks. Os notebooks são executados em um servidor e são acessados por meio de um navegador web.

### Docker
Docker é uma plataforma de código aberto para desenvolvedores e administradores de sistemas para criar, testar e implementar aplicativos. Docker usa recursos do núcleo Linux, como controladores de cgroups e espaços de nomes, e da biblioteca libcontainer para permitir que isolamentos de processos sejam executados em nível de sistema operacional, ao mesmo tempo em que controla a agregação de recursos de forma isolada.
