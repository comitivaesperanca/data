# 🎲 Data

Repositório para armazenar todos os componentes referentes as atividades de Machine Learning do projeto [pantanal.dev](<Pantanal.dev>) da [Comitiva Esperança](<https://github.com/comitivaesperanca>).

## 🆘 Como executar a plataforma de Machine Learning?

A plataforma é composta por 2 componentes principais:

- [Apache Airflow](<https://airflow.apache.org/>)
    [Guia de como usar](./airflow/README.md)
- [Jupyter](<https://jupyter.org/>)
    [Guia de como usar](./jupyter/README.md)

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

- Para acessar o *Apache Airflow*, acesse o endereço: <http://localhost:8080>
- Para acessar o *Jupyter*, acesse o endereço: <http://localhost:8888>
