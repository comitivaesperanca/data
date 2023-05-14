from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
from lxml import etree
import csv 

def export_noticias_cepea(page_number: int):


    # Define uma função para obter os links de uma página
    def get_links(url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }

        response = requests.get(url, headers=headers)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')

        matching_elements = soup.find_all('a', class_='box-texto-relacionado')
        return [element.attrs['href'] for element in matching_elements]

    # Define a lista de URLs a serem requisitadas

    base_url = 'https://www.cepea.esalq.usp.br/br/diarias-de-mercado/arroz-cepea-demanda-aquecida-mantem-indicador-firme.aspx?pagina='
    urls = []
    url = base_url + str(page_number)
    urls.append(url)
    
    print("Iniciando extração de dados")
    print("1/3 - Obtendo links das notícias")
    # Executa as requisições em paralelo e armazena os links em um único vetor
    links = []
    with ThreadPoolExecutor() as executor:
        for result in executor.map(get_links, urls):
            links += result

            
    def obter_noticias(url: str):
        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }

        response = requests.get(url, headers=headers)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')

        dom = etree.HTML(str(soup))
        noticia = dom.xpath('//*[@id="imagenet-content"]/div[2]/div[1]/div/div/div[1]/div[1]/p')[0].text
        titulo = dom.xpath('//*[@id="imagenet-content"]/div[2]/div[1]/div/h2')[0].text
        return noticia, url, titulo

    print("2/3 - Obtendo notícias")

    with ThreadPoolExecutor(max_workers=8) as executor:
        noticias = []
        for result, url, titulo in executor.map(obter_noticias, links):
            padrao_data = r'\b\d{1,2}/\d{2}/\d{4}\b'
            try:
                data_noticia = re.findall(padrao_data, result)[0]
            except:
                data_noticia = None
            noticias.append({'data': data_noticia, 'titulo': titulo, 'url': url, 'noticia': result})
            df = pd.DataFrame(noticias)

        
        df.to_csv('data/ingestion/RAW/noticias_cepea_incremental.csv', index=False, header=True, mode='w+', quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", doublequote=False)


    