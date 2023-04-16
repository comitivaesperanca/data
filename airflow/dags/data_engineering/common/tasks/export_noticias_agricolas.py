from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
from lxml import etree
import csv 

def export_noticias_agricolas(page_number: int):


    # Define uma função para obter os links de uma página
    def get_links(url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }

        response = requests.get(url, headers=headers)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')

        dom = etree.HTML(str(soup))
        links = dom.xpath('//*[@id="content"]/div[1]/ul')[0].xpath('.//a/@href')
        ## filter if the link is not a video
        links = [link for link in links if not re.search('videos', link)]
        ## join https://www.noticiasagricolas.com.br/ to the links
        links = ['https://www.noticiasagricolas.com.br' + link for link in links]

        return links
            
    links = get_links('https://www.noticiasagricolas.com.br/conteudo/?page=' + str(page_number))
    for link in links:
        noticias = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        response = requests.get(link, headers=headers)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        dom = etree.HTML(str(soup))
        padrao_data = r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}'
        data = dom.xpath('//*[@id="content"]/div[1]')[0].xpath('.//text()')[0]
        titulo = dom.xpath('//*[@id="content"]/h1')[0].xpath('.//text()')[0]
        materia = dom.xpath('//*[@id="content"]/div[3]/div[2]')[0].xpath('.//text()')
        ## get only <p> elements
        materia = [m for m in materia if not m.startswith('\n') and not m.startswith('\xa0')]
        materia = ' '.join(materia)
        data = re.findall(padrao_data, data)[0]
        noticias.append({'data': data, 'titulo': titulo, 'materia': materia})
        df = pd.DataFrame(noticias)
        df.to_csv('data/ingestion/RAW/noticias_agricolas.csv', index=False, mode='a', header=False, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", doublequote=False)