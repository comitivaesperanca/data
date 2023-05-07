import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import pickle
import requests

def classification_publish():
    nltk.download('stopwords')
    nltk.download('punkt')

    df = pd.read_csv('data/ingestion/RAW/noticias_cepea_incremental.csv', sep=',', on_bad_lines='skip', encoding='utf-8')
    df = df[df['data'] != 'data']
    df = df.dropna()
    df
    padrao_data_cepea = r"Cepea, \d{2}/\d{2}/\d{4} - "
    df['noticia'] = df['noticia'].apply(lambda x: re.sub(padrao_data_cepea, '', x))

    ## remover a palavra 'cepea' das noticias
    padrao_cepea = r"Cepea"
    df['noticia'] = df['noticia'].apply(lambda x: re.sub(padrao_cepea, '', x, flags=re.IGNORECASE))

    ## remover numeros das noticias
    padrao_numeros = r'[0-9]+'
    df['noticia'] = df['noticia'].apply(lambda x: re.sub(padrao_numeros, '', x))

    ## noticia que contem a palavra 'soja'
    df = df[df['titulo'].str.contains('soja', flags=re.IGNORECASE)]

    df


    # Pré-processamento dos dados
    stop_words = set(stopwords.words('portuguese'))

    def preprocess_text(text):
        # remover acentuação
        text = unidecode(text)
        # Remover pontuações
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenização
        words = word_tokenize(text.lower())
        # Remover stopwords
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)

    df['noticia'] = df['noticia'].apply(preprocess_text)

    df_noticias = df.copy()

    # Vetorização
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import classification_report

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['noticia'])

    # Carregar o modelo 

    model = pickle.load(open('data/model/modelo.pkl', 'rb'))

    df_noticias = df_noticias.dropna(subset=['noticia']).reset_index(drop=True)
    y_pred = model.predict_proba(df_noticias['noticia'])
    y_pred_class = model.predict(df_noticias['noticia'])

    ## eliminar o e+01 do valor numerico em pandas
    pd.options.display.float_format = '{:.2f}'.format

    df_predict = pd.DataFrame(y_pred, columns=model.classes_)
    df_predict['predict_final'] = df_predict.idxmax(axis=1)

    df_pred_concatenado = pd.concat([df_noticias, df_predict], axis=1)
    df_pred_concatenado[['data', 'titulo', 'noticia', 'predict_final', 'Neutra', 'Negativa', 'Positiva']] #.to_excel('validacao.xlsx', index=False)


    ## print group by predict_final
    df_qtde_noticias = df_pred_concatenado.groupby('predict_final').count()['titulo']
    print("="*10, "\nQuantidade de notícias por sentimento", df_qtde_noticias,"\n", "="*10)

    ## publicar na api 

    def publicar_noticia(row):
        url = 'http://20.241.232.187/news'
        data = {
            'title': row['titulo'],
            'newsContent': row['noticia'],
            "commodityType": "Soja",
            "source": "CEPEA",
            "positiveSentiment": row['Positiva'],
            "neutralSentiment": row['Neutra'],
            "negativeSentiment": row['Negativa'],
            "finalSentiment": row['predict_final'],
        }
        response = requests.post(url, json=data)
        print(response.status_code, response.text)

    df_pred_concatenado.apply(publicar_noticia, axis=1)
