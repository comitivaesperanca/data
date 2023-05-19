import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import pickle
import requests
import torch
import torch.nn as nn
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModelForPreTraining  # Or BertForPreTraining for loading pretraining heads
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads
from typing import List, Optional, Tuple, Union
from datetime import datetime

def classification_publish(model: str = 'naive_bayes'):
    if model == 'naive_bayes':
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

        model = pickle.load(open('data/model/modelo_naive_soja.pkl', 'rb'))

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
    elif model == 'rede_neural':    
        nltk.download('stopwords')
        nltk.download('punkt')

        df_original = pd.read_csv('data/ingestion/RAW/noticias_cepea.csv', sep=',', on_bad_lines='skip', encoding='utf-8')[:2000]
        df = df_original.copy()
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

        class LIABertClassifier(nn.Module):
            def __init__(self,model,num_labels):
                super(LIABertClassifier,self).__init__()
                self.bert = model
                self.config = model.config
                self.num_labels = num_labels
                self.cls = nn.Linear(self.config.hidden_size,200)
                self.dropout = nn.Dropout(p=0.5)
                self.cls2 = nn.Linear(768,num_labels)

            def forward(
                self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                ) ->Tuple[torch.Tensor]:

                outputs = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )

                sequence_output = outputs[0][:,0,:]
                prediction = self.dropout(sequence_output)
                prediction = self.cls2(prediction)
                return prediction

        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=3)
        model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

        device = torch.device('cpu')

        # Crie uma instância de seu modelo
        modelo_soja = LIABertClassifier(model, 3)
        modelo_soja.load_state_dict(torch.load("data/model/modelo_redeneural_bert_soja.pt", map_location=torch.device('cpu')))

        modelo_soja.to(device)

        ## predict para o dataframe
        df_noticias_predict = df_noticias.copy()
        df_noticias_predict['noticia'] = df_noticias_predict['noticia'].apply(preprocess_text)
        df_noticias_predict['noticia'] = df_noticias_predict['noticia'].apply(lambda x: tokenizer.encode(x, return_tensors="pt"))
        df_noticias_predict['predict'] = df_noticias_predict['noticia'].apply(lambda x: modelo_soja(x).argmax(axis=1).cpu().numpy().tolist()[0])
        df_noticias_predict['prob_negativa'] = df_noticias_predict['noticia'].apply(lambda x: modelo_soja(x).softmax(axis=1).cpu().detach().numpy().tolist()[0][0])
        df_noticias_predict['prob_positiva'] = df_noticias_predict['noticia'].apply(lambda x: modelo_soja(x).softmax(axis=1).cpu().detach().numpy().tolist()[0][1])
        df_noticias_predict['prob_neutra'] = df_noticias_predict['noticia'].apply(lambda x: modelo_soja(x).softmax(axis=1).cpu().detach().numpy().tolist()[0][2])
        df_noticias_predict['predict'] = df_noticias_predict['predict'].apply(lambda x: 'Negativa' if x == 0 else 'Positiva' if x == 1 else 'Neutra')
        df_noticias_predict.drop(columns=['noticia'], inplace=True)
        df_noticias_predict

        def get_random_date(start, end):
            import random
            from datetime import timedelta
            """
            This function will return a random datetime between two datetime 
            objects.
            """
            # delta between '2023-01-01' and '2023-05-18'
            delta = datetime.strptime(end, '%Y-%m-%d') - datetime.strptime(start, '%Y-%m-%d')
            int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
            random_second = random.randrange(int_delta)
            return datetime.strptime(start, '%Y-%m-%d') + timedelta(seconds=random_second)


        
        df_total = df_original[1:].copy()
        # join df_noticias_predict
        df_total = df_total.merge(df_noticias_predict, how='right', on=['data', 'titulo', 'url'])
        def publicar_noticia(row):
            url = 'http://20.190.249.236/news'
            data = {
                'title': row['titulo'],
                'newsContent': row['noticia'],
                "commodityType": "Soja",
                "source": "CEPEA",
                # parse date to 2023-05-13T17:54:37.242Z
                "publicationDate": get_random_date('2023-05-11', '2023-05-18').strftime('%Y-%m-%dT%H:%M:%S.00Z'),
                "positiveSentiment": row['prob_positiva'],
                "neutralSentiment": row['prob_neutra'],
                "negativeSentiment": row['prob_negativa'],
                "finalSentiment": row['predict'],
            }
            print(data)
            response = requests.post(url, json=data)
            print(response.status_code, response.text)

        df_total.apply(publicar_noticia, axis=1)

    
