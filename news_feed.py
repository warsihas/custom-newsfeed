import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import schedule
from joblib import Parallel, delayed
import time
import pickle
import json
import gspread
import requests
from oauth2client.client import SignedJwtAssertionCredentials
from bs4 import BeautifulSoup
from helper_functions import get_article_text, clean_str
import spacy
import numpy as np

import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

nlp = spacy.load("en_core_web_sm")

pd.set_option('display.max_colwidth', 250)

def fetch_news():
    try:
        vect = pickle.load(open(r'tf_idf_vect.pkl', 'rb'))
        model = pickle.load(open(r'svc_model.pkl', 'rb'))

        gc = gspread.oauth()



        huffpost_sheet = gc.open_by_url('link to google sheet where article urls are stored')
        sh = huffpost_sheet.sheet1
        huffpost_url = pd.DataFrame(huffpost_sheet.sheet1.get_all_records() )[['url', 'Headline']]


        news_digest_texts = Parallel(n_jobs=-1)(delayed(get_article_text)(url) for url in huffpost_url['url'])


        news_digest_df = pd.concat([huffpost_url.reset_index(drop=True), pd.Series(news_digest_texts).reset_index(drop=True)], axis = 1)        
        news_digest_df.columns = ['url', 'headline', 'article_text']

        news_digest_df.replace('', np.nan, inplace=True)
        news_digest_df.dropna(inplace=True)


        news_digest_df['article_text'] = news_digest_df['article_text'].str.lower() 
        news_digest_df['article_text_clean'] = news_digest_df['article_text'].apply(clean_str)

        news_digest_df['article_text_clean'] = news_digest_df['article_text_clean'].apply(lambda x: " ".join(token.lemma_ for token in nlp(x)) )

        news_digest_df['article_text_clean'] = news_digest_df['article_text_clean'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))


        tv = vect.transform(news_digest_df['article_text_clean'])
        res = model.predict(tv)

        rf = pd.DataFrame(res, columns=['wanted'])
        rez = pd.merge(rf, news_digest_df, left_index=True, right_index=True)

        news_str = ''
        for t, u in zip(rez.loc[rez['wanted']=='y', 'headline'], rez.loc[rez['wanted']=='y', 'url']):
            news_str = news_str + t + '\n' + u + '\n'

        print(news_str)    

        payload = {"value1" : news_str}
        r = requests.post('https://maker.ifttt.com/trigger/new_article/with/key/iftttkey', data=payload)

        # clean up worksheet
        lenv = len(sh.col_values(1))
        cell_list = sh.range('A2:F' + str(lenv))
        for cell in cell_list:
            cell.value = ""
        sh.update_cells(cell_list)
        print(r.text)
        
    except:
        print('Failed')

fetch_news()

#schedule.every(480).minutes.do(fetch_news)

# while 1:
#     schedule.run_pending()
#     time.sleep(1)