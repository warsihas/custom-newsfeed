from bs4 import BeautifulSoup
import requests
import re


def get_article_text(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    return " ".join(x.get_text() for x in soup.find_all('p'))

def clean_str(s):
    s = re.sub(r"\'s", " is", s) # replace 's with is
    s = re.sub(r"\'ve", " have", s) #replace 've with have
    s = re.sub(r"n\'t", " not", s) #replace 've with not
    s = re.sub(r"\'ll", " will", s) #replace 'll with will
    s = re.sub(r"http\S+", ' ', s) # replace urls with blank space
    s = re.sub(r"[^a-z \-]", " ", s) # remove all characters that are not alpha
    s = re.sub(r"[\-]", " ", s) # replace hyphens to join words
    s = re.sub(r"\b[a-zA-Z]\b", " ", s) # remove any one letter words
    s = re.sub('\S*@\S*\s?', '', s) # remove emails
    s = re.sub('\s+', ' ', s) # remove newline characters
    s = re.sub(r"\s{2,}", " ", s) # replace any extra spaces with one space
    return s.strip()