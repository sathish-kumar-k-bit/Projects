#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install transformers')


# In[2]:


from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests


# In[3]:


model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)


# In[38]:


monitored_tickers = ['reliance', 'grasim', 'hindalco', 'TCS']


# In[39]:


def search_for_stock_news_urls(ticker):
    search_url = "https://www.google.com/search?q=mint+{}&tbm=nws".format(ticker)
    r = requests.get(search_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    atags = soup.find_all('a')
    hrefs = [link['href'] for link in atags]
    return hrefs


# In[40]:


search_url = "https://www.google.com/search?q=mint+{}&tbm=nws".format('TCS')
search_url


# In[51]:


search_for_stock_news_urls('grasim')


# In[55]:


raw_urls = {ticker:search_for_stock_news_urls(ticker) for ticker in monitored_tickers}
raw_urls


# In[56]:


raw_urls.keys()


# In[57]:


raw_urls.values()


# In[58]:


import re


# In[59]:


exclude_list = ['maps' ,'policies' ,'preferences', 'accounts', 'support']


# In[60]:


def strip_unwanted_urls(urls,exclude_list):
    val = []
    for url in urls:
        if 'https://' in url and not any(exclude_word in url for exclude_word in exclude_list):
            res = re.findall(r'(https?://\S+)',url)[0].split('&')[0]
            val.append(res)
    return list(set(val))


# In[61]:


strip_unwanted_urls(raw_urls['reliance'],exclude_list)


# In[62]:


cleaned_urls = {ticker:strip_unwanted_urls(raw_urls[ticker], exclude_list) for ticker in monitored_tickers}
cleaned_urls


# In[68]:


def scrape_and_process(URLs):
    articles = []
    for url in URLs:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = [paragraph.text for paragraph in paragraphs]
        words = ' '.join(text).split(' ')[:300]
        article = ' '.join(words)
        articles.append(article)
    return articles


# In[69]:


articles = {ticker:scrape_and_process(cleaned_urls[ticker]) for ticker in monitored_tickers}
articles


# In[70]:


articles['reliance']


# In[71]:


def summarize(articles):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article,return_tensors='pt')
        output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries


# In[72]:


summaries = {ticker:summarize(articles[ticker]) for ticker in monitored_tickers}
summaries


# In[73]:


summaries['reliance']


# In[74]:


from transformers import pipeline
sentiment = pipeline('sentiment-analysis')


# In[75]:


scores = {ticker:sentiment(summaries[ticker]) for ticker in monitored_tickers}
scores


# In[76]:


print(summaries['reliance'][0], scores['reliance'][0]['label'], scores['reliance'][0]['score'])


# In[77]:


print(summaries['grasim'][0], scores['grasim'][0]['label'], scores['grasim'][0]['score'])


# In[84]:


def create_output_array(summaries, scores, urls):
    output = []
    for ticker in monitored_tickers:
        for counter in range(len(summaries[ticker])):
            output_this = [
                ticker,
                summaries[ticker][counter],
                scores[ticker][counter]['label'],
                scores[ticker][counter]['score'],
                urls[ticker][counter]
            ]
            output.append(output_this)
    return output


# In[85]:


final_output = create_output_array(summaries, scores, cleaned_urls)
final_output


# In[86]:


final_output.insert(0,['Ticker','Summary', 'Label', 'Confidence', 'URL'])


# In[87]:


final_output


# In[88]:


import csv
with open('stocksummaries.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)


# In[ ]:




