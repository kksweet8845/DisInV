from disindb.models import Report, Brand
import requests
from bs4 import BeautifulSoup as bs
import json
import os, re
from tqdm import tqdm
from datetime import date
from multiprocessing import Pool
from time import time

class cts_crawling:
    brand_name = "華視"
    brand_url = "https://news.cts.com.tw/"
    brand_ID = 18
    menuList = [
        {
            'name': '政治',
            'alias': ['政治']
        },
        {
            'name': '生活',
            'alias': ['生活']
        },
        {
            'name': '社會',
            'alias': ['社會']
        },
        {
            'name': '財經',
            'alias': ['產經','股市']
        },
        {
            'name': '兩岸',
            'alias': ['兩岸']
        },
        {
            'name': '國際',
            'alias': ['全球']
        },
        {
            'name': '體育',
            'alias': ['運動']
        }
    ]
    cateList = ['政治','生活','社會','財經','兩岸','國際','運動']
    idList = [0, 1, 2, 3, 4, 5, 6]

    def __init__(self):
        """ """
        self.brand = Brand.objects.get(id=self.brand_ID)

    def crawl_category(self):
        """ """
        res = requests.get(self.brand_url)
        res_soup = bs(res.content, 'html.parser')
        categories = res_soup.select('div.owl-carousel')[1]
        categories = categories.select('div.item > a')
        ls = []
        for i in categories:
            try:
                index = self.cateList.index(i.get_text())
            except:
                continue
            type_cn = self.menuList[self.idList[index]]['name']
            ls.append({
                'href': self.brand_url + i.attrs['href'],
                'type_cn': type_cn
            })
        self.links = ls


    def request_newsUrl(self, url, type_cn, date):
        ls = []
        res = requests.get(url)
        res_soup = bs(res.content, 'html.parser')
        contents = res_soup.select('div.newslist-container > a')
        for j in contents:
            time = j.select('span.newstime')[0].get_text()
            time = time.replace('/', '-')
            time = re.search('[0-9]+-[0-9]+-[0-9]+', time).group(0)
            if date == 'all' or time in date:
                href = j.attrs['href']
                title = j.attrs['title']
                ls.append({
                    'date': time,
                    'url': href,
                    'title': title
                })
        return ls


    def crawl_newsUrl(self, type_cn='all', date='all'):
        """ """
        ls = []
        pool = Pool(processes=4)
        for i in self.links:
            if type_cn == 'all' or i['type_cn'] in type_cn:
                res = pool.apply_async(self.request_newsUrl, (i['href'], i['type_cn'], date))
                ls.append(res)

        newsUrl = []
        for i in ls:
            newsUrl.extend(i.get())
        self.newsUrl = newsUrl


    def request_newsContent(self, data):
        i = data
        res = requests.get(i['url'])
        res_soup = bs(res.content, 'html.parser')
        contents = res_soup.select('div.artical-content > p')
        if len(contents) == 0:
            contents = res_soup.select('div.artical-content p')
        try:
            author = contents[0].get_text()
            author = re.search('([\S]+)', author).group(0)
        except IndexError:
            author = '綜合報導'
        except AttributeError:
            author = contents[-1].get_text()
            author = re.search('新聞來源：([\S]+)', author).group(1)
        contents = list(map(lambda x: x.get_text(), contents))
        contents = ' '.join(contents[1:-1])

        return {
            'title': i['title'],
            'content': contents,
            'author': author,
            'brand': self.brand,
            'date': i['date'],
            'url': i['url']
        }


    def crawl_newsContent(self):
        """ """
        pool = Pool(processes=8)

        res = pool.map_async(self.request_newsContent, self.newsUrl)
        news = res.get()
        return news


    def getNews(self, date=[date.today().isoformat()]):
        """ """
        self.crawl_category()
        self.crawl_newsUrl(date=date)
        return self.crawl_newsContent()

    def getNewsToday(self):
        """ """
        return self.getNews(date=[date.today().isoformat()])

    def insertNews(self, news):
        ls = []
        cur_news = Report.objects.filter(brand_id=self.brand_ID)
        for dn in news:
            try:
                tmp_news = cur_news.filter(url=dn['url'])
                if len(tmp_news) == 0:
                    tmp = Report(**dn)
                    tmp.save()
                    ls.append(tmp)
            except:
                print(tmp_news)
        return ls if len(ls) != 0 else None