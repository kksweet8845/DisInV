# from newsdb.models import New, Subject, Brand, Brand_sub
# from newsdb.serializers import SubjectSerializer
import requests
from bs4 import BeautifulSoup as bs
import json
import os, re
from tqdm import tqdm
from datetime import date
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from time import time


class ftvnews_crawling:
    brand_name = "民視新聞"
    brand_url = "https://www.ftvnews.com.tw/"
    brand_ID = 12
    cateList = ['政治','生活','社會','財經','兩岸','國際','體育']
    api_url = "https://api.ftvnews.com.tw/api/FtvGetNews"
    def __init__(self):
        self.brand = Brand.objects.get(id=self.brand_ID)
        self.sub = Subject.objects.all()
        pass
    def crawl_category(self):
        """ """
        ls = []
        res = requests.get("https://api.ftvnews.com.tw/api/FtvGetNewsCate")
        cate = json.loads(res.content)
        for i in cate:
            if i['Title'] in self.cateList:
                ls.append({
                    'type_cn': i['Title'],
                    'href': self.brand_url + "overview/"+i['ID'],
                    'cate': i['ID'],
                    'sub': self.sub.get(sub_name=i['Title'])
                })
        self.links = ls


    def encap(self, i, type_cn, date):
        time = i['CreateDate'].replace('/', '-')
        time = re.search('[0-9]+-[0-9]+-[0-9]+', time).group(0)
        if i['Content'] == '':
            content = bs(i['Preface'], 'html.parser')
        else:
            content = bs(i['Content'], 'html.parser')
        try:
            author = content.select('p')[-1].get_text()
            author = re.search(r'[（(][\S]+[)）]', author).group(0)
        except IndexError:
            try:
                author = content.select('li')[-1].get_text()
                author = re.search(r'[（(][\S]+[)）]', author).group(0)
            except IndexError:
                author = ''
            except AttributeError:
                author = ''
                pass
        except AttributeError:
            author = ''
            pass
        author = author.replace(' ', '')
        if author == '':
            author = '綜合報導'
        content = content.get_text()
        if (date == 'all' or time in date) and content != '':
            return {
                'title': i['Title'],
                'url': i['WebLink'],
                'date': time,
                'content': content,
                'brand': self.brand,
                'sub': type_cn,
                'author': author
             }
        else:
            return None

    def aux_news(self, cate, num, date, type_cn):
        ls = []
        query = self.api_url + "?Cate={}&Page={}&Sp=18".format(cate, num)
        res = requests.get(query)
        news = json.loads(res.content)
        try:
            for i in news['ITEM']:
                tmp = self.encap(i, type_cn, date)
                if tmp != None:
                    ls.append(tmp)
        except KeyError:
            print(news)
        return ls

    def request_news(self, info, date):
        query = self.api_url + "?Cate={}&Page={}&Sp=18".format(info['cate'], 1)
        res = requests.get(query)
        news = json.loads(res.content)
        tp = news['PageTotal']
        ls = []
        pool = ThreadPool(processes=8)
        result = []
        for num in tqdm(range(2, int(25)), total=tp, desc="Assign"):
            ls.append(pool.apply_async(self.aux_news, (info['cate'], num, date, info['sub'])))

        for i in tqdm(ls, total=len(ls), desc=info['type_cn']):
            result.extend(i.get())

        for i in news['ITEM']:
            tmp = self.encap(i, info['sub'], date)
            if tmp != None:
                result.append(tmp)
        return result


    def crawl_news(self, type_cn='all', date='all'):

        ls = []
        result = []
        pool = Pool(processes=8)
        for i in tqdm(self.links, total=len(self.links)):
            if type_cn == 'all' or i['type_cn'] in type_cn:
                ls.append(pool.apply_async(self.request_news, (i, date)))
        for i in tqdm(ls, total=len(ls), desc='main'):
            result.extend(i.get())

        return result

    def getNews(self, date=[date.today().isoformat()]):
        self.crawl_category()
        return self.crawl_news(date=date)
    def getNewsToday(self):
        return self.getNews(date=[date.today().isoformat()])

    def insertNews(self, news):
        ls = []
        cur_news = New.objects.filter(brand_id=12)
        for dn in news:
            try:
                tmp_news = cur_news.filter(url=dn['url'])
                if len(tmp_news) == 0:
                    tmp = New(**dn)
                    tmp.save()
                    ls.append(tmp)
            except Exception as err:
                print(err)
                return None
        return ls if len(ls) != 0 else None