# from newsdb.models import New, Subject, Brand, Brand_sub
# from newsdb.serializers import SubjectSerializer
from disindb.models import Report, Brand
import requests
from bs4 import BeautifulSoup as bs
import json
import os, re
from tqdm import tqdm
from datetime import date
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from datetime import datetime

class ltn_crawling:
    brand_name = "自由時報"
    brand_url = "https://news.ltn.com.tw/"
    brand_ID = 11
    ltn_url = {
        1   :{
            'index_href': 'https://news.ltn.com.tw/list/breakingnews/society',
            'ajax_href': 'https://news.ltn.com.tw/ajax/breakingnews/society'
        },
        2   :{
            'index_href': 'https://news.ltn.com.tw/list/breakingnews/politics',
            'ajax_href': 'https://news.ltn.com.tw/ajax/breakingnews/politics'
        },
        3   :{
            'index_href': 'https://news.ltn.com.tw/list/breakingnews/world',
            'ajax_href': 'https://news.ltn.com.tw/ajax/breakingnews/world'
        },
        4   :{
            'index_href': 'https://ec.ltn.com.tw',
            'ajax_href': 'https://ec.ltn.com.tw'
        },
        5   :{
            'index_href': 'https://sports.ltn.com.tw',
            'ajax_href': 'https://sports.ltn.com.tw'
        },
        7   :{
            'index_href': 'https://news.ltn.com.tw/list/breakingnews/life',
            'ajax_href': 'https://news.ltn.com.tw/ajax/breakingnews/life'
        }
    }
    def __init__(self):
        self.brand = Brand.objects.get(id=self.brand_ID)

    # def convertToDict(self, data=None):
    #     """ """
    #     if data == None:
    #         common_subs = Subject.objects.all()
    #         common_subs = SubjectSerializer(common_subs, many=True)
    #         data = common_subs.data

    #     self.subject_dict = {}
    #     for i in data:
    #         self.subject_dict[i['sub_name']] = i['sub_ID']

    # def crawl_categoryUrl(self):
    #     """
    #         Crawl category
    #     It should be used at first.
    #     """

    #     category_sub_res = requests.get(self.brand_url)
    #     category_sub_soup = bs(category_sub_res.content, 'html.parser')
    #     categories = category_sub_soup.select('div.useMobi > ul > li > a')
    #     subs = []
    #     common_subs = Subject.objects.all()
    #     for i in categories:
    #         name = i['data-desc']
    #         for dsub in common_subs:
    #             sub_name = dsub.sub_name
    #             if sub_name == name:
    #                 index_href = i['href']
    #                 ajax_href = i['href'].replace("list", "ajax")
    #                 subs.append({
    #                     'brand' : self.brand,
    #                     'index_href' : index_href,
    #                     'ajax_href' : ajax_href
    #                 })
    #                 break
    #     return subs

    def request_news_url(self, url):
        ls = []
        sub_news_res = requests.get(url)
        sub_news_soup = bs(sub_news_res.text, 'html.parser')
        sub_news = sub_news_soup.find_all('div', attrs={'data-desc': '新聞列表'})
        if len(sub_news) > 0:
            sub_news = sub_news[0]
        else:
            return ls
        sub_news = sub_news.select('div > ul > li > a.tit')
        for ds in sub_news:
            time = ds.select('span.time')[0].get_text().strip()
            try:
                time = datetime.strptime(time, '%Y/%m/%d %H:%M')
            except ValueError:
                cur = date.today()
                time = datetime.strptime(time, '%H:%M')
                time = time.replace(year=cur.year, month=cur.month, day=cur.day)
            tmp = ds.attrs
            ls.append({
                'url' : tmp['href'],
                'title' : tmp['data-desc'],
                'date': time
            })
        return ls

    def aux_ajax_url(self, url, i):
        query_string = os.path.join(url, f"{i}")
        res = requests.get(query_string)
        ls = []
        try:
            text = json.loads(res.text)['data']
            if isinstance(text,dict) and len(text.keys()):
                for key in text.keys():
                    time = text[key]['time']
                    try:
                        time = datetime.strptime(time, '%Y/%m/%d %H:%M')
                    except ValueError:
                        cur = date.today()
                        time = datetime.strptime(time, '%H:%M')
                        time = time.replace(year=cur.year, month=cur.month, day=cur.day)
                    tmp = {
                        'url': text[key]['url'],
                        'title': text[key]['title'],
                        'date': time
                    }
                    ls.append(tmp)
        except ValueError:
            return None
        return ls

    def request_ajax_url(self, url):
        result = []
        ls = []
        pool = ThreadPool(processes=8)
        for i in range(2,25):
            ls.append(pool.apply_async(self.aux_ajax_url, (url, i)))
        for i in ls:
            tmp = i.get()
            if tmp != None:
                result.extend(tmp)

        return result

    def crawlingNewsUrl(self, type_cn='all'):
        """ Category is a QuerySet """
        # if type_cn == 'all':
        #     urls = Brand_sub.objects.filter(brand_id=self.brand_ID)
        # else:
        #     urls = Brand_sub.objects.filter(brand_id=self.brand_ID, sub__sub_name=type_cn)

        # if urls == None:
        #     return None

        pool = Pool(processes=8)
        ls = []
        news = []
        for sub_id in self.ltn_url.keys():
            index_url = self.ltn_url[sub_id]['index_href']
            ajax_url = self.ltn_url[sub_id]['ajax_href']
            ls.append(pool.apply_async(self.request_news_url, (index_url, )))
            ls.append(pool.apply_async(self.request_ajax_url, (ajax_url, )))

        for i in tqdm(ls, total=len(ls)):
            news.extend(i.get())
        self.news = news

    def request_newsContent(self, data):
        """ """
        ls = []
        d = data
        res = requests.get(d['url'])
        if res.status_code != 200:
            return None
        res_soup = bs(res.text, 'html.parser')
        result = res_soup.select('div.boxTitle > p:not(.appE1121, .before_ir, .after_ir, .ga_event)')
        result = map(lambda x: x.get_text(), result)
        result = ' '.join(list(result)[1:-1])
        try:
            author = re.search('〔記者([\S]+?)／', result).group(1)
        except AttributeError:
            author = '即時新聞'
        date = d['date']
        return {
            'title': d['title'],
            'content' : result,
            'author' : author,
            'brand': self.brand,
            'date' : "{}-{}-{}".format(date.year, date.month, date.day),
            'url': d['url']
        }


    def crawlingNewsContent(self, date=[datetime.now()]):
        final_news = []
        news = self.news
        ls = []
        pool = Pool(processes=8)
        for d in tqdm(news, total=len(news)):
            dtime = d['date']
            if date == 'all':
                ls.append(pool.apply_async(self.request_newsContent, (d, )))
                continue
            for dd in date:
                if dtime.year == dd.year and dtime.month == dd.month and dtime.day == dd.day:
                    ls.append(pool.apply_async(self.request_newsContent, (d, )))
                    break
        for i in tqdm(ls, total=len(ls)):
            tmp = i.get()
            if isinstance(tmp, dict):
                final_news.append(tmp)
        return final_news

    # def getSubjectUrl(self):
    #     """ standard api"""
    #     data = self.crawl_categoryUrl()
    #     return data

    # def insertSubjectUrl(self, subs):
    #     """ """
    #     for ds in subs:
    #         try:
    #             tmp = Brand_sub(**ds)
    #             tmp.save()
    #         except:
    #             return False
    #     return True

    def getNews(self, date=[datetime.now()]):
        """ """
        self.crawlingNewsUrl()
        ls = []
        for i in date:
            ls.append(datetime.strptime(i, '%Y-%m-%d'))
        return self.crawlingNewsContent(date=ls)

    def getNewsToday(self):
        """ """
        return self.getNews(date=[datetime.now()])

    def insertNews(self, news):
        ls = []
        cur_news = Report.objects.filter(brand_id=11)
        for dn in news:
            try:
                tmp_news = cur_news.filter(url=dn['url'])
                if len(tmp_news) == 0:
                    tmp = Report(**dn)
                    tmp.save()
                    ls.append(tmp)
            except:
                return None
        return ls if len(ls) != 0 else None


