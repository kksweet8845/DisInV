# from newsdb.models import New, Subject, Brand, Brand_sub
from disindb.models import Report, Brand
import requests
from bs4 import BeautifulSoup as bs
import json, os, re
from datetime import date
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool


class pts_crawling:
    brand_name = "公視"
    brand_url = "https://news.pts.org.tw/"
    brand_ID = 10
    menuList = [
        {
            'name': '政治',
            'link': [
                'https://news.pts.org.tw/category/2'
            ],
            'sublinks' : [
                'https://news.pts.org.tw/subcategory/9'
            ]
        },
        {
            'name': '國際',
            'link': [
                'https://news.pts.org.tw/category/4',
            ],
            'sublinks' : [
                "https://news.pts.org.tw/subcategory/11"
            ]
        },
        {
            'name': '生活',
            'link': [
                'https://news.pts.org.tw/category/5'
            ],
            'sublinks':[
                "https://news.pts.org.tw/subcategory/12"
            ]
        },
        {
            'name': '社會',
            'link': [
                'https://news.pts.org.tw/category/7'
            ],
            'sublinks' : [
                "https://news.pts.org.tw/subcategory/14"
            ]
        },
        {
            'name': '體育',
            'sublinks':[
                'https://news.pts.org.tw/subcategory/154'
            ]
        }
    ]

    def __init__(self):
        self.brand = Brand.objects.get(id=self.brand_ID)

    # def insertSubjectUrl(self):
    #     """ """
    #     for di in self.menuList:
    #         tmp_sub = self.sub.get(sub_name=di['name'])
    #         try:
    #             links = di['link']
    #         except:
    #             links = di['sub_link']
    #         for dl in links:
    #             data = {
    #                 'brand': self.brand,
    #                 'index_href': dl,
    #                 'ajax_href': dl
    #             }
    #             try:
    #                 bs = Brand_sub(**data)
    #                 bs.save()
    #             except:
    #                 print(bs)
    #                 return False
    #     return True


    def fn(self, x):
        try:
            return (x.attrs['href'], x.span.get_text())
        except AttributeError:
            return (x.attrs['href'], x.get_text())


    def request_newsUrl(self, url, type_cn, date):
        ls = []
        res = requests.get(url)
        soup = bs(res.text, 'html.parser')
        contents = soup.select('div.news-right-list')
        # contents = list(map(self.fn, contents))
        for i in contents:
            href, title = self.fn( i.select('div.news-right-list > a')[0])
            try:
                time = i.select('div.sweet-info > span')[1].get_text()
            except IndexError:
                continue
            if date == 'all' or time in date:
                ls.append({
                    'url': href,
                    'title': title,
                    'date' : time,
                })
        return ls

    def aux_request_ajax(self, cid, type_cn, i, date):
        ls = []
        url = "https://news.pts.org.tw/subcategory/category_more.php"
        res = requests.post(url, data={
                'cid': cid,
                'page': i
            })
        try:
            res_json = json.loads(res.text)
            if len(res_json) == 0:
                return None
            for dn in res_json:
                if date == 'all' or dn['news_date'] in date:
                    ls.append({
                        'url': 'https://news.pts.org.tw/article/'+dn['news_id'],
                        'title': dn['subject'],
                        'date' : dn['news_date']
                    })
        except:
            return None
        return ls

    def request_ajax(self, cid, type_cn, date):
        ls = []
        result = []
        pool = ThreadPool(processes=4)
        for i in range(1, 51):
            ls.append(pool.apply_async(self.aux_request_ajax, (cid, type_cn, i, date)))

        for i in ls:
            tmp = i.get()
            if tmp != None:
                result.extend(tmp)
        return result


    def crawl_newsUrl(self, type_cn='', date='all'):
        """ """
        newsUrl = []
        ls = []
        pool = Pool(processes=8)
        for dm in tqdm(self.menuList, total=len(self.menuList), desc="L1"):
            links = dm['sublinks']
            for dnewsUrl in tqdm(links, total=len(links), desc=f"sub category {dm['name']}"):
                cid = dnewsUrl.split('/')[-1]
                ls.append(pool.apply_async(self.request_newsUrl, (dnewsUrl, dm['name'], date)))
                ls.append(pool.apply_async(self.request_ajax, (cid, dm['name'], date)))
        for i in tqdm(ls, total=len(ls)):
            newsUrl.extend(i.get())
        self.newsUrl = newsUrl


    def request_newsContent(self, data):
        """ """
        dn = data
        news = requests.get(dn['url'])
        news_soup = bs(news.content, 'html.parser')
        # time = news_soup.select('div.maintype-wapper > h2')[0].get_text()
        # time = re.sub(r'[年月]','-', time )
        # time = re.sub(r'日','', time)
        try:
            article = news_soup.select('article.post-article')[0].get_text()
            author = news_soup.select('span.article-reporter')[0].get_text()
            return {
                'title': dn['title'],
                'content': article,
                'author': author,
                'brand': self.brand,
                'date': dn['date'],
                'url': dn['url']
            }
        except IndexError:
            print(news_soup.select('article.post-content'), dn['url'])

    def crawl_newsContent(self):
        """ """
        pool = Pool(processes=8)
        final_news = []
        ls = []
        for dn in tqdm(self.newsUrl, total=len(self.newsUrl)):
            ls.append(pool.apply_async(self.request_newsContent, (dn, )))

        for i in tqdm(ls, total=len(ls)):
            tmp = i.get()
            if tmp != None:
                final_news.append(tmp)
        return final_news

    def getNews(self, date=[date.today().isoformat()]):
        """ """
        self.crawl_newsUrl(date=date)
        return self.crawl_newsContent()

    def getNewsToday(self):
        """ """
        return self.getNews(date=[date.today().isoformat()])

    def insertNews(self, news):
        """ """
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