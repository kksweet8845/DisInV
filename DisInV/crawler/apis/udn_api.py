# from newsdb.models import New, Subject, Brand, Brand_sub
import requests
from bs4 import BeautifulSoup as bs
import json, os, re
from datetime import date
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool


class udn_crawling:
    brand_name = "聯合新聞網"
    brand_url = "https://udn.com/news/index"
    domain = "https://udn.com/"
    brand_ID = 16
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
    cateList = ['政治','生活','社會','產經','股市','兩岸','全球','運動']
    idList = [0, 1, 2, 3, 3, 4, 5, 6]
    def __init__(self):
        self.brand = Brand.objects.get(id=self.brand_ID)
        self.sub = Subject.objects.all()
        pass

    def crawl_category(self):
        """ """
        res = requests.get(self.brand_url)
        res_soup = bs(res.content, 'html.parser')
        category = res_soup.select('nav.navigation > a')[1:]
        ls = []
        for i in category:
            try:
                index = self.cateList.index(i.get_text())
            except:
                continue
            type_cn = self.menuList[self.idList[index]]['name']
            tmp = {
                'href': self.domain + i.attrs['href'],
                'type_cn': type_cn
            }
            ls.append(tmp)
        self.links = ls

    def insertSubjectUrl(self):
        """ """
        for di in self.links:
            tmp_sub = self.sub.get(sub_name=di['type_cn'])
            data = {
                'sub': tmp_sub,
                'brand': self.brand,
                'index_href': di['href'],
                'ajax_href': di['href']
            }
            try:
                bs = Brand_sub(**data)
                bs.save()
            except:
                print(bs)
                return False
        return True



    def aux_ajax_more(self, i, form_data, type_cn, num, date):
        """ """
        ls = []
        domain = "https://udn.com/api/more"
        query = "?page={}&channelId=2&type={}&cate_id={}&totalRecNo={}".format(
                        num,
                        i['data_type'],
                        form_data['cate_id'],
                        form_data['totalRecNo']
                    )
        res = requests.get(domain + query)
        try:
            result = json.loads(res.text)
            for dr in result['lists']:
                time = re.search('[0-9]+-[0-9]+-[0-9]+', dr['time']['date']).group(0)
                if date == 'all' or time in date:
                    ls.append({
                        'title': dr['title'],
                        'url': self.domain + dr['titleLink'],
                        'date': time,
                        'sub' : self.sub.get(sub_name=type_cn)
                    })
        except KeyError :
            return None
        except ValueError:
            return None
        return ls

    def aux_ajax_other(self, i, form_data, type_cn, num, date):
        ls = []

        domain = "https://udn.com/api/more"
        query = "?page={}&channelId=2&type={}&cate_id={}&sub_id={}&totalRecNo={}".format(
            num,
            i['data_type'],
            form_data['cate_id'],
            form_data['sub_id'],
            form_data['totalRecNo']
        )
        res = requests.get(domain + query)
        try:
            result = json.loads(res.text)
            for dr in result['lists']:
                time = re.search('[0-9]+-[0-9]+-[0-9]+', dr['time']['date']).group(0)
                if date == 'all' or dr['time']['date'] in date:
                    ls.append({
                        'title': dr['title'],
                        'url': self.domain + dr['titleLink'],
                        'date': time,
                        'sub': self.sub.get(sub_name=type_cn)
                        })
        except KeyError :
            return None
        except ValueError:
            return None
        return ls


    def request_ajaxNewsUrl(self, news, other, type_cn, date):
        """ """
        ls = []
        result = []
        pool = ThreadPool(processes=8)
        for i in tqdm(news, total=len(news), desc="L2-more"):
            i['query'] = i['query'].replace("\'", "\"")
            form_data = json.loads(i['query'])
            for num in range(1,21):
                ls.append(pool.apply_async(self.aux_ajax_more, (i, form_data, type_cn, num, date)))

        for i in tqdm(other, total=len(other), desc="L2-other"):
            i['query'] = i['query'].replace("\'", "\"")
            form_data = json.loads(i['query'])
            for num in range(1,21):
                ls.append(pool.apply_async(self.aux_ajax_other, (i, form_data, type_cn, num, date)))

        for i in tqdm(ls, total=len(ls)):
            tmp = i.get()
            if tmp != None:
                result.extend(tmp)

        return result



    def request_newsUrl(self, url, type_cn, date):

        ls = []
        res = requests.get(url)
        soup = bs(res.text, 'html.parser')
        contents = soup.select('div.story-list__text')
        more_btn = soup.select('button[aria-label="more-news"]')
        more_other = soup.select('button[aria-label="more-other"]')
        more_btn = list(map(lambda x: {
            'query': x.attrs['data-query'],
            'data_type': x.attrs['data-type'],
            'data_page': int(x.attrs['data-page']) + 1}, more_btn))
        more_other = list(map(lambda x: {
            'query': x.attrs['data-query'],
            'data_type': x.attrs['data-type'],
            'data_page': int(x.attrs['data-page']) + 1
        }, more_other))
        for di in tqdm(contents, total=len(contents), desc="L2-news"):
            try:
                a_tag = di.select('h3 > a')[0]
                news_url = a_tag.attrs['href']
                if hasattr(a_tag.attrs, 'title'):
                    title = a_tag.attrs['title']
                else:
                    continue
            except IndexError:
                # print(a_tag.attrs)
                a_tag = di.select('h2 > a')[0]
                # print(a_tag.attrs)
                news_url = a_tag.attrs['href']
                title = a_tag.attrs['title']
            try:
                time = di.select('div.sotry-list__info > time.story-list__info')[0].get_text()
                time = re.search('[0-9]+-[0-9]+-[0-9]+', time).group(0)
            except:
                time = '1999-01-01'
            if date == 'all' or time in date:
                ls.append({
                    'url': self.domain + news_url,
                    'title': title,
                    'sub': self.sub.get(sub_name=type_cn),
                    'date': time
                })
        return ls, more_btn, more_other


    def aux_crawl_newsUrl(self, i, date):
        tmp_newsUrl, news, other = self.request_newsUrl(i['href'], i['type_cn'], date)
        tmp_newsUrl.extend(self.request_ajaxNewsUrl(news, other, i['type_cn'], date))
        return tmp_newsUrl

    def crawl_newsUrl(self, type_cn='all', date=[date.today().isoformat()]):
        """ """
        newsUrl = []
        pool = Pool(processes=8)
        ls = []
        for i in tqdm(self.links, total=len(self.links), desc="L1"):
            if type_cn == 'all' or i['type_cn'] in type_cn:
                ls.append(pool.apply_async(self.aux_crawl_newsUrl, (i, date)))

        for i in tqdm(ls, total=len(ls), desc='News'):
            newsUrl.extend(i.get())

        self.newsUrl = newsUrl


    def fn(self, x):
        x = re.sub(r'[\W]', '', x)
        return x

    def request_newsContent(self, dn):
        """ """
        res = requests.get(dn['url'])
        res_soup = bs(res.content, 'html.parser')
        contents = list(map(lambda x: x.get_text(), res_soup.select('section.article-content__editor > p')))
        contents = list(map(self.fn, contents))
        contents = ' '.join(contents)
        if len(contents) == 0:
            return None
        try:
            author = res_soup.select('span.article-content__author')[0].get_text().strip()
            author = re.search('記者([\S]+?)／',author).group(1)
        except:
            author = "None"
        return {
            'title': dn['title'],
            'content': contents,
            'author': author,
            'brand': self.brand,
            'sub': dn['sub'],
            'date': dn['date'],
            'url': dn['url']
        }


    def crawl_newsContent(self):
        """ """
        final_news = []
        ls = []
        pool = Pool(processes=20)
        for dn in tqdm(self.newsUrl, total=len(self.newsUrl)):
            ls.append(pool.apply_async(self.request_newsContent, (dn, )))

        for i in tqdm(ls, total=len(ls)):
            tmp = i.get()
            if tmp != None:
                final_news.append(i.get())
        return final_news

    def getNews(self, date=[date.today().isoformat()]):
        """ """

        self.crawl_category()
        self.crawl_newsUrl(type_cn='all', date=date)
        return self.crawl_newsContent()

    def getNewsToday(self):
        """ """
        return self.getNews(date=[date.today().isoformat()])

    def insertNews(self, news):
        ls = []
        cur_news = New.objects.filter(brand_id=self.brand_ID)
        for dn in news:
            try:
                tmp_news = cur_news.filter(url=dn['url'])
                if len(tmp_news) == 0:
                    tmp = New(**dn)
                    tmp.save()
                    ls.append(tmp)
            except:
                print(tmp_news)
        return ls if len(ls) != 0 else None