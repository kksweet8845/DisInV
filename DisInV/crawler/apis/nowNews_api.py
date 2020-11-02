# from newsdb.models import New, Subject, Brand, Brand_sub
from disindb.models import Report, Brand
import requests, os, re
from bs4 import BeautifulSoup as bs
from tqdm import tqdm
from datetime import date
from multiprocessing import Pool

# Create your views here.
class nowNews_crawling:
    brand_name = "今日新聞"
    brand_url = "https://www.nownews.com/"
    brand_ID = 9
    menuList = [
        {
            'name': '社會',
            'link': [
                'https://www.nownews.com/cat/society'
            ]
        },
        {
            'name': '政治',
            'link': [
                'https://www.nownews.com/cat/politics/',
                'https://www.nownews.com/cat/politics/constitutionalforum',
                'https://www.nownews.com/cat/politics/military',
                'https://www.nownews.com/cat/politics/analysis',
                'https://www.nownews.com/cat/politics/sen-lian'
            ]
        },
        {
            'name': '國際',
            'link': [
                'https://www.nownews.com/cat/global',
                'https://www.nownews.com/cat/global/frontpage',
                'https://www.nownews.com/cat/global/intlfun',
                'https://www.nownews.com/cat/global/asia-news'
            ]
        },
        {
            'name': '財經',
            'link': [
                'https://www.nownews.com/cat/finance',
                'https://www.nownews.com/cat/finance/nowfinance',
                'https://www.nownews.com/cat/finance/industry',
                'https://www.nownews.com/cat/finance/financial',
                'https://www.nownews.com/cat/finance/people',
                'https://www.nownews.com/cat/finance/careers',
                'https://www.nownews.com/cat/finance/housenews',
                'https://www.nownews.com/cat/finance/sign',
                'https://www.nownews.com/cat/finance/online-banking'
            ]
        },
        {
            'name': '體育',
            'link': [
                'https://www.nownews.com/cat/sport',
                'https://www.nownews.com/cat/sport/dreamers',
                'https://www.nownews.com/cat/sport/nba',
                'https://www.nownews.com/cat/sport/mlb',
                'https://www.nownews.com/cat/sport/npb-sport',
                'https://www.nownews.com/cat/sport/baseball',
                'https://www.nownews.com/cat/sport/baseketball',
                'https://www.nownews.com/cat/sport/comple',
                'https://www.nownews.com/cat/sport/sportsinside',
                'https://www.nownews.com/cat/sport/antarctica-adventure',
                'https://thankyouchia.nownews.com/',
            ]
        },
        {
            'name': '兩岸',
            'link': [
                'https://www.nownews.com/cat/chinaindex',
                'https://www.nownews.com/cat/chinaindex/xfile'
            ]
        },
        {
            'name': '生活',
            'link': [
                'https://www.nownews.com/cat/life',
                'https://www.nownews.com/cat/life/lifetopic',
                'https://www.nownews.com/cat/life/smart-life',
                'https://www.nownews.com/cat/life/food-life',
                'https://www.nownews.com/cat/life/eworld',
                'https://www.nownews.com/cat/life/culture-artistic',
                'https://www.nownews.com/cat/life/life-focus',
                'https://www.nownews.com/cat/life/public'
            ]
        }
    ]
    def __init__(self):
        self.brand = Brand.objects.get(id=self.brand_ID)

    # def getSubjectUrl(self):
    #     """ """
    #     print(self.menuList)

    # def insertSubjectUrl(self):
    #     """ """
    #     subs = Subject.objects.all()
    #     for di in self.menuList:
    #         tmp_sub = subs.get(sub_name=di['name'])
    #         for dl in di['link']:
    #             data = {
    #                 'sub': tmp_sub,
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


    def request_newsUrl(self, url):
            ls = []
            res = requests.get(url)
            soup = bs(res.text, 'html.parser')
            contents = soup.select('div.td-module-meta-info')
            # contents = soup.select('h3.entry-title > a')
            for di in contents:
                h3 = di.select('h3 a')
                time = di.select('time')
                if len(h3) == 0:
                    continue
                else:
                    tmp = h3[0].attrs
                    time = time[0].get_text()
                    ls.append({
                        'url' : tmp['href'],
                        'title' : tmp['title'],
                        'date': time
                    })
            return ls

    def crawl_newsUrl(self, type_cn='all'):
        """ """
        # subs = Subject.objects.all()
        news = []
        ls = []
        pool = Pool(processes=8)
        for dm in tqdm(self.menuList, total=len(self.menuList)):
            if type_cn == 'all' or  dm['name'] in type_cn:
                for durl in dm['link']:
                    index_url = durl
                    ajax_url = index_url + "/page/"
                    # sub = subs.get(sub_name=dm['name'])
                    ls.append(pool.apply_async(self.request_newsUrl, (index_url, )))
                    for j in range(1,7):
                        tmp_ajax_url = ajax_url + "{}/".format(j)
                        ls.append(pool.apply_async(self.request_newsUrl, (tmp_ajax_url, )))
        for i in tqdm(ls, total=len(ls)):
            news.extend(i.get())
        self.newsUrl = news

    def request_newsContent(self, data):
        dn = data
        ls = []
        test = requests.get(dn['url'])
        test_soup = bs(test.content, 'html.parser')
        contents = test_soup.select('span[itemprop="articleBody"] > p')
        if len(contents) == 0:
            contents = test_soup.select('spean[itemprop="articleBody"]  section > p')
        # try:
        #     newsDate = test_soup.select('time.entry-date')[0].attrs['datetime']
        #     newsDate = re.search('[0-9]+-[0-9]+-[0-9]+', newsDate).group(0)
        # except IndexError:
        #     print(test_soup.prettify())
        #     print(test_soup.select('time.entry-date'))
        #     print(dn['url'])
        #     print('Time')
        #     newsDate = "1999-01-01"
        #     pass
        # if date == 'all' or newsDate in date:
        contents = map(lambda x: x.get_text(), contents)
        content = ' '.join(list(contents))
        try:
            author = test_soup.select('div.td-post-author-name')[0].get_text().strip()
            author = author.replace(' ', '')
            author = author.replace('-', '')
        except IndexError:
            print(dn['url'])
            author = "None"
            pass
        ls.append({
            'title': dn['title'],
            'content': content,
            'author': author,
            'brand': self.brand,
            'date': dn['date'],
            'url': dn['url']
        })
        return ls

    def crawl_newsContent(self, date=[date.today().isoformat()]):
        """ """
        final_news = []
        ls = []
        pool = Pool(processes=12)
        for dn in tqdm(self.newsUrl, total=len(self.newsUrl)):
            if date == 'all' or dn['date'] in date:
                ls.append(pool.apply_async(self.request_newsContent, (dn,)))

        for i in tqdm(ls, total=len(ls)):
            tmp = i.get()
            if tmp != None:
                final_news.extend(i.get())
        return final_news

    def getNews(self, date=[date.today().isoformat()]):
        """ """
        self.crawl_newsUrl()
        return self.crawl_newsContent(date=date)

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
                print(tmp)
        return ls if len(ls) != 0 else None