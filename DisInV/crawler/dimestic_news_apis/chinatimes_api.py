# local Django
from newsdb.models import New

# third-party
from bs4 import BeautifulSoup
import requests
import pytz

# standard library
from datetime import datetime

class ChinatimesCrawler:

    def __init__(self):
        self.subjects = {
            'society/total': 1,
            'politic/total': 2,
            'world/total': 3,
            'money/total': 4,
            'sport/total': 5,
            'chinese/total': 6,
            'life/total': 7
        }

    def get_news_info (self, url, sub, date):
        soup = self.get_news_soup(url)
        return {
            'brand_id':  8,
            'sub_id':    self.subjects[sub],
            'url':     url,
            'title':   self.get_title(soup),
            'content': self.get_content(soup),
            'date':    date,
            'author':  self.get_author(soup),
        }

    def get_news_soup (self, url):
        res = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(res.text, 'lxml')
        return soup

    def get_title (self, soup):
        try:
            title = soup.select('header.article-header h1.article-title')[0].get_text()
            return "".join( title.split() )
        except:
            return None

    def get_author (self, soup):
        try:
            author = soup.select('div.author a')[0].get_text()
            return author
        except:
            return None

    def get_content (self, soup):
        try:
            news_DOM = soup.select('div.article-body p')
            content = ''
            for DOM in news_DOM:
                content += DOM.get_text()
            return "".join( content.split() )[:2000]
        except:
            return None

    def get_url_by_date(self, sub, date):
        flag = True
        url_category = []
        for page in range(1, 20):
            try:
                res  = requests.get('https://www.chinatimes.com/%s?page=%d&chdtv' % (sub, page), timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                soup = BeautifulSoup(res.text, 'lxml')
                news_DOM_list = soup.select('section.article-list ul.vertical-list li')

                for news_DOM in news_DOM_list:
                    news_date = news_DOM.select('div.row div.col div.meta-info time')[0]['datetime']
                    news_href  = news_DOM.select('div.row div.col h3.title a')[0]['href']

                    if datetime.strptime(news_date, '%Y-%m-%d %H:%M').date() > date:
                        continue
                    elif datetime.strptime(news_date, '%Y-%m-%d %H:%M').date() == date:
                        url_category.append( 'https://www.chinatimes.com%s' % news_href )
                    else:
                        flag = False
                        break

                if flag == False:
                    break
            except Exception as e:
                print(e)
                print('error in get news url')
                break

        return url_category


    def get_news_today( self ):
        timezone = pytz.timezone('Asia/Taipei')
        date_today = datetime.now(timezone).date()

        return self.get_news_by_date( [str(date_today)] )

    def get_news_by_date(self, date_list):
        news_list = []
        for date in date_list:
            for sub in self.subjects:
                url_list = self.get_url_by_date( sub, datetime.strptime(date, '%Y-%m-%d').date() )
                for url in url_list:
                    temp_news = self.get_news_info( url, sub, str(date) )
                    news_list.append( temp_news )

        return news_list

    def insert_news( self, newsList ):
        for news in newsList:
            try:
                temp_news = New.objects.filter(url=news['url'])
                if len(temp_news) == 0:
                    # tmp = New(
                    #     title=news['title'],
                    #     content= news['content'],
                    #     author= news['author'],
                    #     brand_id=news['brand_id'],
                    #     sub_id= news['sub_id'],
                    #     date=news['date'],
                    #     url=news['url'],
                    # )
                    tmp = New(**news)
                    tmp.save()
            except Exception as e:
                print( e )
        return True
