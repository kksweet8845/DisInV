# local Django
from newsdb.models import New

# third-party
from bs4 import BeautifulSoup
import requests
import pytz

# standard library
from datetime import datetime, date
import re

class NewtalkCrawler:

    def __init__(self):
        self.subjects = {
            '2/政治': 2,
            '1/國際': 3,
            '4/司法': 1,
            '14/社會': 1,
            '3/財經': 4,
            '7/中國': 6,
            '5/生活': 7,
            '102/體育': 5,
        }

    def get_news_info (self, url, sub, date):
        soup = self.get_news_soup(url)
        if soup != None:
            return {
                'brand_id':  15,
                'sub_id':    self.subjects[sub],
                'url':     url,
                'title':   self.get_title(soup),
                'content': self.get_content(soup),
                'date':    date,
                'author':  self.get_author(soup),
            }
        else:
            return None

    def get_news_soup (self, url):
        try:
            res = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            res.encoding = res.apparent_encoding
            soup = BeautifulSoup(res.text, 'lxml')
            return soup
        except:
            print( 'error in get_news_soup' )
            return None

    def get_title (self, soup):
        try:
            title = soup.find('h1', class_='content_title').get_text()
            return "".join( title.split() )
        except:
            return None

    def get_author (self, soup):
        try:
            author = soup.find('div', class_='content_reporter').find('a').get_text()
            return author
        except:
            return None

    def get_content (self, soup):
        try:
            news_DOM = soup.find('div', {'itemprop': 'articleBody'}).contents
            content = ''
            for DOM in news_DOM:
                if DOM.name == 'p':
                    content += DOM.get_text()
            return "".join( content.split() )[:2000]
        except Exception as e:
            print( 'error in get_content' )
            print(e)
            return None

    def get_url_by_date(self, sub, date):
        flag = True
        url_category = []
        for page in range(1, 10):
            try:
                res  = requests.get('https://newtalk.tw/news/subcategory/%s/%d' % (sub, page), timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                res.encoding = res.apparent_encoding
                soup = BeautifulSoup(res.text, 'lxml')
            except Exception as e:
                print( e )
                print( 'error in get news categoty' )
                continue

            news_category_DOM = soup.find_all('div', class_='news_box1')
            for news_DOM in news_category_DOM:
                try:
                    url = news_DOM.find('div', class_='news-title').find('a')['href']
                    news_date = re.search( r'https://newtalk.tw/news/(.*)/(.*)/(.*)', url ).group(2)

                    if datetime.strptime(news_date, '%Y-%m-%d').date() > date:
                        continue
                    elif datetime.strptime(news_date, '%Y-%m-%d').date() == date:
                        url_category.append( url )
                    else:
                        flag = False
                        break

                except Exception as e:
                    print( 'error in crawling news category' )
                    print( e )
                    continue
            
            if flag == False:
                break

            news_category_DOM = soup.find('div', id='category').find_all('div', class_='news-list-item')
            for news_DOM in news_category_DOM:
                try:
                    url = news_DOM.find('div', class_='news_title').find('a')['href']
                    news_date = re.search( r'https://newtalk.tw/news/(.*)/(.*)/(.*)', url ).group(2)

                    if datetime.strptime(news_date, '%Y-%m-%d').date() > date:
                        continue
                    elif datetime.strptime(news_date, '%Y-%m-%d').date() == date:
                        url_category.append( url )
                    else:
                        flag = False
                        break

                except Exception as e:
                    print( 'error in crawling news gategory' )
                    print( e )
                    continue
        
            if flag == False:
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
                    tmp = New(
                        title=news['title'],
                        content= news['content'],
                        author= news['author'],
                        brand_id=news['brand_id'],
                        sub_id= news['sub_id'],
                        date=news['date'],
                        url=news['url'],
                    )
                    tmp.save()
            except Exception as e:
                print( e )
        return True