# local Django
from newsdb.models import New

# third-party
from bs4 import BeautifulSoup
import requests
import pytz

# standard library
from datetime import datetime, date

class TVBSCrawler:

    def __init__(self):
        self.subjects = {
            'local': 1,
            'politics': 2,
            'world':3,
            'sports':5,
            'life': 7,
            'focus': 7,
        }

    def get_news_info (self, url, sub, date):
        soup = self.get_news_soup(url)
        return {
            'brand_id':  1,
            'sub_id':    self.subjects[sub],
            'url':     url,
            'title':   self.get_title(soup),
            'content': self.get_content(soup),
            'date':    date,
            'author':  self.get_author(soup),
        }
    
    def get_news_soup (self, url):
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, 'lxml')
        return soup

    def get_title (self, soup):
        try:
            title = soup.find('h1', class_='margin_b20').get_text()
            return "".join( title.split() )
        except:
            return None
    
    def get_date (self, soup):
        try:
            time_string = soup.find('div', class_='title').find('div', class_='time').get_text()
            date_string = time_string.split(" ")[0]
            return(str(datetime.strptime(date_string, "%Y/%m/%d").date()))
        except:
            return None

    def get_author (self, soup):
        try:
            author = soup.find('div', class_='title').find('h4').find('a').get_text()
            return author
        except:
            return None
    
    def get_content (self, soup):
        try:
            content = ''
            content_DOM = soup.find('div', id='news_detail_div').contents
            
            skip_str = 'htmlPUBLIC"-//W3C//DTDHTML4.0Transitional//EN""http://www.w3.org/TR/REC-html40/loose.dtd'

            for item in content_DOM:
                if item.name is None and 'bs4.element.Doctype' not in str(type(item)): 
                    content += item
                elif item.name == 'p':
                    content += item.get_text()

            return "".join( content.split() )[:2000]
        except Exception as e:
            print(e)
            return None
    
    def get_url_by_date(self, sub, date):
        flag = True
        url_category = []

        res = requests.get('https://news.tvbs.com.tw/%s' % sub, timeout=10)
        soup = BeautifulSoup(res.text, 'lxml')
        news_category = soup.find('ul', id='block_pc').find_all('li')

        for news in news_category:
            news_date = news.select('a div.time')[0].get_text()
            href = news.find('a')['href']
            url  = 'https://news.tvbs.com.tw%s' % href
            if datetime.strptime(news_date, '%Y/%m/%d %H:%M').date() > date:
                continue
            elif datetime.strptime(news_date, '%Y/%m/%d %H:%M').date() == date:
                url_category.append( url )
            else:
                flag = False
                break
        
        return url_category

    def get_news_today( self ):
        timezone   = pytz.timezone('Asia/Taipei')
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