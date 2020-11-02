# local Django
from newsdb.models import New

# third-party
from bs4 import BeautifulSoup
import requests
import pytz

# standard library
from datetime import datetime, date
import re

class CNACrawler:

    def __init__(self):
        self.subjects = {
            'asoc': 1,
            'aipl': 2,
            'aopl': 3,
            'aie': 4,
            'asc': 4,
            'aspt': 5,
            'acn': 6,
            'ait': 7,
            'ahel': 7,
            'aloc': 7,
        }

    def get_news_info (self, url, sub, date):
        soup = self.get_news_soup(url)
        return {
            'brand_id':  7,
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

    def get_author( self, soup ):
        try:
            paragraph_DOM = soup.find('div', class_='centralContent').find('div', class_='paragraph')
            paragrap_string = paragraph_DOM.find_all('p')[0].get_text()
            author_section = re.search( r'（(.*)記者(.*)）', paragrap_string ).group(2)

            return author_section[0:3]
        except:
            return None

    def get_title (self, soup):
        try:
            title = soup.find('div', class_='centralContent').find('h1').get_text()
            return title
        except:
            return None

    def get_content (self, soup):
        try:
            content = soup.find('div', class_='centralContent').find('div', class_='paragraph').get_text()
            return "".join( content.split() )[:2000]
        except:
            return None
    
    def get_url_by_date(self, sub, date):
        flag = True
        url_category = []
        for page in range(1, 20):
            res = requests.get(url = 'https://www.cna.com.tw/cna2018api/api/simplelist/categorycode/%s/pageidx/%d/' % (sub, page))
            news_list = res.json()['result']['SimpleItems']
            for news in news_list:
                if news['IsAd'] == 'N':
                    if datetime.strptime(news['CreateTime'], '%Y/%m/%d %H:%M').date() > date:
                        continue
                    elif datetime.strptime(news['CreateTime'], '%Y/%m/%d %H:%M').date() == date:
                        url_category.append(news['PageUrl'])
                    else:
                        flag = False
                        break
            
            if flag == False:
                break
        
        return url_category
    
    def get_news_today(self):
        timezone = pytz.timezone('Asia/Taipei')
        date_today = datetime.now(timezone).date()

        return self.get_news_by_date( [str(date_today)] )
    
    def get_news_by_date( self, date_list ):
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
