# local Django
from newsdb.models import New

# third-party
from bs4 import BeautifulSoup
import requests
import pytz

# standard library
from datetime import datetime

class EBCCrawler:
    def __init__ (self):
        self.page_num = 2

    def get_news_info (self, url, date):
        soup = self.get_news_soup(url)
        return {
            'brand_id':  14,
            'sub_id':    self.get_subject(url),
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
            temp = soup.select('.fncnews-content')[0]
            title = temp.find_all('h1')[0].get_text()
        except:
            return None
        return title

    def get_author (self, soup):
        try:
            temp = soup.select('.info')[0]
            date_string = temp.select('.small-gray-text')[0].get_text()
            date_list = date_string.split(" ")
        except:
            print( 'author error' )
            return None
            
        if len(date_list) < 5:
            return None
        else:
            return date_list[4]
    
    def get_subject (self, url):
        temp = url.split("/")
        subject = temp[4]
        if('society' in subject):
            return 1
        elif('politics' in subject):
            return 2
        elif('world' in subject ):
            return 3
        elif('business' in subject):
            return 4
        elif('sport' in subject):
            return 5
        elif('china' in subject):
            return 6
        elif('living' or 'story' or 'travel' in subject):
            return 7
        else:
            return 0
    
    def get_content (self, soup):
        try:
            temp = soup.find('span', {"data-reactroot": True})
            temp = temp.find_all('p')
            content = ""
            if len(temp) == 0:
                content = soup.find('span', {"data-reactroot": True}).get_text()
            for node in temp:
                if node.findChild() == None:
                    if len(node.get_text()) > 0:
                        content += node.get_text()
                    elif node.contents != None and len(node.contents) > 0:
                        print(node.contents[0])
                        content += node.contents
            content = "".join(content.split())
            return content[:2000]
        except:
            print('error in get_content')
            return None

    def get_url_by_date(self, date):
        timezone = pytz.timezone('Asia/Taipei')
        date_today = datetime.now(timezone).date()
        flag = True
        url_category = []
        for page in range(1, 21):
            res = requests.get('https://news.ebc.net.tw/Realtime?page=%d' % page, timeout=10)
            soup = BeautifulSoup(res.text, 'lxml')
            news_list_area = soup.select('.news-list-area')[0]
            news_list = news_list_area.select('.white-box')

            for news in news_list:
                if 'list-ad' in news['class']:
                    continue
                
                news_date = news.select('a > div.text > span.small-gray-text')[0].get_text().split(' ')[0]
                news_date = '%s/%s' % (str( date_today.year ), news_date)
                if datetime.strptime(news_date, '%Y/%m/%d').date() > date:
                    continue
                elif datetime.strptime(news_date, '%Y/%m/%d').date() == date:
                    news_url = news.find_all('a')[0]['href']
                    url_category.append ('https://news.ebc.net.tw%s' % news_url)
                else:
                    flag = False
                    break
            
            if flag == False:
                return url_category
        
        return url_category

    def get_news_today( self ):
        timezone = pytz.timezone('Asia/Taipei')
        date_today = datetime.now(timezone).date()

        return self.get_news_by_date( [str(date_today)] )

    def get_news_by_date(self, date_list):
        news_list = []
        for date in date_list:
            url_list = self.get_url_by_date( datetime.strptime(date, '%Y-%m-%d').date() )
            for url in url_list:
                temp_news = self.get_news_info( url, str(date) )
                if temp_news['brand_id'] != 0:
                    news_list.append( temp_news )

        return news_list

    def get_subject_url( self ):
        return ['https://news.ebc.net.tw/realtime']

    def insert_news( self, newsList ):
        for news in newsList:
            try:
                temp_news = New.objects.filter(url=news['url'])
                if len(temp_news) == 0:
                    tmp = New(
                        title=news['title'],
                        content=news['content'],
                        author= news['author'],
                        brand_id=news['brand_id'],
                        sub_id= news['sub_id'],
                        date=news['date'],
                        url=news['url']
                    )
                    tmp.save()
            except Exception as e:
                print( e ) 
                print( news )
        return True