# local Django
from newsdb.models import New

# third-party
from bs4 import BeautifulSoup
import requests
import pytz

# standard library
from datetime import datetime, date

class UpmediaCrawler:

    def __init__(self):
        self.subjects = {
            '1': 1,
            '2': 7,
            '3': 3,
            '24': 7,
            '5': 7,
            '154': 3
        }

    def get_news_info (self, url, sub, date):
        soup = self.get_news_soup(url)
        return {
            'brand_id':  5,
            'sub_id':    self.subjects[sub],
            'url':     url,
            'title':   self.get_title(soup),
            'content': self.get_content(soup)[:2000],
            'date':    date,
            'author':  self.get_author(soup),
        }

    def get_news_soup (self, url):
        res = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(res.text, 'lxml')
        return soup

    def get_title (self, soup):
        try:
            title = soup.find('h2', id='ArticleTitle').get_text()
            return "".join( title.split() )
        except:
            return None

    def get_date (self, soup):
        try:
            date_string = soup.find('div', class_='author').contents[1]
            date_string = "".join( date_string.split() )
            return(str(datetime.strptime(date_string, "%Y年%m月%d日%H:%M:%S").date()))
        except:
            return None

    def get_author (self, soup):
        try:
            author = soup.find('div', class_='author').contents[0].get_text()
            return author
        except:
            return None

    def get_content (self, soup):
        news_DOM = soup.find('div', id='news-info').find('div', class_='editor').find_all('p')
        content = ''
        for DOM in news_DOM:
            content += DOM.get_text()
        return "".join( content.split() )

    def get_url_by_date(self, sub, date):
        flag = True
        url_category = []
        for page in range(1, 20):
            try:
                res  = requests.get('https://www.upmedia.mg/news_list.php?currentPage=%d&Type=%s?' % (page, sub), timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                soup = BeautifulSoup(res.text, 'lxml')
                news_category_DOM = soup.find('div', id='news-list')
                href = news_category_DOM.find('dl', class_='main').find('a')['href']
                news_date = news_category_DOM.select('dl.main dd div.author')[0].contents[1]
                news_date = ''.join(news_date.split())
                if datetime.strptime(news_date, '%Y年%m月%d日%H:%M').date() > date:
                    continue
                elif datetime.strptime(news_date, '%Y年%m月%d日%H:%M').date() == date:
                    url_category.append( 'https://www.upmedia.mg/%s' % href )
                else:
                    flag = False
                    break
            except Exception as e:
                print(e)
                print( 'error in get main news' )     
                continue

            news_category = news_category_DOM.find_all('div', class_='top-dl')
            for news_DOM in news_category:
                try:
                    news_date = news_DOM.find('div', class_='time').get_text()
                    news_date = ''.join(news_date.split())
                    href = news_DOM.find('dt').find('a')['href']

                    if datetime.strptime(news_date, '%Y年%m月%d日%H:%M').date() > date:
                        continue
                    elif datetime.strptime(news_date, '%Y年%m月%d日%H:%M').date() == date:
                        url_category.append( 'https://www.upmedia.mg/%s' % href )
                    else:
                        flag = False
                        break
                except:
                    print( 'error in get news category' )
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