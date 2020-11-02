# local Django
from newsdb.models import New

# third-party
from bs4 import BeautifulSoup
import requests

# standard library
import time
import datetime

class StormCrawler:
    def __init__(self):
        self.textfolder = 'news/'
        self.newsCount = 0
        # 社會(1),政治(2),國際(3),財經(4),體育(5),兩岸(6),生活(7)
        self.urlArray = [
            '',
            'https://www.storm.mg/localarticles/',
            'https://www.storm.mg/category/118/',
            'https://www.storm.mg/category/117/',
            'https://www.storm.mg/category/23083/',
            '',
            'https://www.storm.mg/category/121/',
            'https://www.storm.mg/lifestyles/']

    def getSubjectUrl(self):
        return self.urlArray

    def concateContent(self, htmlArray):
        outputString = ''
        for text in htmlArray:
            outputString += text.text
        return outputString

    def get_news_today(self):
        output = []
        subjectList = self.getSubjectUrl()
        today = datetime.date.today()
        subject_code = -1
        for (sub_id, subject) in enumerate(subjectList):
            subject_code += 1
            if subject == '':
                continue
            # to record if today's news are finished
            flag = False
            for page in range(10):
                page_request = requests.get(subject + str(page))
                soup = BeautifulSoup(page_request.text, 'html.parser')
                sel = soup.select('div.category_card div.card_inner_wrapper a.card_link.link_title')
                time.sleep(0.1)
                for urls in sel:
                    obj = {
                        'title': '',
                        'content': '',
                        'author': '',
                        'brand_id': 13,
                        'sub_id': subject_code,
                        'date': '',
                        'url': urls['href'],
                        'update_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    }
                    time.sleep(0.1)
                    soup = BeautifulSoup(requests.get(urls['href']).text, 'html.parser')
                    timeArray = soup.select('span#info_time')[0].text.split('-')
                    date = datetime.date(int(timeArray[0]), int(timeArray[1]), int(timeArray[2].split(' ')[0]))
                    if date != today:
                        flag = True
                        break

                    obj['title'] = soup.select('h1#article_title')[0].text
                    obj['content'] = self.concateContent(soup.select('div#CMS_wrapper p'))[:2000]
                    obj['author'] = soup.select('span.info_author')[0].text[:15]
                    obj['date'] =  soup.select('div#time_pop_block span#info_time')[0].text.split(' ')[0]
                    # print(obj)
                    output.append(obj)
                    self.newsCount = self.newsCount + 1
                # crawling finished
                if flag:
                    break

        return output

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
