from django.shortcuts import render
from django.http import HttpResponse


from disindb.models import Brand, Report
from disindb.serializers import ReportSerializer

from datetime import date

## Crawler
from crawler.apis import pts_crawling, cts_crawling, ltn_crawling, nowNews_crawling




def test_pts_crawler(request):
    c = pts_crawling()
    data = c.getNews(date=['2020-09-18'])

    errors = []
    valided_data = []
    for j in data:
        r = ReportSerializer(data=j)
        try:
            if not r.is_valid():
                raise ValueError
            valided_data.append(j)
        except ValueError:
            errors.append({'error': r.errors, 'data' : r.data})
    # result = c.insertNews(valided_data)
    return HttpResponse([errors, len(data)])


def test_cts_crawler(request):
    c = cts_crawling()
    data = c.getNews(date=['2020-09-18'])

    errors = []
    valided_data = []
    for j in data:
        r = ReportSerializer(data=j)
        try:
            if not r.is_valid():
                raise ValueError
            valided_data.append(j)
        except ValueError:
            errors.append({'error': r.errors, 'data' : r.data})
    # result = c.insertNews(valided_data)
    return HttpResponse([errors, data])

def test_ltn_crawler(request):
    c = ltn_crawling()
    data = c.getNews(date=['2020-09-18'])

    errors = []
    valided_data = []
    for j in data:
        r = ReportSerializer(data=j)
        try:
            if not r.is_valid():
                raise ValueError
            valided_data.append(j)
        except ValueError:
            errors.append({'error': r.errors, 'data' : r.data})
    # result = c.insertNews(valided_data)
    return HttpResponse([errors, data])

def test_nowNews_crawler(request):
    c = nowNews_crawling()
    data = c.getNews(date=['2020-09-18'])

    errors = []
    valided_data = []
    for j in data:
        r = ReportSerializer(data=j)
        try:
            if not r.is_valid():
                raise ValueError
            valided_data.append(j)
        except ValueError:
            errors.append({'error': r.errors, 'data' : r.data})
    # result = c.insertNews(valided_data)
    return HttpResponse([errors, data])