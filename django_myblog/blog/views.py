from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from . import models
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
)


def index(request):
    articles = models.Article.objects.all()
    return render(request, 'blog/index.html', {'articles': articles})


def article_page(request, article_id):
    article = models.Article.objects.get(pk=article_id)
    return render(request, 'blog/article_page.html', {'article': article})


def edit_page(request, article_id):
    if str(article_id) == '0':
        return render(request, 'blog/edit_page.html')
    article = models.Article.objects.get(pk=article_id)
    return render(request, 'blog/edit_page.html', {'article': article})


def edit_action(request):
    article_id = request.POST['article_id']
    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" + article_id)
    title = request.POST['title']
    content = request.POST['content']
    if str(article_id) == '0':
        models.Article.objects.create(title=title, content=content)
        return HttpResponseRedirect("/blog/index/")
    article = models.Article.objects.get(pk=article_id)
    article.title = title
    article.content = content
    article.save()
    return HttpResponseRedirect("/blog/index/")
