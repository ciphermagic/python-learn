from django.contrib import admin
from . import models


class ArticlsAdmin(admin.ModelAdmin):
    list_display = ('title', 'content', 'pub_time')
    list_filter = ('pub_time',)


admin.site.register(models.Article, ArticlsAdmin)
