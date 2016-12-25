from django.shortcuts import render
from django.http import HttpResponse


def index(request):
    return render(request, 'index.html')


def generate(request, num):
    return HttpResponse("Sequences %s." % num)
