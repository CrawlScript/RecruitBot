# coding=utf-8

import webcollector as wc
from bs4 import BeautifulSoup

f = open("title.txt", "w", encoding="utf-8")


class RubyCrawler(wc.RamCrawler):
    def __init__(self, **kwargs):
        super().__init__(auto_detect=False, **kwargs)
        self.num_threads = 10
        self.add_seeds(["https://ruby-china.org/jobs?page={}".format(i) for i in range(1, 151)])

    def visit(self, page, detected):
        soup = BeautifulSoup(page.content)

        for a in soup.select("div.title.media-heading > a[title]"):
            title = a["title"].strip()
            # print(title)
            f.write("{}\n".format(title))


crawler = RubyCrawler()
crawler.start(10)