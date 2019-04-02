import scrapy
from scrapy.http import Request

class TripSpider(scrapy.Spider):

    name = 'tripadvisor'
    allowed_domains = ['tripadvisor.co.uk']
    start_urls = ['https://www.tripadvisor.co.uk/Restaurants-g187069-Manchester_Greater_Manchester_England.html']
    custom_settings = {
       'DOWNLOAD_DELAY': 1,
       # 'DEPTH_LIMIT': 3,
       'AUTOTHROTTLE_TARGET_CONCURRENCY': 0.5,
       'USER_AGENT': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
       # 'DEPTH_PRIORITY': 1,
       # 'SCHEDULER_DISK_QUEUE': 'scrapy.squeues.PickleFifoDiskQueue',
       # 'SCHEDULER_MEMORY_QUEUE': 'scrapy.squeues.FifoMemoryQueue'
    }

    def scrape_review(self, response):
        restaurant_name_review = response.xpath('//div[@class="wrap"]//span[@class="taLnk "]//text()').extract()
        reviewer_name = response.xpath('//div[@class="username mo"]//text()').extract()
        review_rating = response.xpath('//div[@class="wrap"]/div[@class="rating reviewItemInline"]/span[starts-with(@class,"ui_bubble_rating")]').extract()
        review_title = response.xpath('//div[@class="wrap"]//span[@class="noQuotes"]//text()').extract()
        full_reviews = response.xpath('//div[@class="wrap"]/div[@class="prw_rup prw_reviews_text_summary_hsx"]/div[@class="entry"]/p').extract()
        review_date = response.xpath('//div[@class="prw_rup prw_reviews_stay_date_hsx"]/text()[not(parent::script)]').extract()
        restaurant_name = response.xpath('//div[@id="listing_main_sur"]//a[@class="HEADING"]//text()').extract() * len(full_reviews)
        restaurant_rating = response.xpath('//div[@class="userRating"]//@alt').extract() * len(full_reviews)
        restaurant_review_count = response.xpath('//div[@class="userRating"]//a//text()').extract() * len(full_reviews)

        for rvn, rvr, rvt, fr, rd, rn, rr, rvc in zip(reviewer_name, review_rating, review_title, full_reviews, review_date, restaurant_name, restaurant_rating, restaurant_review_count):
            reviews_dict = dict(zip(['reviewer_name', 'review_rating', 'review_title', 'full_reviews', 'review_date', 'restaurant_name', 'restaurant_rating', 'restaurant_review_count'], (rvn, rvr, rvt, fr, rd, rn, rr, rvc)))
            yield reviews_dict
            # print(reviews_dict)

    def parse(self, response):
        ### The parse method is what is actually being repeated / iterated
        for review in self.scrape_review(response):
            yield review
            print(review)

        # access next page of resturants
        next_page_restaurants = response.xpath('//a[@class="nav next rndBtn ui_button primary taLnk"]/@href').extract_first()
        next_page_restaurants_url = response.urljoin(next_page_restaurants)
        yield Request(next_page_restaurants_url)
        # print(next_page_restaurants_url)

        # access next page of reviews
        next_page_reviews = response.xpath('//a[@class="nav next taLnk "]/@href').extract_first()
        next_page_reviews_url = response.urljoin(next_page_reviews)
        yield Request(next_page_reviews_url)
        # print(next_page_reviews_url)

        # access each restaurant page:
        url = response.xpath('//div[@id="EATERY_SEARCH_RESULTS"]/div/div/div/div/a[@target="_blank"]/@href').extract()
        for url_next in url:
            url_full = response.urljoin(url_next)
            yield Request(url_full)

        # "accesses the first review to get to the full reviews (not the truncated versions)"
        first_review = response.xpath('//a[@class="title "]/@href').extract_first() # extract first used as I only want to access one of the links on this page to get down to "review level"
        first_review_full = response.urljoin(first_review)
        yield Request(first_review_full)
        # print(first_review_full)
