# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from glob import glob

path_to_html = "crawled_pages/"

df = open("crawling_paths.csv").readlines()[:1]

def clean_elem( elem ):
	return elem.replace("\n"," ").strip()

# print header
print( "\t".join( [ "fname", "views", "date_posted", "likes", "dislikes", "channel_name", "channel_subscribers" ] ) )

for fname in glob( path_to_html + "*" ):
	try:
		soup = BeautifulSoup( open( fname ).read(), 'html.parser' )

		views = clean_elem( soup.select("#info-text .view-count")[0].getText() )

		channel_name = clean_elem( soup.select("#meta .ytd-channel-name")[0].getText() )
		channel_subscribers = clean_elem( soup.select("#owner-sub-count")[0].getText() )
		
		likes = clean_elem( soup.select("#segmented-like-button")[0].getText() )
	
		# print results in tab-seperated format
		print( "\t".join( [ fname, views, likes, channel_name, channel_subscribers ] ) )

	except Exception as e:
		print( "Unable to process", fname )