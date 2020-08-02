import pandas as pd
import os
import json
import string
import argparse
from progress.bar import ChargingBar
from os import path
import nltk
from time import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import linear_kernel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


RAW_REVIEW_PATH = "./yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json"
RAW_BUSINESS_PATH = "./yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json"

STOP_WORDS = set(stopwords.words('english'))


lemmatizer = WordNetLemmatizer()


def nltk_tag_to_wordnet_tag(nltk_tag):
	if nltk_tag.startswith('J'):
		return wordnet.ADJ
	elif nltk_tag.startswith('V'):
		return wordnet.VERB
	elif nltk_tag.startswith('N'):
		return wordnet.NOUN
	elif nltk_tag.startswith('R'):
		return wordnet.ADV
	else:          
		return None


def clean_sentence(sentence):

	sentence = sentence.translate(str.maketrans('', '', string.punctuation))
	sentence = sentence.lower()

	stop_words = set(stopwords.words('english'))

	word_tokens = nltk.word_tokenize(sentence)

	filtered_sentence = []
	for w in word_tokens:
		w = w.strip()
		if not w in stop_words and len(w) > 3:
			filtered_sentence.append(w)

	filtered_sentence = [w for w in word_tokens if not w in stop_words and len(w) > 3]

	# tokenize the sentence and find the POS tag for each token
	nltk_tagged = nltk.pos_tag(filtered_sentence)

	#tuple of (token, wordnet_tag)
	wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
	lemmatized_sentence = []
	for word, tag in wordnet_tagged:
		if tag is None:
			#if there is no available tag, append the token as is
			lemmatized_sentence.append(word)
		else:        
			#else use the tag to lemmatize the token
			lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

	final = " ".join(lemmatized_sentence)

	return final


def restaurants_for_cuisine(cuisine):

	restaurants = []

	with open(RAW_BUSINESS_PATH) as f:
		all_lines = f.readlines()
		for line in all_lines:
			d = json.loads(line)
			categories = d['categories']
			if len(categories) > 0 and any(s.startswith(cuisine) for s in categories):
				restaurants.append(d['business_id'])

	return restaurants


def all_reviews():
	
	reviews = []

	with open(RAW_REVIEW_PATH) as f:
		all_lines = f.readlines()
		bar = ChargingBar('Reading reviews...', max=len(all_lines))
		for line in all_lines:
			d = json.loads(line)
			if len(d) > 0:
				reviews.append(d)
			bar.next()
		bar.finish()

	return reviews


def reviews_to_csv(reviews):
	
	if len(reviews) > 0:
		
		print("Reading cuisines...")
		c_df = pd.read_csv('cuisines.csv')

		print("----------------------------------------------------------")

		for i, row in c_df.iterrows():
			cuisine_name = row['cuisine_name']
			restaurants = restaurants_for_cuisine(cuisine_name)

			filepath = "./cuisines/" + cuisine_name

			print("{0}/{1} Writing reviews for: {2}".format(i + 1, len(c_df.index), cuisine_name))
			
			columns = ['business_id', 'votes_funny', 'votes_useful', 'votes_cool', 'user_id', 'review_id', 'stars', 'date', 'text', 'cleaned_text']

			data = {
				'business_id' : [],
				'votes_funny' : [],
				'votes_useful' : [],
				'votes_cool' : [],
				'user_id' : [],
				'review_id' : [],
				'stars' : [],
				'date' : [],
				'text' : [],
				'cleaned_text' : []
			}

			if len(restaurants) > 0:
				bar = ChargingBar('Creating CSV', max=len(reviews), suffix='%(index)d / %(max)d | %(percent)d%%')
				for r in reviews:
					business_id = r['business_id']
					if business_id in restaurants:
						data['business_id'].append(business_id)
						data['votes_funny'].append(r['votes']['funny'])
						data['votes_useful'].append(r['votes']['useful'])
						data['votes_cool'].append(r['votes']['cool'])
						data['user_id'].append(r['user_id'])
						data['review_id'].append(r['review_id'])
						data['stars'].append(r['stars'])
						data['date'].append(r['date'])
						data['text'].append(r['text'])
						data['cleaned_text'].append(clean_sentence(r['text']))
					bar.next()
				bar.finish()

			new_df = pd.DataFrame(data, columns=columns)
			new_df.to_csv(filepath + '/reviews.csv', index=False)
			
			print("----------------------------------------------------------")


def reviews():
	
	reviews = all_reviews()

	reviews_to_csv(reviews)


def reviews_to_txt():

	print("Reading cuisines...")
	c_df = pd.read_csv('cuisines.csv')

	print("----------------------------------------------------------")

	for i, row_i in c_df.iterrows():
		cuisine_name = row_i['cuisine_name']

		filepath = './cuisines/' + cuisine_name + '/reviews.csv'

		print("{0}/{1} Processing reviews for: {2}".format(i + 1, len(c_df.index), cuisine_name))

		if path.exists(filepath):
			rev_df = pd.read_csv(filepath)
			all_reviews = ''
			all_cleaned_reviews = ''
			
			bar = ChargingBar('Reading reviews', max=len(rev_df.index), suffix='%(index)d / %(max)d | %(percent)d%%')
			for j, row_j in rev_df.iterrows():
				all_reviews += row_j['text'] + ' '
				all_cleaned_reviews += clean_sentence(row_j['text']) + ' '
				bar.next()
			bar.finish()
			
			write_path = './cuisines/' + cuisine_name + '/'
			
			print('Writing to txt...')

			with open(write_path + 'reviews_text.txt', 'w') as w:
				w.write(all_reviews)

			with open(write_path + 'reviews_cleaned_text.txt', 'w') as w:
				w.write(all_cleaned_reviews)

			print("----------------------------------------------------------")


def write_similarity(cuisine_matrix, cuisines, clusters=[]):
	
	print("Writing files...")

	with open('cuisine_sim_matrix' + '.csv', 'w') as f:
		for i_list in cuisine_matrix:
			s = ""
			my_max = max(i_list)
			
			for tt in i_list:
				s = s + str(tt / my_max) + " "
			s = s.strip()
			
			f.write(",".join(s.split())+"\n") #should the list be converted to m

	with open('cuisine_indices' + '.txt', 'w') as f:
		f.write(",".join(cuisines)+"\n")

	write_similarity_matrix_for_viz(cuisines, clusters)


def write_similarity_matrix_for_viz(cuisines, clusters=[], symmetric=True):

	d = {}
	count = 0
	for c in cuisines:
		d[c] = clusters[count] if len(clusters) > 0 else 0
		count += 1

	new_cat = cuisines

	print("Writing CSV for Viz")

	all_lines = []

	with open('cuisine_sim_matrix.csv', 'r') as f:
		for line in f:
			all_lines.append(line)

	colors = ["#3a86ff", "#8338ec", "#ff006e", "#fb5607", "#ffbe0b"]

	df = pd.DataFrame(columns=["name_x", "name_y", "sim", "color", "cluster"])

	sim_matrix = []
	bar = ChargingBar('Writing', max=len(all_lines))
	for line in all_lines:
		all_sims = line.split(',')
		sim_matrix.append(all_sims)

	count = 0	
	for cat in new_cat:
		cur_cat = cat
		all_sims = sim_matrix[cuisines.index(cat)]
		i = 0
		for i in range(len(all_sims)):
			to_write = {}
			to_write['name_x'] = cur_cat
			to_write['name_y'] = new_cat[i]
			to_write['sim'] = all_sims[cuisines.index(new_cat[i])]
			to_write['cluster'] = d[cur_cat]
			if len(clusters) == 0:
				to_write['color'] = colors[0]
			else:
				col = colors[clusters[count]] if count >= i else colors[clusters[i]]
				to_write['color'] = col #colors[clusters[count]]
			df = df.append(to_write, ignore_index=True)
			i += 1
		count += 1
		bar.next()
	bar.finish()

	df.to_csv('sim_matrix_viz.csv')


def similarity():

	print("Reading cuisines...")
	c_df = pd.read_csv('cuisines.csv')

	print("----------------------------------------------------------")

	reviews = []

	for i, row_i in c_df.iterrows():
		cuisine_name = row_i['cuisine_name']

		filepath = './cuisines/' + cuisine_name + '/reviews_text.txt'

		print("{0}/{1} Processing reviews for: {2}".format(i + 1, len(c_df.index), cuisine_name))

		if path.exists(filepath):
			with open(filepath) as f:
				reviews.append(f.read().replace("\n", " "))
		print("----------------------------------------------------------")


	if len(reviews) > 0:

		idf = True
		max_f = 4000
		K_clusters = 5

		t0 = time()
		print("Extracting features from the training dataset using TfidfVectorizer")

		vectorizer = TfidfVectorizer(min_df=3, max_df=0.5, max_features=max_f, stop_words='english', use_idf=idf)

		tfidf = vectorizer.fit_transform(reviews)
		
		print("done in %fs" % (time() - t0))
		print("n_samples: %d, n_features: %d" % tfidf.shape)

		print('Computing similarity...')

		cosine_similarities = linear_kernel(tfidf, tfidf).flatten()
		print(cosine_similarities.shape)

		shape = len(c_df.index)
		cuisine_matrix = cosine_similarities.reshape((shape, shape))

		print('Clustering cuisines...')
		
		km = KMeans(n_clusters=5)
		km.fit(tfidf)

		clusters = km.labels_.tolist()

		items = {'cluster' : clusters}
		frame = pd.DataFrame(items, index = [clusters] , columns = ['cluster'])
		print(frame['cluster'].value_counts())

		write_similarity(cuisine_matrix, c_df['cuisine_name'].tolist(), clusters=clusters)


def sentiment_scores(sentence): 
  
	# Create a SentimentIntensityAnalyzer object. 
	sid_obj = SentimentIntensityAnalyzer() 
  
	# polarity_scores method of SentimentIntensityAnalyzer 
	# oject gives a sentiment dictionary. 
	# which contains pos, neg, neu, and compound scores. 
	sentiment_dict = sid_obj.polarity_scores(sentence)
	  
	# print("Overall sentiment dictionary is : ", sentiment_dict) 
	# print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative") 
	# print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 
	# print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive") 
  
	# print("Sentence Overall Rated As", end = " ") 
  
	# decide sentiment as positive, negative and neutral 
	# if sentiment_dict['compound'] >= 0.05 :
	# 	print("Positive") 
  
	# elif sentiment_dict['compound'] <= - 0.05 : 
	# 	print("Negative") 
  
	# else : 
	# 	print("Neutral")

	return sentiment_dict


def sentiment():
	
	print("Reading cuisines...")
	c_df = pd.read_csv('cuisines.csv')

	print("----------------------------------------------------------")

	reviews = []

	for i, row_i in c_df.iterrows():
		cuisine_name = row_i['cuisine_name']

		filepath = './cuisines/' + cuisine_name + '/reviews.csv'

		print("{0}/{1} Processing reviews for: {2}".format(i + 1, len(c_df.index), cuisine_name))

		if path.exists(filepath):
			rev_df = pd.read_csv(filepath)

			data = {
			'sentiment_positive' : [],
			'sentiment_neutral' : [],
			'sentiment_negative' : [],
			'sentiment_compound' : [],
			}

			for j, row_j in rev_df.iterrows():
				s = sentiment_scores(row_j['text'])
				data['sentiment_positive'].append(s['pos'])
				data['sentiment_neutral'].append(s['neu'])
				data['sentiment_negative'].append(s['neg'])
				data['sentiment_compound'].append(s['compound'])

			rev_df['sentiment_positive'] = data['sentiment_positive']
			rev_df['sentiment_neutral'] = data['sentiment_neutral']
			rev_df['sentiment_negative'] = data['sentiment_negative']
			rev_df['sentiment_compound'] = data['sentiment_compound']

			rev_df.to_csv(filepath, index=False)

		print("----------------------------------------------------------")


def ratings_over_time()
	pass


def main():
	# Next
	# 1. Find Similarity 
	# 2. Find Dishes
	# 3. Find sentiment rating
	# 4. Distribute reviews
	# 5. checkin
	# 6. tips
	pass


if __name__ == '__main__':
	main()