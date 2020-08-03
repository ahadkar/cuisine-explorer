import pandas as pd
import os
import json
import argparse
from progress.bar import ChargingBar
from os import path


RAW_BUSINESS_PATH = "./yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json"


def sort():
	
	print("Reading cuisines...")
	df = pd.read_csv("./cuisines.csv")

	print("----------------------------------------------------------")
		
	for i, row in df.iterrows():
		
		cuisine_name = row["cuisine_name"]
		filepath = "./cuisines/" + cuisine_name + "/restaurants.csv"

		if path.exists(filepath):

			print("{0}/{1} Sorting restaurants for: {2}".format(i + 1, len(df.index), cuisine_name))

			rest_df = pd.read_csv(filepath)
			rest_df = rest_df.sort_values(by=['stars'], ascending=False)

			rest_df.to_csv(filepath, index=False)

		print("----------------------------------------------------------")


def all_restaurants(filePath=RAW_BUSINESS_PATH):

	restaurants = []
	count = 0

	r = 'Restaurants'
	with open (filePath, 'r') as f:
		for line in f.readlines():
			business_json = json.loads(line)
			bjc = business_json['categories']
			if len(bjc) > 1 and r in bjc:
				restaurants.append(business_json)
				print("Found {0} restaurants".format(count), end="\r", flush=True)

	return restaurants


def open_close_for_hours(hours):

	time = ""

	if len(hours['open']) > 0:
		time = hours['open']

	if len(hours['close']) > 0:
		time += " - "
		time += hours['close']

	return time


def hours_for_restaurant(hours):

	all_hours = {}

	if len(hours) > 0:
		for key, value in hours.items():
			if key == "Monday":
				all_hours['mon_hours'] = open_close_for_hours(value)
			elif key == "Tuesday":
				all_hours['tue_hours'] = open_close_for_hours(value)
			elif key == "Wednesday":
				all_hours['wed_hours'] = open_close_for_hours(value)
			elif key == "Thursday":
				all_hours['thu_hours'] = open_close_for_hours(value)
			elif key == "Friday":
				all_hours['fri_hours'] = open_close_for_hours(value)
			elif key == "Saturday":
				all_hours['sat_hours'] = open_close_for_hours(value)
			elif key == "Sunday":
				all_hours['sun_hours'] = open_close_for_hours(value)

	return all_hours


def  hours_for_col_name(col_name, hours):
	
	value = ""

	if col_name in hours.keys():
		value = hours[col_name]

	return value


def categories_for_restaurant(categories):

	value = ""

	if len(categories) > 0:
		index = 0
		for c in categories:
			value += c
			if index != len(categories) - 1:
				value += ","
			index += 1

	return value


def  neighborhoods_for_restaurant(neighborhoods):

	value = ""

	if len(neighborhoods) > 0:
		index = 0
		for n in neighborhoods:
			value += n
			if index != len(neighborhoods) - 1:
				value += ","
			index += 1

	return value


def restaurants_to_csv(restaurants):

	print("Reading cuisines...")
	df = pd.read_csv("./cuisines.csv")

	print("----------------------------------------------------------")

	if len(restaurants) > 0:
		
		for i, row in df.iterrows():
			
			cuisine_name = row["cuisine_name"]
			filepath = "./cuisines/" + cuisine_name

			print("{0}/{1} Writing restaurants for: {2}".format(i + 1, len(df.index), cuisine_name))

			new_df = pd.DataFrame(columns= ['business_id', 'full_address', 'mon_hours', 'tue_hours', 'wed_hours', 
											'thu_hours', 'fri_hours', 'sat_hours', 'sun_hours', 'open', 'categories', 
											'city', 'review_count', 'name', 'neighborhoods', 'longitude', 'latitude',
											'state', 'stars'])

			bar = ChargingBar('Creating CSV', max=len(restaurants))
			for r in restaurants:

				categories = categories_for_restaurant(r['categories'])

				if cuisine_name in categories:

					to_write = {}
					to_write['business_id'] = r['business_id']
					to_write['full_address'] = r['full_address']

					hours = hours_for_restaurant(r['hours'])
					to_write['mon_hours'] = hours_for_col_name('mon_hours', hours)
					to_write['tue_hours'] = hours_for_col_name('tue_hours', hours)
					to_write['wed_hours'] = hours_for_col_name('wed_hours', hours)
					to_write['thu_hours'] = hours_for_col_name('thu_hours', hours)
					to_write['fri_hours'] = hours_for_col_name('fri_hours', hours)
					to_write['sat_hours'] = hours_for_col_name('sat_hours', hours)
					to_write['sun_hours'] = hours_for_col_name('sun_hours', hours)

					to_write['open'] = r['open']
					to_write['categories'] = categories
					to_write['city'] = r['city']
					to_write['review_count'] = r['review_count']
					to_write['name'] = r['name']
					to_write['neighborhoods'] = neighborhoods_for_restaurant(r['neighborhoods'])
					to_write['longitude'] = r['longitude']
					to_write['latitude'] = r['latitude']
					to_write['state'] = r['state']
					to_write['stars'] = r['stars']

					bus_filepath = filepath + '/restaurants/'  + r['business_id']
					if not path.exists(bus_filepath):
						os.makedirs(bus_filepath)

					new_df = new_df.append(to_write, ignore_index=True)
				bar.next()
			bar.finish()

			new_df.to_csv(filepath + '/restaurants.csv', index=False)
			print("----------------------------------------------------------")


def restaurants():

	print("Reading raw businesses...")
	restaurants = all_restaurants()

	restaurants_to_csv(restaurants)


def main():
	pass


if __name__ == '__main__':
	main()