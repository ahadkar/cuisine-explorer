import argparse
import restaurants
import cuisines
import reviews


def main():
	
	parser = argparse.ArgumentParser(description='Hygiene Prediction')
	
	parser.add_argument('--cuisines', action='store_true')
	parser.add_argument('--restaurants', action='store_true')
	parser.add_argument('--reviews', action='store_true')
	parser.add_argument('--similarity', action='store_true')
	parser.add_argument('--sentiment', action='store_true')
	parser.add_argument('--all', action='store_true')
	
	args = parser.parse_args()

	if args.cuisines:
		cuisines.create_filesystem()
	elif args.restaurants:
		restaurants.restaurants()
	elif args.reviews:
		reviews.reviews()
	elif args.similarity:
		reviews.similarity()
	elif args.sentiment:
		reviews.sentiment()
	elif args.all:
		cuisines.create_filesystem()
		restaurants.restaurants()
		reviews.reviews()


if __name__ == '__main__':
	main()