import pandas as pd
import os
import json
import argparse
from progress.bar import ChargingBar
import nltk
import string
from os import path
import numpy as np


def create_filesystem():
	
	print("Reading Cuisines...")
	df = pd.read_csv("./cuisines.csv")

	bar = ChargingBar('Processing', max=df.shape[0])
	for i, row in df.iterrows():
		filepath = "./cuisines/" + row["cuisine_name"]
		if not path.exists(filepath):
			os.makedirs(filepath)
		bar.next()
	bar.finish()


if __name__ == '__main__':
	main()	