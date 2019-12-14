import numpy as np
import pandas as pd
import time
import datetime
from sklearn import preprocessing
import geopy.geocoders
from geopy.geocoders import Nominatim
import pickle

def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=1):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


def sampling_k_elements(group, k=344):
    if len(group) < k:
        return group
    return group.sample(k)

# Create new feature that transforms region into city importance, gps coordinates
def get_region(data, restart = False):
	if restart:
		city_info = {}
		state_info = {}
	else:
		city_info = pickle.load(open('./data/city_info', 'rb')) # 'city, state' -> (importance, lon, lat)
		state_info = pickle.load(open('./data/state_info', 'rb')) # 'city, state' -> (importance, lon, lat)

	# CITIES
	importances = []
	longitudes = []
	lattitudes = []

	num_iter = 0
	num_rows = len(data.index)

	for index, row in data.iterrows():
		num_iter += 1
		address = str(row['state_code']).lower()
		print('\n' + str(num_iter) + ' out of ' + str(num_rows) + ': ' + address)
		if address not in state_info:
			# Obtaining location info
			geolocator = Nominatim(timeout = 3)
			location = geolocator.geocode(address, addressdetails = True)

			if location is not None:
				importance = location.raw['importance']
				longitude = location.raw['lon']
				lattitude = location.raw['lon']
			else:
				importance = np.nan
				longitude = np.nan
				lattitude = np.nan

			# Adding location info to dictionary for future use
			state_info[address] = (importance, longitude, lattitude)
			with open('./data/state_info', 'wb') as handle:
				pickle.dump(state_info, handle, protocol=pickle.HIGHEST_PROTOCOL) # Saving for backup

			# Debugging output
			print('state not in dict')
			print(importance)
			print(longitude)
			print(lattitude)

		else:
			(importance, longitude, lattitude) = state_info[address]

			# Adding location info to dataframe
		importances.append(importance)
		longitudes.append(longitude)
		lattitudes.append(lattitude)		

	data['state_importance'] = importances
	data['state_longitude'] = longitudes
	data['state_lattitude'] = lattitudes

	# CITIES
	importances = []
	longitudes = []
	lattitudes = []

	num_iter = 0
	num_rows = len(data.index)

	for index, row in data.iterrows():
		num_iter += 1
		address = str(row['city']).lower() + ', ' + str(row['state_code']).lower()
		print('\n' + str(num_iter) + ' out of ' + str(num_rows) + ': ' + address)
		if address not in city_info:
			# Obtaining location info
			geolocator = Nominatim(timeout = 3)
			location = geolocator.geocode(address, addressdetails = True)

			if location is not None:
				importance = location.raw['importance']
				longitude = location.raw['lon']
				lattitude = location.raw['lon']
			else:
				importance = np.nan
				longitude = np.nan
				lattitude = np.nan

			# Adding location info to dictionary for future use
			city_info[address] = (importance, longitude, lattitude)
			with open('./data/city_info', 'wb') as handle:
				pickle.dump(city_info, handle, protocol=pickle.HIGHEST_PROTOCOL) # Saving for backup

			# Debugging output
			print('city not in dict')
			print(importance)
			print(longitude)
			print(lattitude)

		else:
			(importance, longitude, lattitude) = city_info[address]

			# Adding location info to dataframe
		importances.append(importance)
		longitudes.append(longitude)
		lattitudes.append(lattitude)		

	data['city_importance'] = importances
	data['city_longitude'] = longitudes
	data['city_lattitude'] = lattitudes

	return data

def get_data(csv_path, ignored_preds = ['founded_quarter', 'country_code', 'category_code', 'region', 'city', 'state_code', 'name', 'permalink'], scale = True, preprocess = True):
	if preprocess:
		data = pd.read_csv(csv_path, encoding = "ISO-8859-1", na_values = [''])
		# data = get_region(data)
		data = pd.read_pickle('./data/data_with_region.csv')

		#Removing useless columns:
		date_cols = ['founded_month', 'founded_at', 'first_funding_at', 'last_funding_at', 'last_milestone_at']

		good_preds_only = False
		if good_preds_only:
			ignored_preds.extend(['founded_year', 'founded_month', 'first_funding_at', 'founded_at', 'last_milestone_at', 'last_funding_at'])

		data = data.drop(columns = ignored_preds)
		#Transforming dates to floats:
		for col in date_cols:
			if col not in ignored_preds:
				datetimes = pd.to_datetime(data.loc[:, col])
				for i in range(data.shape[0]):
					if (datetimes[i] != datetimes[i]):
						datetimes[i] = datetimes[i]
					else:
						datetimes[i] = datetimes[i].timestamp()
				data[col] = datetimes
		# data = pd.to_numeric(data, errors = 'coerce')

		for col in data.columns:
			if col != 'status':
				 new_col = data[col].apply(pd.to_numeric, errors='coerce')
				 new_col[np.isfinite(new_col) == False] = np.mean(new_col)
				 if scale:
				 	new_col = preprocessing.scale(new_col)
				 data.loc[:, col] = new_col
			# else:
			# 	for i in range(data.shape[0]):
			# 		data.loc[i, col] = 0

		# print(data.columns.dtype)

		# print(data)
		data.to_pickle('./data/processed_data.csv')
	
	data = pd.read_pickle('./data/processed_data.csv')
	return train_validate_test_split(data)