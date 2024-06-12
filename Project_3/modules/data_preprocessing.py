"""
In this module, we will implement the data preprocessing steps for the project.
"""

# Import necessary libraries
import logging
import os
import re
from typing import List

import numpy as np
from underthesea import sent_tokenize, text_normalize, word_tokenize
import pandas as pd
from tqdm import tqdm
from Project_3.modules.VietnameseOcrCorrection.tool.predictor import Predictor

tqdm.pandas()


class DataPreprocessing:
	def __init__(self,
	             data_path_1=None,
	             data_path_2=None,
	             preprocessed_data_path=None,
	             ready_data_path=None,
	             emoji_dict_path='data/files/unicode_all_emojis_vi.txt',
	             teen_dict_path='data/files/teencode.txt',
	             english_dict_path='data/files/english-vnmese.txt',
	             stopwords_path='data/files/vietnamese-stopwords.txt',
	             corrector_model_path='modules/VietnameseOcrCorrection/weights/seq2seq_0.pth'
	             ):
		# Set the paths
		self.data_path_1 = data_path_1
		self.data_path_2 = data_path_2
		self.preprocessed_data_path = preprocessed_data_path
		self.ready_data_path = ready_data_path
		self.corrector_model_path = corrector_model_path
		
		# Setup files
		self.emoji_dict = self.clean_dict(self.dict_handler(emoji_dict_path))
		self.teen_dict = self.dict_handler(teen_dict_path)
		self.english_dict = self.dict_handler(english_dict_path)
		self.stopwords_lst = self.list_handler(stopwords_path)
		
		# Initialize the Vietnamese OCR Correction model
		self.predictor = Predictor(
			device='cuda',
			model_type='seq2seq',
			weight_path=self.corrector_model_path,
		)
	
	def clean_dict(self, dictionary):
		cleaned_dict = {}
		for key, value in dictionary.items():
			cleaned_key = key.replace(':', '')
			cleaned_value = value.replace(':', '')
			cleaned_dict[cleaned_key] = cleaned_value
		return cleaned_dict
	
	def dict_handler(self, file_path):
		"""This function reads the dictionary files."""
		with open(file_path, 'r', encoding='utf-8') as file:
			lines = file.read().splitlines()
		dictionary = {}
		for line in lines:
			parts = line.split('\t')
			if len(parts) < 2:
				parts = line.split(':', 1)
			if len(parts) >= 2:
				key, value = parts[0], parts[1]
				dictionary[key.strip()] = value.strip()
		return dictionary
	
	def list_handler(self, file_path):
		"""This function reads the list files."""
		with open(file_path, 'r', encoding='utf-8') as file:
			lst = file.read().splitlines()
		return lst
	
	def load_raw_data(self):
		data_1 = pd.read_csv(self.data_path_1)
		data_2 = pd.read_csv(self.data_path_2)
		logging.info("Raw data loaded successfully.")
		return data_1, data_2
	
	def load_preprocessed_data(self):
		if os.path.exists(self.ready_data_path):
			df = pd.read_csv(self.ready_data_path)
			logging.info("Ready data loaded successfully.")
			return df
		else:
			return self.preprocess_data_shopee_food()
	
	def data_preparation(self):
		df = self.load_preprocessed_data()
		if not os.path.exists(self.ready_data_path):
			df = self.vnese_text_column_handler(df)
			df.to_csv(self.ready_data_path, index=False)
			logging.info("Ready data saved successfully.")
		else:
			logging.info("Ready data already exists.")
			df = pd.read_csv(self.ready_data_path)
		return df
	
	def preprocess_data_shopee_food(self):
		if os.path.exists(self.preprocessed_data_path):
			df = pd.read_csv(self.preprocessed_data_path)
			logging.info("Preprocessed data loaded successfully.")
			return df
		else:
			df_restaurants, df_reviews = self.load_raw_data()
		
		df_restaurants.rename(columns={'ID': 'IDRestaurant', 'Time': 'OpeningTime'}, inplace=True)
		df_reviews.rename(columns={'Time': 'ReviewTime'}, inplace=True)
		
		df_restaurants, df_reviews = self.miss_aligned_ids_handler(df_restaurants, df_reviews)
		
		df = pd.merge(df_reviews, df_restaurants, on='IDRestaurant', how='left')
		df['ReviewCount'] = df['IDRestaurant'].map(df['IDRestaurant'].value_counts())
		df = self.time_column_handler(df)
		df = self.price_range_handler(df)
		
		df = df.dropna(subset=['Comment'])
		df = df[df['Comment'].str.len() > 10]
		df = self.remove_duplicates(df)
		
		if not os.path.exists(self.preprocessed_data_path):
			df.to_csv(self.preprocessed_data_path, index=False)
			logging.info("Preprocessed data saved successfully.")
		
		return df
	
	def vnese_text_column_handler(self, df):
		df = self.lowercase_text(df)
		df = self.replace_english_words(df)
		df = self.handle_emojis(df)
		df = self.handle_teencode(df)
		df = self.remove_weird_characters(df)
		df = self.handle_wrong_words(df)
		df = self.tokenize_text(df)
		df = self.remove_stopwords(df, use_default=False) # This step is optional
		df = self.remove_all_punctuation(df)
		df = df.dropna(subset=['Comment'])
		df = df[df['Comment'].isna() == False]
		
		print("Vietnamese text column handled successfully.")
		
		return df
	
	def remove_duplicates(self, df):
		# If there are duplicate reviews, keep only the first one.
		# These should have the same 'Comment', 'IDRestaurant', 'Rating', 'User', 'Restaurant', 'Address'.
		df.drop_duplicates(
			subset=['Comment', 'IDRestaurant', 'Rating', 'User', 'Restaurant', 'Address'],
			keep='first',
			inplace=True
		)
		
		return df
	
	def lowercase_text(self, df):
		df['Comment'] = df['Comment'].astype(str).str.lower()
		return df
	
	def remove_weird_characters(self, df):
		# Define the Vietnamese characters range, including accented characters
		vietnamese_chars = (
			"a-zA-Zàáạảãâầấậẩẫăằắặẳẵ"
			"èéẹẻẽêềếệểễ"
			"ìíịỉĩ"
			"òóọỏõôồốộổỗơờớợởỡ"
			"ùúụủũưừứựửữ"
			"ỳýỵỷỹ"
			"đ"
			"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ"
			"ÈÉẸẺẼÊỀẾỆỂỄ"
			"ÌÍỊỈĨ"
			"ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
			"ÙÚỤỦŨƯỪỨỰỬỮ"
			"ỲÝỴỶỸ"
			"Đ"
		)
		valid_chars_pattern = re.compile(f"[{vietnamese_chars}0-9\s]+")
		
		def remove_weird_characters_from_text(text):
			# Remove all characters that are not in the valid Vietnamese characters set
			return ''.join(re.findall(valid_chars_pattern, text))
		
		# Remove weird characters for the 'Comment' column
		print("Removing weird characters from the 'Comment' column...")
		df['Comment'] = df['Comment'].progress_apply(lambda x: remove_weird_characters_from_text(str(x)))
		
		return df
	
	def replace_english_words(self, df):
		def replace_english_words_in_text(text):
			words = text.split()
			for i, word in enumerate(words):
				if word in self.english_dict:
					words[i] = self.english_dict[word]
			return ' '.join(words)
		
		# Replace English words for the 'Comment' column
		print("Replacing English words in the 'Comment' column...")
		df['Comment'] = df['Comment'].progress_apply(lambda x: replace_english_words_in_text(str(x)))
		
		return df
	
	def handle_emojis(self, df):
		# Create a single regex pattern to match any emoji in the dictionary
		emoji_pattern = re.compile('|'.join(re.escape(emoji) for emoji in self.emoji_dict.keys()))
		
		# Function to replace matched emojis with their corresponding values and ensure whitespace.
		def replace_emojis_in_text(text):
			def replace_with_space(match):
				emoji = match.group(0)
				replacement = self.emoji_dict[emoji]
				# Check if the emoji is surrounded by non-whitespace characters and add spaces if needed
				left_space = ' ' if match.start() > 0 and text[match.start() - 1] not in ' \t\n' else ''
				right_space = ' ' if match.end() < len(text) and text[match.end()] not in ' \t\n' else ''
				return left_space + replacement + right_space
			
			return emoji_pattern.sub(replace_with_space, text)
		
		# Replace emojis for the 'Comment' column
		print("Replacing emojis in the 'Comment' column...")
		df['Comment'] = df['Comment'].progress_apply(lambda x: replace_emojis_in_text(str(x)))
		
		return df
	
	def handle_teencode(self, df):
		def replace_teencode_in_text(text):
			words = text.split()
			for i, word in enumerate(words):
				if word in self.teen_dict:
					words[i] = self.teen_dict[word]
			return ' '.join(words)
		
		# Replace teencode for the 'Comment' column
		print("Replacing teencode in the 'Comment' column...")
		df['Comment'] = df['Comment'].progress_apply(lambda x: replace_teencode_in_text(str(x)))
		
		return df
	
	def handle_wrong_words(self, df):
		def correct_vietnamese_text(paragraph):
			try:
				output = str(self.predictor.predict(paragraph.strip(), NGRAM=5))
			except Exception as e:
				print(f'paragraph cannot be processed: {paragraph}')
				print(f"It will be dropped. Error: {e}")
				return None
			return output
		
		# Correct wrong words for the 'Comment' column
		print("Correcting wrong words in the 'Comment' column...")
		df['Comment'] = df['Comment'].progress_apply(lambda x: correct_vietnamese_text(str(x)))
		
		# Remove rows with None values in the 'Comment' column, reset the index.
		df = df.dropna(subset=['Comment']).reset_index(drop=True)
		
		return df
	
	def tokenize_text(self, df):
		def tokenize_vi(text: str) -> List[List[str]]:
			# Step 1: Sentence Tokenization
			sentences = sent_tokenize(text)
			
			# Step 2: Text Normalization
			sentences = [text_normalize(sentence) for sentence in sentences]
			
			# Step 3: Word Tokenization
			tokenized_sentences = [word_tokenize(word_tokenize(sentence, format="text")) for sentence in sentences]
			
			# Flatten the list
			tokenized_sentences = [word for sentence in tokenized_sentences for word in sentence]
			
			# Lowercase all tokens
			tokenized_sentences = [word.lower() for word in tokenized_sentences]
			
			return tokenized_sentences
		
		print("Tokenizing the 'Comment' column...")
		df['Comment'] = df['Comment'].progress_apply(lambda x: ' '.join(tokenize_vi(x)))
		
		return df
	
	def remove_stopwords(self, df, use_default=True):
		# Add some common Vietnamese stopwords to the list
		if not use_default:
			self.stopwords_lst = []
		self.stopwords_lst.extend([
			'nhân viên', 'ở đây', 'phục vụ', 'không gian', 'quán ăn', 'quán', 'nhà hàng', 'đồ ăn', 'mình ăn', 'món ăn',
			'nói chung', 'quán này', 'mình gọi', 'các bạn', 'mình thấy', 'mà', 'của mình', 'nào cũng', 'mình cũng',
			'nhà hàng', 'gọi', 'và', 'có', 'nếu', 'được', 'đến', 'đi', 'là', 'ở', 'rất', 'quá', 'vì', 'để', 'điểm',
		])
		
		def remove_stopwords_from_text(text):
			words = text.split()
			words = [word for word in words if word not in self.stopwords_lst]
			return ' '.join(words)
		
		# Remove stopwords for the 'Comment' column
		print("Removing stopwords from the 'Comment' column...")
		df['Comment'] = df['Comment'].progress_apply(lambda x: remove_stopwords_from_text(str(x)))
		
		return df
	
	def remove_all_punctuation(self, df):
		def remove_punctuation(text):
			# Remove all punctuation except underscores
			return re.sub(r'[^\w\s_]', '', text)
		
		# Remove punctuation for the 'Comment' column
		print("Removing punctuation from the 'Comment' column...")
		df['Comment'] = df['Comment'].progress_apply(lambda x: remove_punctuation(str(x)))
		
		return df
	
	def miss_aligned_ids_handler(self, df1, df2):
		duplicates = df1[df1.duplicated(subset=['Restaurant', 'Address'], keep=False)]
		id_map = {}
		for _, group in duplicates.groupby(['Restaurant', 'Address']):
			first_id = group['IDRestaurant'].iloc[0]
			for id_ in group['IDRestaurant']:
				id_map[id_] = first_id
		df2['IDRestaurant'] = df2['IDRestaurant'].replace(id_map)
		df1.drop_duplicates(subset=['Restaurant', 'Address'], keep='first', inplace=True)
		return df1, df2
	
	def time_column_handler(self, df):
		df['ReviewTime'] = pd.to_datetime(df['ReviewTime'], format='%d/%m/%Y %H:%M')
		df['ClosingTime'] = df['OpeningTime'].str.split(' - ').str[1]
		df['OpeningTime'] = df['OpeningTime'].str.split(' - ').str[0]
		return df
	
	def price_range_handler(self, df):
		df['MaxPrice'] = df['Price'].str.split(' - ').str[1].str.replace('.', '').astype(float)
		df['MinPrice'] = df['Price'].str.split(' - ').str[0].str.replace('.', '').astype(float)
		df.drop('Price', axis=1, inplace=True)
		return df
	
	def preprocess_predictions(self, comments: List[str]) -> List[str]:
		# Create a DataFrame from the comments
		df = pd.DataFrame(comments, columns=['Comment'])
		
		# Preprocess the comments
		df = self.vnese_text_column_handler(df)
		output = df['Comment'].tolist()
		print(output)
		
		return df['Comment'].tolist()


if __name__ == '__main__':
	data_path_1 = '../data/1_Restaurants.csv'
	data_path_2 = '../data/2_Reviews.csv'
	preprocessed_data_path = '../data/preprocessed_data.csv'
	ready_data_path = '../data/ready_data.csv'
	emoji_dict_path = '../data/files/unicode_all_emojis_vi.txt'
	teen_dict_path = '../data/files/teencode.txt'
	english_dict_path = '../data/files/english-vnmese.txt'
	stopwords_path = '../data/files/vietnamese-stopwords.txt'
	corrector_model_path = 'VietnameseOcrCorrection/weights/seq2seq_0.pth'
	
	data_preprocessing = DataPreprocessing(
		data_path_1, data_path_2, preprocessed_data_path, ready_data_path,
		emoji_dict_path, teen_dict_path, english_dict_path, stopwords_path,
		corrector_model_path,
	)
	data_preprocessing.data_preparation()
