from GUI.modules.pages import (
	homepage,
	about_us,
	collaborative_filtering,
	# how_it_work,
	# customer_segmentation,
	content_base_filtering,
	collaborative_filtering,
	widget_test,
)
import streamlit as st

class PageManager:
	def __init__(self):
		self.homepage = homepage.Homepage()
		self.about_us = about_us.AboutUs()
		self.content_base_filtering = content_base_filtering.ContentBaseFiltering()
		self.collaborate_filtering = collaborative_filtering.CollaborateFiltering()
		# self.customer_segmentation = customer_segmentation.CustomerSegmentation()
		# self.how_it_work = how_it_work.howitwork()
		self.widget_test = widget_test.WidgetTest()
	
	def gen_homepage(self):
		self.homepage.gen_page()
	
	def gen_about_us_page(self):
		self.about_us.gen_page()
	
	# def gen_customer_segmentation_page(self):
	# 	self.customer_segmentation.gen_page()
	
	def gen_recommendation_system_page(self):
		methods = ['Content Base Filtering', 'Collaborate Filtering']
		selected_method = st.selectbox('Select a method:', methods)
		
		if selected_method == 'Content Base Filtering':
			self.content_base_filtering.gen_page()
		elif selected_method == 'Collaborate Filtering':
			self.collaborate_filtering.gen_page()
	
	# def gen_how_it_work_page(self):
	# 	self.how_it_work.gen_page()
	
	def gen_testing_widgets(self):
		self.widget_test.gen_testing_widgets()
