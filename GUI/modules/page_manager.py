from GUI.modules.pages import (
	homepage,
	content_base_filtering,
	collaborative_filtering,
	widget_test,
)


class PageManager:
	def __init__(self):
		self.homepage = homepage.Homepage()
		self.content_base_filtering = content_base_filtering.ContentBaseFiltering()
		self.collaborate_filtering = collaborative_filtering.CollaborateFiltering()
		self.widget_test = widget_test.WidgetTest()
	
	def gen_homepage(self):
		self.homepage.gen_page()
	
	def gen_content_base_filtering_page(self):
		self.content_base_filtering.gen_page()
	
	def gen_collaborative_filtering_page(self):
		self.collaborate_filtering.gen_page()
	
	def gen_testing_widgets(self):
		self.widget_test.gen_testing_widgets()
