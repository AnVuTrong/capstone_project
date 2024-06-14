import random

import streamlit as st
import time

class Widgets:
	def __init__(self):
		self.bar = None
	
	def search_bar(
			self,
			label: str,
			max_chars: int = 200,
			button_text: str = "Button Text",
			result_text: str = "You typed",
	):
		# A Searchbar for the user to input.
		search_query = st.text_input(
			label=label,
			max_chars=max_chars
		)
		
		# A button to get the value of the search bar.
		submit = st.button(label=button_text, use_container_width=True)
		
		# Print the value of the search bar
		if submit:
			st.write(f"{result_text}: {search_query}")
		
		return search_query, submit

	def small_selectbox(
			self,
			label: str,
			options: list,
			index: str = 1,
	):
		# A small selectbox widget
		selected_option = st.selectbox(
			label=label,
			options=options,
			index=index,
		)
		
		return selected_option
	
	def progress_bar(self, progress: int, sleep_time: float = 0.1):
		# A progress bar widget
		if self.bar is None:
			self.bar = st.progress(0)
		else:
			self.bar.progress(progress)
		time.sleep(sleep_time)
		if progress == 100:
			st.success("Success")
	
	def display_courses_to_columns(self, df, n_columns=3, height=350):
		""" Display courses from a dataframe using columns """
		cols = st.columns(n_columns)
		for index, row in df.iterrows():
			col = cols[index % n_columns]
			with col:
				with st.container(border=True, height=height):
					st.markdown(f"##### {row['Course Name']}")
					st.write(":green[By]", f":blue-background[{row['Provider']}]")
					delta = round(random.uniform(0, 1), 1)
					self._change_metric_size()
					st.metric(label="Average rating: ", value=f"⭐{row['Average Rating']}", delta=delta)
					st.text(f"Level: {row['Level']}")
					with st.popover("Description"):
						st.write(row['Description'])
	
	def _change_metric_size(self, size: int = 20):
		st.markdown(
			f"""
	        <style>
	        [data-testid="stMetricValue"] {{
	            font-size: {size}px;
	        }}
	        </style>
	        """,
			unsafe_allow_html=True,
		)