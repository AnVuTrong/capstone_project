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
			