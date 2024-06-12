import streamlit as st
import pandas as pd
import plotly.graph_objects as go

class WidgetTest:
	def __init__(self):
		self.data = {
			'latitude' : [51.5074, 48.8566, 40.7128],
			'longitude': [-0.1278, 2.3522, -74.0060]
		}
		self.df = pd.DataFrame(self.data)
		self.fig = go.Figure(data=go.Scatter(x=[1, 2, 3, 4], y=[10, 15, 13, 17]))
		
	
	def gen_testing_widgets(self):
		st.write("You selected Option 1")
		# A slider widget
		st.slider("Slider", 0, 100, 50)
		
		# Test various widgets
		st.write("Checkbox")
		st.checkbox("Checkbox")
		st.write("Radio")
		st.radio("Radio", ["Option 1", "Option 2"])
		st.write("Selectable")
		st.selectbox("Selectable", ["Option 1", "Option 2"])
		st.write("Multiselect")
		st.multiselect("Multiselect", ["Option 1", "Option 2"])
		st.write("Text input")
		st.text_input("Text input")
		st.write("Number input")
		st.number_input("Number input")
		st.write("Text area")
		st.text_area("Text area")
		st.write("Date input")
		st.date_input("Date input")
		st.write("Time input")
		st.time_input("Time input")
		
		# A button widget
		st.button("Button")
		
		# Vietnamese text
		st.write("Tiếng Việt có dấu thì sao?")
		
		# A file uploader widget
		st.file_uploader("File uploader")
		
		# A color picker widget
		st.color_picker("Color picker")
		
		# A progress bar widget
		st.progress(50)
		
		# A spinner widget
		st.spinner()
		
		# A success message
		st.success("Success")
		
		# An info message
		st.info("Info")
		
		# A warning message
		st.warning("Warning")
		
		# An error message
		st.error("Error")
		
		# A exception message
		st.exception("Exception")
		
		# A code block
		st.code("print('Hello, world!')")
		
		# A JSON block
		st.json({"key": "value"})
		
		# A table block
		st.table({"key": "value"})
		
		# A data frame block
		st.dataframe({"key": "value"})
		
		# A line chart
		st.line_chart({"key": "value"})
		
		# A area chart
		st.area_chart({"key": "value"})
		
		# A bar chart
		st.bar_chart({"key": "value"})
		
		# A map
		st.map(self.df)
		
		# A plotly chart
		st.plotly_chart(self.fig)
