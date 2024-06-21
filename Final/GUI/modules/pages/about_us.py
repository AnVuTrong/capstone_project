import streamlit as st


class AboutUs:
	def __init__(self):
		self.header = "About Us"
		self.subheader = '<h3 style="color:#A4C3A2;">Who are we?</h3>'
		self.img_path = "GUI/img/"
		self.vta = f"{self.img_path}vta.png"
		self.lthb = f"{self.img_path}lthb.png"
	
	def gen_page(self):
		""" Introduce myself and team-member """
		st.header(self.header)
		st.divider()
		st.markdown(self.subheader, unsafe_allow_html=True)
		col1, col2 = st.columns(2, gap='large')
		with col1:
			self._portraits(
				img=self.vta,
				name="Vu Trong An",
				description="I am a student at the CSC. I am passionate about data science and machine learning."
			)
		with col2:
			self._portraits(
				img=self.lthb,
				name="Le Thi Hai Binh",
				description="I am a lecturer at IUH. I'm studying data science and machine learning to support my research."
			)
	
	def _portraits(self, img, name, description):
		# with st.container(border=True, height=600):
			name = f"<h3 style='color:#5D7B6F;'>{name}</h3>"
			st.image(img)
			st.write("")
			st.markdown(name, unsafe_allow_html=True)
			st.write(
				description
			)
			click = st.button("Contact", key=name, use_container_width=True)
			if click:
				st.warning("No contact information available.")
				