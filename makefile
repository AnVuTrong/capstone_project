# Description: Makefile for the project
run:
	streamlit run app.py

requirements.txt:
	poetry export -f requirements.txt --output requirements.txt --without-hashes