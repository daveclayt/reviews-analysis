# reviews-analysis
Final project at SPICED - Scrapy and Sentiment Analysis on Manchester Restaurant Reviews from TripAdvisor

Scrapy Spider - scrapy_tripadvisor3.py

Jupyter Notebook of data wrangling and findings - final_project.ipynb. Recommended to run in GPU environment, e.g. Google Colab.

train_model.py (and pre-requisite read_clean_data.py) take the output of the scraper, clean it and train LSTM neural network to predict the review ratings from the text, producing saved model and weights files.

model.json and model.h5 are the model and weights files of the results I achieved in my initial implementation (see the Jupyter Notebook file).
