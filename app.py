from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	filename = "https://raw.githubusercontent.com/mabidnadzif/analisis_santimen/main/review_tanpa_preprocessing.csv"
	df = pd.read_csv(filename)
	df.drop(columns=['rumah sakit semarang', 'name'], inplace=True)

	import string
	import re
	def clean_review(review):
		return re.sub('[^a-zA-Z]', ' ', review).lower()

	df['cleaned_review'] = df['review'].apply(lambda x: clean_review(str(x)))
	df['label'] = df['rating'].map({1.0: 0, 2.0: 0, 3.0: 0, 4.0: 1, 5.0: 1})

	def count_punct(review):
		count = sum([1 for char in review if char in string.punctuation])
		return round(count / (len(review) - review.count(" ")), 3) * 100

	df['review_len'] = df['review'].apply(lambda x: len(str(x)) - str(x).count(" "))
	df['punct'] = df['review'].apply(lambda x: count_punct(str(x)))
	df

	def tokenize_review(review):
		tokenized_review = review.split()
		return tokenized_review

	df['tokens'] = df['cleaned_review'].apply(lambda x: tokenize_review(x))
	df.head()

	import nltk
	nltk.download('wordnet')
	nltk.download('omw-1.4')
	nltk.download('stopwords')
	from nltk.corpus import stopwords
	all_stopwords = stopwords.words('english')
	all_stopwords.remove('not')

	def lemmatize_review(token_list):
		return " ".join([lemmatizer.lemmatize(token) for token in token_list if not token in set(all_stopwords)])

	lemmatizer = nltk.stem.WordNetLemmatizer()
	df['lemmatized_review'] = df['tokens'].apply(lambda x: lemmatize_review(x))
	df.head()

	X = df[['lemmatized_review', 'review_len', 'punct']]
	y = df['label']
	print(X.shape)
	print(y.shape)

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
	print(X_train.shape)
	print(X_test.shape)
	print(y_train.shape)
	print(y_test.shape)

	from sklearn.feature_extraction.text import TfidfVectorizer
	tfidf = TfidfVectorizer(max_df=0.5,
							min_df=2)  # ignore terms that occur in more than 50% documents and the ones that occur in less than 2
	tfidf_train = tfidf.fit_transform(X_train['lemmatized_review'])
	tfidf_test = tfidf.transform(X_test['lemmatized_review'])

	from sklearn.feature_extraction.text import CountVectorizer
	cv = CountVectorizer()
	X_cv = cv.fit_transform(df['lemmatized_review'])  # Fit the Data
	y_cv = df['label']

	from sklearn.model_selection import train_test_split
	X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_cv, y_cv, test_size=0.3, random_state=42)

	# Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()

	clf.fit(X_train_cv, y_train_cv)
	clf.score(X_test_cv, y_test_cv)
	





	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
