import pandas as pd
import re
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


#load tweet data
df= pd.read_csv("Tweets.csv")
df.head()

#DELETE ALL UNWANTED COLUMNS
del df['tweet_id']
del df['airline_sentiment_gold']
del df['negativereason_gold']
del df['airline_sentiment_confidence']
del df['negativereason']
del df['negativereason_confidence']
del df['airline']
del df['name']
del df['retweet_count']
del df['tweet_coord']
del df['tweet_created']
del df['tweet_location']
del df['user_timezone']

df.head()


#CHECK THE SHAPE OF THE DATA
print("Shape of the dataframe is",df.shape)
# print("The number of nulls in each column are \n", df.isna().sum())


#PRINT FIRST FIVE ROWS OF DATA
print("First Five rows",df[:5])


# DATA PREPROCESSING

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in text]

#REMOVE HTML TAG,TOKENIATION,REMOVE NUMBERS,E.T.C
def text_processor(text):
    #REMOVE HTML TAG
    text = re.sub(re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});'), '', text)

    #TOKENIZE
    text = word_tokenize(text)
    word_text_string = ""
    for tokenized_text in text:

        #REMOVE NUMBERS
        tokenized_text = re.sub("[^a-zA-Z]", " ",tokenized_text)

        #CONVERT TO LOWERCASE
        tokenized_text = tokenized_text.lower()


        #LEMMATIZE
        real_words = lemmatize_text(tokenized_text)

        #JOIN THE WORDS IN A LIST TO CONVERT BACK TO TEXT STRING
        word_text_string += " ".join( real_words )

    #RETURN THE CONVERTED WORD
    return word_text_string


# create a new column in the dataframe and append the formated text for each tweet text to it
df['final_tweet_text'] = df['text'].apply(lambda x: text_processor(x))

# so we have an extra column "Final_tweet_text" that contains the cleaned data to be used in the program


# PRINT FIRST 5 ROWS after pre-processing
print("FISRT FIVE ROWS AFTER PRE-PROCESSING ", df[:5])


# spliting our data into test and train data

#please note sir i used....70% train and 30% test
train,test = train_test_split(df,test_size=0.3,random_state=40)


#get train and test tweet from the column final_tweet_text and append it to a list defined for each

train_tweets=[] #train list
for tweet in train['final_tweet_text']:
    train_tweets.append(tweet)


test_tweets=[] #test list
for tweet in test['final_tweet_text']:
    test_tweets.append(tweet)


#USING COUNTVECTORIZER
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
count_vectorizer = CountVectorizer(analyzer = "word")
trained_count_vectorizer= count_vectorizer.fit_transform(train_tweets)
test_count_vectorizer=count_vectorizer.transform(test_tweets)


#USING TFID VECTORIER
tfid_vectorizer = TfidfVectorizer()
trained_tfid_vectorizer= tfid_vectorizer.fit_transform(train_tweets)
test_tfid_vectorizer=tfid_vectorizer.transform(test_tweets)






#FIT AND EVALUATE MODEL FOR BOTH TYPE OF VECTORIATION
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#I will be using RandomFOrestClassifier model......you can change this to whatever model you want sir
classifier = RandomForestClassifier(n_estimators=200)




test_count_vectorizer_densed_array =test_count_vectorizer.toarray()
train_count_vectorizer_densed_array =trained_count_vectorizer.toarray()


test_tfidf_vectorizer_densed_array =test_tfid_vectorizer.toarray()
train_tfidf_vectorizer_densed_array =trained_tfid_vectorizer.toarray()



#FOR COUNT VECTORIZER
try:
    classifier_model = classifier.fit(trained_count_vectorizer,train['airline_sentiment'])
    print(classifier_model.evaluate(trained_count_vectorizer,train['airline_sentiment']))
    pred = classifier_model.predict(test_count_vectorizer)
except Exception:
    classifier_model = classifier.fit(train_count_vectorizer_densed_array,train['airline_sentiment'])
    pred = classifier_model.predict(test_count_vectorizer_densed_array)
accuracy = accuracy_score(pred,test['airline_sentiment'])
print("COUNT VECTORIZER ACCURACY USING RANDOMFORESTMODEL: ",str(accuracy))




#FOR TFIDF
try:
    classifier_model = classifier.fit(trained_tfid_vectorizer,train['airline_sentiment'])
    print(classifier_model.evaluate(trained_tfid_vectorizer,train['airline_sentiment']))
    pred = classifier_model.predict(test_tfid_vectorizer)
except Exception:
    classifier_model = classifier.fit(train_tfidf_vectorizer_densed_array,train['airline_sentiment'])
    pred = classifier_model.predict(test_tfidf_vectorizer_densed_array)
accuracy = accuracy_score(pred,test['airline_sentiment'])
print("TFIDI VECTORIZER ACCURACY USING RANDOMFORESTMODEL: ",str(accuracy))