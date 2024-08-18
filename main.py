
from sklearn import svm

from sklearn.feature_extraction.text import TfidfVectorizer

input_data = [ ]

# this data is used to train the model to determine whether the sentence is positive or negative, NOTICE** in the second sentence the word 'better'  will be classified as positive
sentences = ["I loved the meal, it was great!", "This place is better than the last.", "The soup was cold and the service was bad.", "My son says the food was good."]

vectorizer = TfidfVectorizer()
input_data = vectorizer.fit_transform(sentences)

# the output data is used in the model to determine whether the sentence is positive or negative, 1 being positive and 0 being negative
output_data = [1, 1, 0, 1]

model = svm.SVC()

model.fit(input_data, output_data)

# this portion tests the model to see if the sentence is positive or negative, when using the training model you will see that specific words have been used to determine the outcome 
test_sentences = [
 "The food I ate is better than the old store.",
 "My dad said good food."
]

# the outcome will be two positive sentences 
print(model.predict(vectorizer.transform(test_sentences)))

