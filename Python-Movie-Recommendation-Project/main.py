
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
import http.client
import json
import requests

#For you to do
APIKey = "hidden"

#For you to do:
#You're gonna need some lists to store movie titles
#Remember you'll need a list of positive, negative, and some to be tested
good_movies = [ 
  "Parasite",
  "Joker",
  "1917",
  "Inception"
]
bad_movies = [ 
  "Crazy Rich Asians",
  "Cats",
  "The Last Airbender",
  "Dolittle"
]
test_movies = [ 
  "Uncut Gems",
  "Jojo Rabbit",
  "Toy Story 4",
  "Venom"
]
#For you to do:
#You're gonna need some lists to store the movie features(overviews)
good_overview = [ ]
bad_overview = [ ]
test_overview = [ ]

#The following lines of code calls the API and gives me a JSON object.
def movieplot(movieName):
  httpRequest = "https://api.themoviedb.org/3/search/movie?include_adult=false&page=1&query="+movieName+"&language=en-US&api_key="+APIKey
  response = requests.get(httpRequest)
  data = response.json()
  plot = data["results"][0]["overview"]
  return plot

for title in good_movies:
  overview = movieplot(title)
  good_overview.append(overview)
for title in bad_movies:
  overview = movieplot(title)
  bad_overview.append(overview)
for title in test_movies:
  overview = movieplot(title)
  test_overview.append(overview)
#For you to do
''' 
This line gives me the overview of the first movie found in my search call
to the API
If you searched for The Imitation Game you will get 2 results back (0 and 1).
There was a version in the 80s and the most recent one in 2014.  If I used
data["results"][1]["overview"], I'd get the second movie in the search results.
Make life simple, always use the first movie returned.
'''

'''
Remember you'll have to do this for all your movies, the ones you
like, dislike, and ones you want to predict.  You'll have to store the results
in the appropriate lists
'''


# We combine our known positive and negative texts 
# into a combined training set to feed into the classifier 
#For you to do
training_texts = good_overview + bad_overview
'''
Rememeber the order you add the lists,  Did you put good movies, then bad ones
or vice versa
'''

# We also prepare an equivalent set of labels, to tell the machine
# that the first five texts are negative and the second ones are positive. 
# When we feed these into the classifier, it'll use indices to match up 
# the texts, e.g. the first label in the list is "negative", so it'll learn
# to associate the "negative" class with the first text.
# This works if there are the same number of positive & negative examples.
#For you to do
training_labels = ["good"] * len(good_overview) + ["bad"] * len(bad_overview)
'''
make sure you respect the order, the line above assumes a certain type of movie
first, then next the opposite type
'''

'''
Here we set up the vectorizer, the first main component of our machine learning
solution. 
'''
vectorizer = CountVectorizer()

# This isn't really learning anything difficult yet. We just 
# feed the data we have into our vectorizer so it can keep a 
# consistent mapping. E.g. it might map "bad" to 0, "love" to 1, 
# you to 2, etc.
vectorizer.fit(training_texts)
#print(vectorizer.vocabulary_)

# Now we transform all of our training texts into vector form. 
# At this point, each text is represented by a list of numbers,
# showing how often that word occurs in the text.
training_vectors = vectorizer.transform(training_texts)


# We'll do the same to our test texts. Each of these is a list 
# of numbers too after this step.
test_texts = test_overview #here you need your list of features of stuff you want
#the program to predict, just put the name of your list after the = sign

testing_vectors = vectorizer.transform(test_texts)

# Here we create our classifier and train it  by "showing" it the training
# texts and the associated labels. It will iterate over the data a few times, 
# trying different rules, until it finds a set of rules that works. 
classifier = tree.DecisionTreeClassifier()
classifier.fit(training_vectors, training_labels)


# It's easy to "overfit" -- find a set of rules that works very well
# for the set of data that we show the classifier, but which doesn't 
# work very well on other data, even if it's similar. Here we ask 
# the computer to guess whether our test texts (which it has never 
# seen) are more similar to the positive texts or the negative ones
# so we can check how well it works.


Results = classifier.predict(testing_vectors)
# You should tell the user what the predictions are
print("Uncut Gems is "+ Results[0])
print("Jojo Rabbit is "+ Results[1])
print("Toy Story 4 is "+ Results[2])
print("Venom is "+ Results[3])
    
# Then we export the model to a file so that we can visualise it. You can
# copy the content from `tree.dot` to http://www.webgraphviz.com/ to 
# see what the tree looks like
tree.export_graphviz(
    classifier,
    out_file='tree.dot',
    feature_names=vectorizer.get_feature_names(),
#For you to do
    class_names=["bad","good"]
    #Rememeber order matters in the above line
) 


""" Multi-line comment starts here
For you to do:
 We could hand code the rules ourselves. This function 
 does exactly the same thing, but we had to explicitly 
 tell the computer which words were good and which were bad
 instead of leaving it to figure things out for itself."""

def manual_classify(test_overview):
  if "only" in test_overview:
    return "bad"
  else:
    if "is" in test_overview:
      return "good"
    else:
      if "that" in test_overview:
        return "good"
      else:
        return "bad"
guess = [ ]
for titles in test_overview:
  predict = manual_classify(titles)
  guess.append(predict)
print("(Manual)Uncut Gems is "+ guess[0])
print("(Manual)Jojo Rabbit is "+ guess[1])
print("(Manual)Toy Story 4 is "+ guess[2])
print("(Manual)Venom is "+ guess[3])