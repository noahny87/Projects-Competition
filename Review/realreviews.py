#here is where the real reviews will go to get evaluated 
import Review.reviews as rv 
from sklearn.model_selection import train_test_split 
import pandas as pd 

def main():
    #load testing data 
    test = pd.read_csv("Test Data.csv")
    #make a data frame 
    test = pd.DataFrame(test)
    #drop rating column 
    test2 = test.drop('rating',axis=1)
    #now we can preprocess data test2
    test2 = test['reviews'].apply(rv.preprocess)
    #now we can split our data 
    X_train, X_test, y_train, y_test = train_test_split(test2, test['liked'], test_size=0.25)
    # im not importing random state since i want to re-append the predicted values to the data set to see outputs 
    #now call our models, since naive bayes did better i am going to call that one and evaluate it 
    nb_model,cv = rv.train_naive_bayes(X_train,y_train)
    #evaluate model 
    rv.evaluate_model(nb_model,cv,X_test,y_test)
if __name__ == "__main__":
    main()

