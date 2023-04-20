# MODEL TRAINING FILE

# importing dependencies
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle
import warnings

# declaring a class
class LogisticRegressionClass:

    # constructor
    def __init__(self):
        self.path = "./Data/titanic.csv"
        self.data = pd.DataFrame()
        self.target = pd.Series()
        self.features = pd.DataFrame()
        self.model = None
        self.y_predicted = None
        self.x_train, self.y_train, self.x_test, self.y_test = pd.DataFrame(),pd.DataFrame(), \
                                                               pd.DataFrame(),pd.DataFrame()

    # reading a file
    def readFile(self):
        self.data = pd.read_csv(self.path)

    # data printing
    def printData(self):
        print(self.data)

    # pre-processing data
    def preprocessingData(self):
        # reading file
        self.readFile()

        print("Pre-processing Data..............")

        # removing null values from Age column
        simple_impute = SimpleImputer(missing_values=np.NaN, strategy="mean")
        self.data['Age'] = simple_impute.fit_transform(np.array(self.data['Age']).reshape(-1, 1))

        # removing un important column
        self.data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=True,inplace=True)

        # removing row with missing embarked values
        self.data.dropna(inplace=True)
        # print(self.data.duplicated().sum())
        # exit()

        # # Label Encoding Sex and Embarked
        label_encoder = LabelEncoder()
        self.data['Sex'] = label_encoder.fit_transform(self.data['Sex'])
        print("GENDER :")
        print("Male Encoded with : ",label_encoder.transform(['male']))
        print("Female Encoded with : ",label_encoder.transform(['female']))

        self.data['Embarked'] = label_encoder.fit_transform(self.data['Embarked'])
        print("EMBARK STATION :")
        print("Queenstown encoded with : ",label_encoder.transform(['Q']))
        print("Cherbourg encoded with : ",label_encoder.transform(['C']))
        print("Southampton encoded with : ",label_encoder.transform(['S']))
        print("Pre-processing Competed.")

    # feature target split
    def dataSplitting(self):
        print("Data Splitting .................")
        # splitting target and feature columns
        self.target = self.data.iloc[:,0]
        self.features = self.data.iloc[:,1:]
        print("Splitting Done")

    # train test split
    def dataTrainTestSplit(self):
        print("Train test split ...............")
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.features,
                                                                         self.target,
                                                                         test_size=0.20)
        print("Splitting Completed")

    # model training function
    def modelTraining(self):

        # data pre-processing
        self.preprocessingData()
        # data splitting
        self.dataSplitting()
        # train test split
        self.dataTrainTestSplit()

        # Logistic Regression function
        print("Model Training .........")
        self.model = LogisticRegression()
        self.model.fit(self.x_train,self.y_train)
        print("Model Trained")

        # exporting model to respective pickle file
        self.exportPickle()

        # predicting values of xtest
        self.predictingValues()

    # predicting the values
    def predictingValues(self):
        # predict values
        self.y_predicted = self.model.predict(self.x_test)
        self.y_predicted = pd.Series(self.y_predicted)

    # performance -> accuracy score confusion matrix and classification report
    def performance(self):
        # accuracy score
        acc_score = accuracy_score(y_true=self.y_test, y_pred=self.y_predicted)
        print("\nAccuracy Score : ", acc_score * 100, "%")

        confusion = confusion_matrix(y_true=self.y_test, y_pred=self.y_predicted)
        print("\nConfusion Matrix : ")
        print(confusion)

        class_report = classification_report(y_true=self.y_test, y_pred=self.y_predicted)
        print("\nClassification Report : ")
        print(class_report)

    # exporting .pkl file
    def exportPickle(self):
        print("GENERATING MODEL ......")

        # creating a pickle object
        pkl_file = open("model.pkl","wb")
        pickle.dump(self.model,pkl_file)
        pkl_file.close()

        print("MODEL GENERATED READY TO USE [Pickle file created]\n")

# main function
if __name__ == "__main__":
    # for removing warnings
    warnings.filterwarnings("ignore")

    # creating class object
    logistic_regressor = LogisticRegressionClass()

    # model training
    logistic_regressor.modelTraining()
    # evaluating performance
    logistic_regressor.performance()
    print("Prepared by : Gracy Patel")

# end of file
