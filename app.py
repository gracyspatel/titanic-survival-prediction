# Titanic Survival Application

# importing dependencies
from flask import Flask,render_template,request
import pickle

# defining a class
class TitanicApp:

    # ctor initializes the variables
    def __init__(self):
        self.model=None
        self.name=str
        self.p_class = 0
        self.gender = 0
        self.age = 0
        self.sib_sp = 0
        self.parch = 0
        self.fare = 0.00
        self.embarked = 0

    # main route method
    def home(self):
        if request.method == "POST":

            # getting all the requested values
            self.name = request.form.get("name")
            self.p_class = int(request.form.get("p_class"))
            gender_form = request.form.get("gender")
            self.age = int(request.form.get("age"))
            self.sib_sp = int(request.form.get("sib_sp"))
            self.parch = int(request.form.get("parch"))
            self.fare = float(request.form.get("fare"))
            embarked_form = request.form.get("embarked")

            # encoding gender
            if gender_form == "Male":
                self.gender = 1
            else:
                self.gender = 0

            # encoding embarked station
            if embarked_form == "Cherbourg":
                self.embarked = 0
            elif embarked_form == "Queenstown":
                self.embarked = 1
            else:
                self.embarked = 2

            # predicting survival rate
            ans = self.predictingSurvival()

            # rendering template after submission
            return render_template("index.html",output=ans,name=self.name)
        return render_template('index.html')

    # method to load pickle file
    def modelExtraction(self):
        # model loading
        model_obj = open("model.pkl","rb")
        self.model = pickle.load(model_obj)

    # method to predict survival
    def predictingSurvival(self):
        return self.model.predict([[self.p_class,self.gender,self.age,self.sib_sp,self.parch,
                                    self.fare,
                                    self.embarked]])

# main method
if __name__ == "__main__":
    # object of titanic class
    titanic_app = TitanicApp()

    # extracting model
    titanic_app.modelExtraction()

    # flask object
    app = Flask(__name__)

    # creating routes
    # main page route
    app.add_url_rule('/',view_func=titanic_app.home,methods=['GET','POST'])

    # running application
    app.run(debug=True)

print("PREPARED BY GRACY PATEL")
# end of file
