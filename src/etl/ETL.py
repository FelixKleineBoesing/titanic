import pandas as pd
import numpy as np

class ETL:

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.validation_data = None

        self.train_label = None
        self.validation_label = None

    def extract_data(self, train_path: str, test_path: str):
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)

    def clean_data(self, train_validation_split: float = 0.7):
        train_data = self.train_data
        test_data = self.test_data

        train_data = train_data.drop(["Name", "Ticket", "Cabin"], axis=1)
        test_data = test_data.drop(["Name", "Ticket", "Cabin"], axis=1)

        train_data["PassengerId"] = pd.to_numeric(train_data["PassengerId"])
        test_data["PassengerId"] = pd.to_numeric(test_data["PassengerId"])

        train_data["PassengerId"] = train_data["PassengerId"].astype(int)
        test_data["PassengerId"] = test_data["PassengerId"].astype(int)

        train_data["Pclass"] = train_data["Pclass"].astype(str)
        test_data["Pclass"] = test_data["Pclass"].astype(str)

        train_data["Age"] = pd.to_numeric(train_data["Age"])
        test_data["Age"] = pd.to_numeric(test_data["Age"])

        train_data["SibSp"] = pd.to_numeric(train_data["SibSp"])
        test_data["SibSp"] = pd.to_numeric(test_data["SibSp"])

        train_data["Parch"] = pd.to_numeric(train_data["Parch"])
        test_data["Parch"] = pd.to_numeric(test_data["Parch"])

        train_data["Fare"] = pd.to_numeric(train_data["Fare"])
        test_data["Fare"] = pd.to_numeric(test_data["Fare"])

        index = np.random.rand(len(train_data)) < train_validation_split
        validation_data = train_data[~index]
        train_data = train_data[index]

        self.train_label = train_data["Survived"]
        self.train_label.rename("Survived")
        train_data = train_data.drop("Survived", axis=1)
        self.train_data = train_data

        self.validation_label = validation_data["Survived"]
        self.validation_label.rename("Survived")
        validation_data = validation_data.drop("Survived", axis=1)
        self.validation_data = validation_data

        self.test_data = test_data

    def load_data(self, output_train_data: str, output_train_label: str,
                  output_test_data: str, output_validation_label: str,
                  output_validation_data: str):
        self.train_data.to_csv(output_train_data, index=False)
        self.train_label.to_csv(output_train_label, index=False, header=True)

        self.test_data.to_csv(output_test_data, index=False)

        self.validation_label.to_csv(output_validation_label, index=False, header=True)
        self.validation_data.to_csv(output_validation_data, index=False)
