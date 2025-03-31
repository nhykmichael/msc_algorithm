import os
import pickle
import json  
from colorama import Fore

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics  
from datetime import datetime  

from APKManifestoExtractor import APKDetailsExtractor
from Tool import WarningCode

# For install androguard: pip install androguard

# This is a prototype for a mobile phone hacking detection tool
""" 
Prototype for Firmware/Hardware Mobile Phone Hacking Detection Tool Entry Point
Name: MNA Ahimbisibwe
Version: Prototype A

Model Evaluation Metrics for Random Forest Classifier Model with 100 trees and max depth of 50 
Trained under virtual machine environment due to restrictions MS WIN10 imposes
Training and testing data were randomly selected.
Training data was obtained from 1000 malware and 1000 normal APKs. 
Testing data was obtained from 100 malware and 100 normal APKs.
data set available at: https://github.com/ashishb/android-malware, 
We chose 100 trees because it is a good balance between accuracy and SPEED. We chose a max depth of 50 because it is a 
good balance between accuracy and OVERFITTING. 
"""


class APKAnalyseWithRandomForest:
    def __init__(self, normal_apks_dir, malware_apks_dir):
        """
        :type malware_apks_dir: object
        :param normal_apks_dir:
        :param malware_apks_dir:
        """

        self.malware_apks_dir = malware_apks_dir
        self.normal_apks_dir = normal_apks_dir
        self.feature_extractor = None
        self.model = None
        self.unpack_apk = None
        self.apk_data_list = []

        self.mirror = load_permissions_from_file()

        self.f_measure = None
        self.precision = None
        self.recall = None
        self.accuracy = None

        self.date_created = None
        self.parent_name = None
        self.run()

    def return_unpack(self, apk_path):
        # Implement feature extraction logic using androguard , also other methods can apply
        # Return the extracted features as a numpy array
        pass

    def collect_data_from_normal_apks(self, apk_folder_path, is_malware=0):
        """
        This method collects data from normal APKs and stores it in a list.
        type is_malware: object
        :param apk_folder_path:
        :param is_malware:
        :return:
        """
        for root_dir, sub_dirs, apk_files in os.walk(apk_folder_path):  # Walk through the directory tree
            for apk_filename in apk_files:  # For each file in the directory
                try:
                    print(Fore.CYAN, f"Processing {apk_filename}")
                    apk_full_path = os.path.join(root_dir, apk_filename)
                    if apk_filename.endswith(".apk"):  # and apk_filename.startswith("com"):
                        self.unpack_apk = APKDetailsExtractor(apk_full_path)
                        unpacked_apk_data = self.unpack_apk.extract()  # Call the extract method
                        if unpacked_apk_data is not None:
                            list_apk_feature_data = self.convert_apk_to_feature_data(unpacked_apk_data, is_malware)
                            self.apk_data_list.append(list_apk_feature_data)
                except Exception as e:
                    print(f"{Fore.RED}Failed processing {apk_filename}: {e}")
                    print(Style.RESET_ALL)

        # return apk_data_list

    def collect_data_from_malware_apks(self, apk_folder_path, is_malware=1):
        """
        This method collects data from malware APKs and stores it in a list.
        :param apk_folder_path:
        :param is_malware:
        :return:
        """
        for root_dir, sub_dirs, apk_files in os.walk(apk_folder_path):
            for apk_filename in apk_files:
                try:
                    # print(Fore.CYAN, f"Processing {apk_filename}")
                    apk_full_path = os.path.join(root_dir, apk_filename)
                    if apk_filename.endswith(".apk"):  # and apk_filename.startswith("com"):
                        self.unpack_apk = APKDetailsExtractor(apk_full_path)
                        unpacked_apk_data = self.unpack_apk.extract() # Call the extract method
                        if unpacked_apk_data is not None:
                            list_apk_feature_data = self.convert_apk_to_feature_data(unpacked_apk_data, is_malware)
                            self.apk_data_list.append(list_apk_feature_data)
                except Exception as e:
                    print(f"Failed processing {apk_filename}: {e}")

        # return apk_data_list

    """ 
    def prepare_data(self):
    
    """

    def train(self):
        """
        This method trains the model. Trainnig data is obtained from 1000 malware and 1000 normal APKs.
        Training Details: 80% of the data is used for training and 20% is used for testing.
        Training is achieven by randomly selecting 80% of the data for training and 20% for testing.
        scipy.stats.pearsonr is used to calculate the correlation between the features and the labels.
        :param malware_apks_folder_path:
        :return:
        """
        data_flame = pd.DataFrame(columns=self.mirror)
        for i in range(0, len(self.apk_data_list)):
            data_flame.loc[i] = self.apk_data_list[i]
        feature = data_flame
        if 'ID' in feature.keys():  # Remove the ID column
            feature.drop(feature.columns[0], axis=1, inplace=True)  # Remove the ID column
        feature.reset_index(drop=True, inplace=True)  # Reset the index
        y = feature[['is_malware']]  # This is the label column (is_malware) as a dataframe (y) with one column
        X = feature.drop(axis=1, labels=['is_malware'])  # Features

        # X, y = self.prepare_data()
        """ This is the training part of the code 
            test_size: 20% of the data is used for testing and 80% is used for training
            random_state: ensures that the data is randomly selected for training and testing
        """
        # Split the data into training and testing sets, 80% for training and 20% for testing
        # random_state ensures that the data is randomly selected for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the Random Forest model
        """This is the training part of the code classifying the data using the Random Forest Classifier
            oob_score=True: This parameter stands for "out-of-bag" score. In a random forest, each tree is trained on 
            a bootstrapped subset of the data, leaving out about one-third of the data. The out-of-bag score is an 
            estimate of the model's performance on the data it hasn't seen during training. It provides a quick estimate
            of the model's accuracy without the need for a separate validation set.
        """
        random_forest = RandomForestClassifier(n_estimators=100, max_depth=50, oob_score=True)

        """Out-of-Bag Evaluation: As the trees are trained on different subsets of the data, the out-of-bag samples 
        (data not used in training a particular tree) are used to estimate the model's performance."""

        """
        trains the random forest model on the training data. X_train is the feature matrix, and y_train.values.ravel() 
        is the flattened array of target labels. ravel() convert a 2D array to a 1D array by flattening the array row"""
        # Each tree is trained on a random subset of the training data using a process called bootstrapping.
        random_forest.fit(X_train, y_train.values.ravel())  # Train the model

        # Evaluate the model. The final prediction is made by combining the predictions of all the trees in the forest.
        # The prediction with the most votes is the final prediction of the model.
        y_prediction = random_forest.predict(X_test)
        # Print accuracy, recall, precision, F-measure
        """"""
        self.evaluate_model(y_test, y_prediction)
        self.model = random_forest
        self.parent_name = 'model'
        self.date_created = datetime.today().strftime('%Y-%m-%d')

        """f1 is a reserved keyword in Python, so we use f_measure instead of f1
            F_measure of F_score is the harmonic mean of precision and recall
            Used in machine learning to evaluate the accuracy (performance) of a test
            F_measure = 2 * (precision * recall) / (precision + recall)
            Particularly useful when you have an uneven class distribution
            i.e you have more false negatives than false positives or vice versa
        """
        mdl_data = {"model": self.model,
                    "parent_date": self.parent_name,  # parent_date is a reserved keyword in Python
                    "date_created": self.date_created,  # date_created is a reserved keyword in Python
                    "accuracy": self.accuracy,
                    "recall": self.recall,
                    "precision": self.precision,
                    "f1": self.f_measure,  # f1 is a reserved keyword in Python
                    }
        pickle.dump(mdl_data, open('apk_hacking_model', 'wb'))

    """ Precision is the number of true positives divided by the number of true positives plus the number of false
        positives. Precision is also called the Positive Predictive Value (PPV). Both precision and recall are therefore
        based on an understanding and measure of relevance. In ML Precision and Recall are used to evaluate the
        performance of a classification model. Precision is the fraction of relevant instances among the retrieved"""
    # Precision the exactness of a classifier, percentage of correct predictions that are relevant made by model
    '''Example: 100 people predicted to be sick by the model, and 70 of them are actually sick, then precision is 70%'''
    # Precision = True Positives / (True Positives + False Positives)
    # Recall is the fraction of relevant instances that have been retrieved over the total amount of relevant instances
    # Recall is also called Sensitivity or True Positive Rate (TPR)
    # Recall = True Positives / (True Positives + False Negatives)
    ''''Example: 100 people are actually sick, and the model correctly identified 70 of them, then recall is 70%'''

    # F1 Score is the weighted average of Precision and Recall
    # F1 Score reaches its best value at 1 (perfect precision and recall) and worst at 0

    def evaluate_model(self, y_test, y_prediction):
        self.accuracy = metrics.accuracy_score(y_test, y_prediction)
        self.recall = metrics.recall_score(y_test, y_prediction)
        self.precision = metrics.precision_score(y_test, y_prediction)
        self.f_measure = metrics.f1_score(y_test, y_prediction)

        print(Fore.CYAN + "Model Evaluation Metrics")
        print("Accuracy: {}".format(accuracy))
        print("Recall: {}".format(recall))
        print("Precision: {}".format(precision))
        print("F-Measure: {}".format(f_measure))
        print(Style.RESET_ALL)

    def convert_apk_to_feature_data(self, apk, is_malware=WarningCode.SUCCESS.value):

        # Create a dictionary with default values set to 0 for each permission in mirror
        # mirror is a list of permissions which are used as features to train the model
        permission_mapping = dict((i, WarningCode.SUCCESS.value) for i in self.mirror)

        # Ensure there's a placeholder for permissions not in the mirror
        permission_mapping.setdefault("other_permission", WarningCode.SUCCESS.value)

        # Go through the APK permissions and update the dictionary
        for perm in apk["allowed_permissions"]:
            if perm in list(self.mirror):
                permission_mapping[perm] = WarningCode.PASS.value
            else:
                permission_mapping["other_permission"] += WarningCode.PASS.value

        # Attach additional data to the dictionary
        permission_mapping["num_of_permissions"] = len(apk["allowed_permissions"])

        if is_malware is not None:  # If the is_malware parameter is provided, add it to the dictionary
            permission_mapping["is_malware"] = is_malware
        else:
            permission_mapping.pop("is_malware")

        # Convert the dictionary values to a list and return
        return list(permission_mapping.values())

    def identify_is_hacked(self, app_apk_dir, hacking_pretrained_model_dir):
        # If the model is not loaded, load it from the pretrained model file
        if self.model is None:
            # Load the pre-trained model and associated metrics
            model_load = pickle.load(open(hacking_pretrained_model_dir, 'rb'))
            self.model = model_load["model"]  # Load the pre-trained model from the pickle file
            self.accuracy = model_load["accuracy"]  # Load the accuracy metric
            self.recall = model_load["recall"]
            self.precision = model_load["precision"]
            self.f_measure = model_load["f1"]  # Load the F1-score metric from the pickle file
        # Accuracy: 0.996. Trained on 80% of the data and tested on 20% of the data.
        # Calculate and save feature importance scores
        weights = {self.mirror[i]: weight for i, weight in enumerate(self.model.feature_importances_)}
        sorted_weights = dict(sorted(weights.items(), key=lambda item: item[1]))
        '''To improve accuracy of the model, we can increase the number of trees in the forest, 
        increase the max depth of the trees, or increase the number of features used to train the model.
        We can also get high quality data to train the model. ie more malware and normal APKs. Meaning getting 
        data that is more representative of the real world. More specifically, we can get more malware and normal'''
        with open("model_stats.json", "w") as stats:
            json.dump(sorted_weights, stats, indent=4)  # Save the feature importance scores to a JSON file

        # Extract APK data and prepare for prediction

        # Create an instance of APKDetailsExtractor
        apk_extractor = APKDetailsExtractor(app_apk_dir)

        # Extract APK data and prepare for prediction
        apk_data = apk_extractor.extract()  # Call the extract method
        list_of_data = self.convert_apk_to_feature_data(apk_data, is_malware=None)

        # Make a prediction using the loaded model
        result = self.model.predict([list_of_data])

        return result[0], apk_data

    def run(self):
        self.collect_data_from_malware_apks(self.malware_apks_dir)
        self.collect_data_from_normal_apks(self.normal_apks_dir)


def load_permissions_from_file():
    with open("permissions.txt", "r") as permissions_file:
        permissions = [line.strip() for line in permissions_file]
    return permissions


"""
The method for detecting malicious apps, given an APK file to analyse and a pre-trained model, is implemented below.

 """


def _detect_malicious_app(app_apk_to_analyse: str, results_destination_JSON: str, pre_trained_model):
    we_can_analyse_with = APKAnalyseWithRandomForest("android-malware", "normal_apks")  # If you need training again
    # pass both files
    hacking_model_path = f"{os.path.dirname(os.path.abspath(__file__))}" + pre_trained_model

    # Provide the APK file to check
    # app_apk_to_analyse = input("Enter the path to the APK file to analyze: ")

    if not app_apk_to_analyse.endswith(".apk"):
        raise Exception("Please provide an .apk file.")

    # Check if model should be re-trained
    if not os.path.isfile(hacking_model_path):
        malware_folder = "android-malware"
        normal_folder = "normal_apks"

        if os.path.isdir(malware_folder) and os.path.isdir(normal_folder):
            apk_info = we_can_analyse_with.train_model(malware_apks_folder_path=malware_folder,
                                                       normal_apks_folder_path=normal_folder)
        else:
            raise Exception("Malware and normal APK folders not found for training.")

    # Check if the model exists
    if os.path.exists(hacking_model_path):
        outcomes, apk_data = we_can_analyse_with.identify_is_hacked(app_apk_to_analyse, hacking_model_path)

        if outcomes == 1:
            print(Fore.YELLOW + "Analysed App " + Fore.RED + "{}', Status--> Malicious!".format(
                app_apk_to_analyse) + Fore.WHITE)
        else:
            print(Fore.YELLOW + "Analysed App " + Fore.GREEN + "'{}', Status-->Not Malicious.".format(
                app_apk_to_analyse) + Fore.WHITE)

        # Provide the destination JSON file if needed
        # results_destination_JSON = input("Enter the path to the destination JSON file (optional): ")

        if results_destination_JSON.endswith(".json"):
            outcomes = True if outcomes == 1 else False
            package_name = apk_data.get("package", "Unknown Package")  # Get the package name or use a default
            data_to_write = {package_name: outcomes}
            # data_to_write = {apk_data["package"]: outcomes}

            if os.path.isfile(results_destination_JSON) and os.stat(results_destination_JSON).st_size != 0:
                with open(results_destination_JSON) as json_file:
                    current_json_data = json.load(json_file)
                    current_json_data.update(data_to_write)
                    data_to_write = current_json_data

            with open(results_destination_JSON, 'w') as file_p:
                json.dump(data_to_write, file_p, indent=4)
            print(Fore.CYAN + "Data written to JSON file." + Fore.WHITE)

        else:
            print(Fore.RED, "Destination file provided was not a JSON file.")
            print(Style.RESET_ALL)

    else:
        raise Exception("No model found. Please train the model.")
    return outcomes

# import joblib
# Assuming you have a trained model 'random_forest_model'
# joblib.dump("\\apk_hacking_model\\apk_good.model", 'updated_good_model.pkl')
# Load the saved model
# loaded_model = joblib.load('updated_good_model.pkl')
# import sklearn
# print(sklearn.__version__)

# _detect_malicious_app("a5starapps.com.drkalamquotes.apk", "model_stats.json", "\\apk_hacking_model\\apk_good.model")
# _detect_malicious_app("a5starapps.com.drkalamquotes.apk", "model_stats.json", loaded_model)
# _detect_malicious_app("ackman.placemarks.apk", "model_stats.json", "\\apk_hacking_model\\apk_good.model")
# _detect_malicious_app("C:\\Users\\Michael\\OneDrive\\Documents\\APK\\SuperVPN.apk", "model_stats.json",
#                       "\\apk_hacking_model\\apk_high.model")
# _detect_malicious_app("QR.apk", "model_stats.json", "\\apk_hacking_model\\apk_good.model")
