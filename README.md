# A-Machine-Learning-Approach-to-Anomaly-Detection-in-DS2OS-IoT-Traffic-Traces

The dataset used in this code is the DS2OS traffic traces dataset from Kaggle. This dataset contains IoT traffic traces gathered in the DS2OS IoT environment. The traces are from the application layer and are quite different from conventional network traces. The main dataset was gathered from a day of capture using four simulated IoT sites with different types of services: light controller, thermometer, movement sensors, washing machines, batteries, thermostats, smart doors and smart phones. Each site had a different organization and a different number of services.


This code is designed to perform anomaly detection on DS2OS traffic traces from Kaggle. The code starts by importing necessary libraries and then proceeds to impute missing values in the "value" column of the dataset using the mean strategy. The code then builds TF-IDF features on text columns and applies PCA to reduce dimensionality. The target variable is prepared by encoding it into integers and binarizing the output for multiclass ROC curve plotting. The data is then split into train and test sets. A list of classifiers is defined to compare their performance. The "plot_roc" function is used to plot the ROC curve for each classifier. The "run_classifier" function takes a classifier as an argument and returns its name and score. All classifiers are run in parallel and their results are stored in a list. The "plot_roc" function is called with the test labels, score, and name of each classifier.

This code performs anomaly detection on DS2OS traffic traces from Kaggle. It imputes missing values, builds TF-IDF features on text columns, applies PCA to reduce dimensionality, prepares the target variable, splits the data into train and test sets, and compares the performance of multiple classifiers by plotting their ROC curves.

To use this code, you will need to have a dataset stored in a variable named "df". This dataset should have columns named "value", "timestamp", and "normality". You will also need to have the following libraries installed: "numpy", "pandas", "sklearn", "xgboost", "matplotlib". 

