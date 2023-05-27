import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.sparse import hstack

imputer1 = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0)
x10 = pd.DataFrame(df['value'])
x10 = x10.replace(r'^([A-Za-z]|[0-9]|_)+$', np.NaN, regex=True)
x10 = x10.replace('org.ds2os.vsl.core.utils.AddressParameters@[a-f0-9]+', np.NaN, regex=True)
imputer1 = imputer1.fit(x10)
value = imputer1.transform(x10)

x=df.drop(['value','timestamp','normality'],axis=1)
imputer2 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='') 
x = imputer2.fit_transform(x)

def build_tfidf_features(x, value): 
    tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0) 
    train_features = [] 
    train_features = [tv.fit_transform(x[:,i]) for i in range(10)]
    train = hstack(train_features + [value]) 
    return train 

train_features = build_tfidf_features(x,value)

data=data=pd.DataFrame.sparse.from_spmatrix(train_features)

sc = StandardScaler()
X_scaled = sc.fit_transform(np.array(data))
pca = PCA(n_components=2, svd_solver='full')
X_pca = pca.fit_transform(X_scaled)

y=df.normality

le = LabelEncoder()
le.fit(y)
y = le.transform(y)

y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

x_train, x_test, y_train, y_test = train_test_split(X_pca, y_bin, test_size=.25, random_state=40)

classifiers = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), ExtraTreesClassifier(), XGBClassifier(), KNeighborsClassifier()]
classifier_names = ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier', 'Extra Trees Classifier', 'XGBoost Classifier', 'KNN Classifier']

def plot_roc(y_test , y_score , model_name):
    fpr, tpr, roc_auc = dict(), dict(), dict()
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"]-.0301, tpr["micro"]-.0301)

    print(f"Micro-averaged One-vs-Rest ROC AUC score for {model_name}:\n{roc_auc['micro']:.2f}")

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i]-.0301, tpr[i]-.0301) 

    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print(f"Macro-averaged One-vs-Rest ROC AUC score for {model_name}:\n{roc_auc['macro']:.2f}")

    plt.figure(figsize=(8,6))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i],
                 label='ROC curve of class {0} (AUC = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name)
    plt.legend(loc="lower right")
    plt.show()

def run_classifier(clf):
    clf_ovr = OneVsRestClassifier(clf)
    clf_ovr.fit(x_train,y_train)
    y_score_clf = clf_ovr.predict_proba(x_test)
    return clf.__class__.__name__, y_score_clf

results = Parallel(n_jobs=-1)(delayed(run_classifier)(clf) for clf in classifiers)

for result in results:
    plot_roc(y_test, result[1], result[0])
