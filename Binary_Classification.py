# Import Libraries
import pandas as pd 
import streamlit as st
import numpy as np 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,plot_precision_recall_curve,precision_score,recall_score

st.title("Binary Classification Web App")
st.sidebar.title("Binary Classification Web App")
st.markdown("Are your mushrooms edible or poisonous?🍄")
st.sidebar.markdown("Are your mushrooms edible or poisonous?🍄")

#Load Data
data_url = r"C:\Users\Pranay\DATA_ML\mushrooms.csv"

@st.cache(persist = True)
def load_data():
	data = pd.read_csv(data_url)
	label = LabelEncoder()
	for col in data.columns:
		data[col] = label.fit_transform(data[col])
	return data 

@st.cache(persist=True)
def split(df):
	y = df.type
	x = df.drop(columns=['type'])
	x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)
	return x_train,x_test,y_train,y_test

#Plot metrics
def plot_metrics(metrics):
	if "Confusion Matrix" in metrics:
		st.subheader("Confusion Matrix")
		plot_confusion_matrix(model,x_test,y_test,display_labels = class_name)
		st.pyplot()
	if "ROC Curve" in metrics:
		st.subheader("ROC Curve")
		plot_roc_curve(model,x_test,y_test)
		st.pyplot()
	if "Precision-Recall Curve" in metrics:
		st.subheader("Precision-Recall Curve")
		plot_precision_recall_curve(model,x_test,y_test)
		st.pyplot()

df = load_data()
x_train,x_test,y_train,y_test = split(df)
class_name = ['edible','poisonous']
st.sidebar.subheader("Choose Classifier")
classifier = st.sidebar.selectbox("Classifier",('Support Vector Machine (SVM)','Logistic Regression','Random Forest Classifier',
	'Naive Bayes Classifier','K-Nearest Neighbors'))

#SVM
if classifier == 'Support Vector Machine (SVM)':
	st.sidebar.subheader("Model Hyperparameters")
	C = st.sidebar.number_input("C (Regularization Parameter)",0.0,10.0,step=0.1,key='c')
	kernel = st.sidebar.radio("Kernel",('rbf','linear','poly','sigmoid'),key='kernel')
	gamma = st.sidebar.radio("Gamma (Kernel Coeffecient)",('scale','auto'),key='gamma')
	metrics = st.sidebar.multiselect("Metrics To Plot",("Confusion Matrix","ROC Curve","Precision-Recall Curve"),key='metrics')
	if st.sidebar.button("Classify",key='classify'):
		st.header("Support Vector Machine (SVM) Results")
		model = SVC(C = C,kernel=kernel,gamma=gamma)
		model.fit(x_train,y_train)
		y_pred = model.predict(x_test)
		accuracy = model.score(x_test,y_test)
		st.write("Accuracy: ",accuracy.round(2))
		st.write("Precision: ",precision_score(y_test,y_pred,labels=class_name).round(2))
		st.write("Recall: ",recall_score(y_test,y_pred,labels=class_name).round(2))
		plot_metrics(metrics)

#Logistic Regression
if classifier == 'Logistic Regression':
	st.sidebar.subheader("Model Hyperparameters")
	C = st.sidebar.number_input("C (Regularization Parameter)",0.0,10.0,step=0.01,key='c')
	penalty = st.sidebar.radio("Penalty",('l1','l2','elasticnet'),key='penalty')
	max_iter = st.sidebar.slider("Maximum Iterations",100,500,key='max_iter')
	metrics = st.sidebar.multiselect("Metrics To Plot",("Confusion Matrix","ROC Curve","Precision-Recall Curve"),key='metrics')
	if st.sidebar.button("Classify",key='classify'):
		st.header("Logistic Regression Results")
		model = LogisticRegression(C=C,penalty=penalty,max_iter=max_iter)
		model.fit(x_train,y_train)
		y_pred = model.predict(x_test)
		accuracy = model.score(x_test,y_test)
		st.write("Accuracy: ",accuracy.round(2))
		st.write("Precision: ",precision_score(y_test,y_pred,labels=class_name).round(2))
		st.write("Recall: ",recall_score(y_test,y_pred,labels=class_name).round(2))
		plot_metrics(metrics)

#Random Forest Classifier
if classifier == 'Random Forest Classifier':
	st.sidebar.subheader("Model Hyperparameters")
	n_estimators = st.sidebar.number_input("Number Of Trees",100,5000,step=10,key='n_estimators')
	max_depth = st.sidebar.number_input("Maximum Depth Of Tree",1,20,step=1,key='max_depth')
	criterion = st.sidebar.radio("Criterion",('gini','entropy'),key='criterion')
	bootstrap = st.sidebar.radio("Bootstrap",('True','False'),key='bootstrap')

	metrics = st.sidebar.multiselect("Metrics To Plot",("Confusion Matrix","ROC Curve","Precision-Recall Curve"),key='metrics')
	if st.sidebar.button("Classify",key='classify'):
		st.header("Random Forest Classifier Results")
		model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,criterion=criterion,bootstrap=bootstrap)
		model.fit(x_train,y_train)
		y_pred = model.predict(x_test)
		accuracy = model.score(x_test,y_test)
		st.write("Accuracy: ",accuracy.round(2))
		st.write("Precision: ",precision_score(y_test,y_pred,labels=class_name).round(2))
		st.write("Recall: ",recall_score(y_test,y_pred,labels=class_name).round(2))
		plot_metrics(metrics)

#Naive Bayes Classifier
if classifier == 'Naive Bayes Classifier':
	st.sidebar.subheader("Model Hyperparameters")

	metrics = st.sidebar.multiselect("Metrics To Plot",("Confusion Matrix","ROC Curve","Precision-Recall Curve"),key='metrics')
	if st.sidebar.button("Classify",key='classify'):
		st.header("Naive Bayes Classifier Results")
		model = GaussianNB()
		model.fit(x_train,y_train)
		y_pred = model.predict(x_test)
		accuracy = model.score(x_test,y_test)
		st.write("Accuracy: ",accuracy.round(2))
		st.write("Precision: ",precision_score(y_test,y_pred,labels=class_name).round(2))
		st.write("Recall: ",recall_score(y_test,y_pred,labels=class_name).round(2))
		plot_metrics(metrics)

#K-Nearest Neighbors
if classifier == 'K-Nearest Neighbors':
	st.sidebar.subheader("Model Hyperparameters")
	n_neighbors = st.sidebar.number_input("Number Of Neighbors",1,25,step=1,key='n_neighbors')
	algorithm = st.sidebar.radio("Algorithm",('auto','ball_tree','kd_tree','brute'),key='algorithm')
	weights = st.sidebar.radio("Weights",('uniform','distance'),key='weights')

	metrics = st.sidebar.multiselect("Metrics To Plot",("Confusion Matrix","ROC Curve","Precision-Recall Curve"),key='metrics')
	if st.sidebar.button("Classify",key='classify'):
		st.header(" K-Nearest Neighbors Classifier Results")
		model = KNeighborsClassifier(n_neighbors=n_neighbors,algorithm=algorithm,weights=weights)
		model.fit(x_train,y_train)
		y_pred = model.predict(x_test)
		accuracy = model.score(x_test,y_test)
		st.write("Accuracy: ",accuracy.round(2))
		st.write("Precision: ",precision_score(y_test,y_pred,labels=class_name).round(2))
		st.write("Recall: ",recall_score(y_test,y_pred,labels=class_name).round(2))
		plot_metrics(metrics)

#Show Raw Data
if st.sidebar.checkbox("Show Raw Data",False):
	st.subheader("Mushroom Data Set (Classification)")
	st.write(df)

