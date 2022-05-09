from flask import Flask, request, jsonify
import pandas as pd
import joblib
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

app = Flask(__name__)


@app.route("/")
def hello():
    return """
    <h1>Welcome to Test Bench App!</h1>
    <p>For example, please add:
     \"/predict?alim1=1.561000&alim2=0.651&sensor1=71.399760&sensor2=61.0&sensor3=0.0&sensor4=3.470364e%2b07&sensor5=-3.423649&pr1=148.015998&pr2=109.902395&pr3=242.148524&pr4=92.217590&temp1=130.130740&temp2=137.508594&brake1=2.752851\" to the URL to get your prediction.</p>
     """



@app.route('/predict')
def predict():
     # if key doesn't exist, returns None
     alim1 = request.args.get('alim1')
     alim2 = request.args.get('alim2')
     sensor1 = request.args.get('sensor1')
     sensor2 = request.args.get('sensor2')
     sensor3 = request.args.get('sensor3')
     sensor4 = request.args.get('sensor4')
     sensor5 = request.args.get('sensor5')
     pr1 = request.args.get('pr1')
     pr2 = request.args.get('pr2')
     pr3 = request.args.get('pr3')
     pr4 = request.args.get('pr4')
     temp1 = request.args.get('temp1')
     temp2 = request.args.get('temp2')
     brake1 = request.args.get('brake1')

     d = {'ALIM_1': [alim1],
          'ALIM_2': [alim2],
          'SENSOR_1': [sensor1],
          'SENSOR_2': [sensor2],
          'SENSOR_3': [sensor3], 
          'SENSOR_4': [sensor4],
          'SENSOR_5': [sensor5], 
          'PR_1': [pr1],
          'PR_2': [pr2],
          'PR_3': [pr3],
          'PR_4': [pr4],
          'TEMP_1': [temp1],
          'TEMP_2': [temp2],
          'BRAKE_1': [brake1]}

     test_bench_data = pd.DataFrame(data=d)

     # Load PCA
     pca_reload = pickle.load(open("pca.pkl",'rb'))
     X = pca_reload.transform(test_bench_data)

     # load the model from disk
     loaded_model = pickle.load(open('lgr_classifier_model.pkl', 'rb'))
     prediction = loaded_model.predict(X)
     return(f'prediction: {list(prediction)}')



@app.route('/predict_csv')
def predict_csv():
     # Load data
     test_bench_data = pd.read_csv('/home/fitec/Mise en situation professionnelle/Projets FITEC/Test Bench/POC/X_test.csv')
     y_test = pd.read_csv('/home/fitec/Mise en situation professionnelle/Projets FITEC/Test Bench/POC/y_test.csv', names=['y_test'], header=0)

     sc = pickle.load(open("sc.pkl",'rb'))
     X_scaled = sc.transform(test_bench_data)

     # Load PCA
     pca_reload = pickle.load(open("pca.pkl",'rb'))
     X = pca_reload.transform(X_scaled)

     # load the model from disk
     loaded_model = pickle.load(open('lgr_classifier_model.pkl', 'rb'))
     y_pred = loaded_model.predict(X)

     # Load predictions in a dataframe
     y_pred_df = pd.DataFrame(y_pred, columns=['y_pred'])

     # Save predictions and true labels in a single CSV file
     predictions_df = pd.concat([y_test,y_pred_df], axis=1)
     print(predictions_df)
     predictions_df.to_csv('predictions.csv',index=False)

     return """
     <p>Predictions: {}</p>
     <p>Accuracy: {}</p>
     <p>Recall: {}</p>
     """.format(y_pred, accuracy_score(y_test, y_pred), recall_score(y_test, y_pred))


if __name__ == '__main__':
     app.run(port=8080)