import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

## import ridge regresor model and standard scaler pickle
ridge = pickle.load(open('models/ridge.pkl','rb'))
scaler = pickle.load(open('models/scaler.pkl','rb'))

#corr_features = ['BUI','DC']

print(f"Number of features the model was trained on: {len(ridge.coef_)}")

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:

           Temperature=float(request.form.get('Temperature'))
           RH = float(request.form.get('RH'))
           Ws = float(request.form.get('Ws'))
           Rain = float(request.form.get('Rain'))
           FFMC = float(request.form.get('FFMC'))
           DMC = float(request.form.get('DMC'))
           ISI = float(request.form.get('ISI'))
           Classes = float(request.form.get('Classes'))
           Region = float(request.form.get('Region'))
       
           new_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes,Region]]
           print(f"Input data: {new_data}")
           print(f"Shape of input data: {np.array(new_data).shape}")  # Should print (1, 9)
        
                
           new_data_df = pd.DataFrame(new_data, columns=['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region'])
           #new_data_df.drop(corr_features,axis = 1,inplace = True)
           new_data_scaled = scaler.transform(new_data_df)
           result = ridge.predict(new_data_scaled)
           print(f"Prediction result: {result[0]}")  # Log the result for debugging

            # Return the result to the user
           return render_template('home.html', result=result[0])
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('home.html', result="Error in prediction")
       
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(port = 5000,debug = True)