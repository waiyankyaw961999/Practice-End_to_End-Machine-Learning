import pickle
from flask import Flask, request,render_template,flash, redirect, request, url_for
from model_files.ml_model import predict_mpg

app =  Flask(__name__,template_folder='templates')


model = pickle.load(open('./model_files/model.bin','rb'))
        

@app.route('/')
def index():    
    result ="result"

    return render_template("index.html",result=result)
    

@app.route("/predict",methods=["POST"])

def predict():
    result = "result"
   
    vehicle_config = {
        "cylinders": [request.form.get("cylinders")],
        "displacement": [request.form.get("displacement")],
        "horsepower": [request.form.get("horsepower")],
        "modelyear": [request.form.get("modelyear")],
        "weight": [request.form.get("weight")],
        "acceleration": [request.form.get("acceleration")],
        "origin":[request.form.get("origin")] 
    }    

    result = predict_mpg(vehicle_config,model)  
    result = round(result[0],2)  
    
        
    return render_template("index.html",result=result)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=9696)