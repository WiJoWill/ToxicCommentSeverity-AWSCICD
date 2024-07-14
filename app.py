from flask import Flask,request,render_template

from src.pipeline.predict_pipeline import PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = request.form.get('text-to-detect')
        pred_df = [data]
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        toxic_pred, toxic_score = predict_pipeline.predict(pred_df)
        print("after Prediction")
        output_labels = ["Toxic","Severe Toxic","Obscene","Threat","Insult","Identity Hate"]
        output_data = {label: (round(result, 4)) for label, result in zip(output_labels, toxic_pred[0])}
        output_data['Overall'] = (round(toxic_score[0] / sum(predict_pipeline.weight_list), 4))

        return render_template('home.html',results = output_data)
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug = True)        

