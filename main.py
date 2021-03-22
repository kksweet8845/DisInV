from flask import Flask, jsonify, request
from flask import render_template
import json
import jieba
import pandas as pd
from Classifier.TriFnPredictor import TriFn_Predictor

app = Flask(__name__)

predictor = TriFn_Predictor('cuda', model_dir="./model/model_src-official")


@app.route('/checkIsFake', methods=['POST'])
def hello_world():
    print(request.get_json())
    text = request.form.get('text')
    text = request.get_json()['text']

    print(text)

    seg_list = jieba.cut(text, cut_all=False)

    seg_list = list(seg_list)
    print(seg_list)

    df = pd.DataFrame([[list(seg_list)]], columns=['token_title'])

    pred = predictor.predict(df)


    print(pred)

    return jsonify({
        'pos' : pred[1],
        'neg' : pred[0]
    })

@app.route('/result')
def result():
    return render_template('result.html', title="test",
                                         pos="30%",
                                         neg="70%")



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

