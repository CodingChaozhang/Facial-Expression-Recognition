from keras.models import model_from_json
import numpy as np

# 创建FacialExpressionModel类
# 功能：提供先前训练模型的预测
class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Sad", "Surprise",
                     "Neutral"]

    def __init__(self, model_json_file, model_weights_file):
        # 从JSON文件中加载模型
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # 将权重加载到新模型中
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()
		#print("Model loaded from disk")
        #self.loaded_model.summary()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)

        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
