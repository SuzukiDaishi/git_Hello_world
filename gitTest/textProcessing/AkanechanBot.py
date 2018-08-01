import numpy as np
from keras.models import model_from_json
from keras.optimizers import Adam
from gensim.models import word2vec
import MeCab
import json

mecab = MeCab.Tagger("-Owakati -d/usr/local/lib/mecab/dic/mecab-ipadic-neologd")
w2vModel = word2vec.Word2Vec.load("./latest-ja-word2vec-gensim-model/word2vec.gensim.model")

def textToVector(text, errorPrint=False) :
	wakati = mecab.parse(text).strip().split(" ")
	textVec = np.zeros(50)
	for t in wakati :
		try :
			textVec += w2vModel.wv[t]
			break
		except:
			if errorPrint :
				print(t, " is because unknown data")
	return textVec

class AkanechanBot :
	
    def __init__(self) :
        self.replyList = [
            ["せやな", "わかる", "そやな", "ほんま"],
            ["ええんちゃう？"],
            ["あほくさ"]
        ]
        self.model = model_from_json(open("./akanechan_model/cnn_model.json").read())
        self.model.load_weights("./akanechan_model/akanechanCNN_model_weights.h5")

    def replay(self, text) :
        textVec = textToVector(text).reshape(1, 50, 1)
        predictNum = self.model.predict(textVec).reshape(3).argmax()
        return np.random.choice(self.replyList[predictNum])

if __name__ == '__main__':
    # 使い方
    bot = AkanechanBot()
    replay = bot.replay("ぬわああああん疲れたもおおおおん")
    print(replay)
