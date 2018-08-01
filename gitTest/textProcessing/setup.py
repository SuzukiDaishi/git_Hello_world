
from gensim.models import word2vec
import numpy as np
import MeCab
import json

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects
from matplotlib import pylab
import itertools

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Activation
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix


# 自然言語解析系ライブラリのセットアップ
mecab = MeCab.Tagger("-Owakati -d/usr/local/lib/mecab/dic/mecab-ipadic-neologd")
w2vModel = word2vec.Word2Vec.load("./latest-ja-word2vec-gensim-model/word2vec.gensim.model")

def main() :

	# テストデータの作成、整形
	(x_train, y_train), (x_test, y_test) = loadData("./akanechan_model/dataset.json")
	x_train = np.array(x_train)
	x_test  = np.array(x_test)
	x_train = x_train.reshape(x_train.shape[0], 50, 1)
	x_test  = x_test.reshape(x_test.shape[0], 50, 1)
	y_train = keras.utils.to_categorical(y_train, 3)
	y_test = keras.utils.to_categorical(y_test, 3)
	input_shape = (50, 1)
	class_count = 3
	print(f"x_train: {x_train.shape}, y_train: {y_train.shape},\nx_test:  {x_test.shape},  y_test:  {y_test.shape}")

	# モデルの作成
	model = Sequential()
	model.add( Dense(input_shape[0]*2, activation="tanh", input_shape=input_shape) )
	model.add( Dropout(0.25) )
	model.add( Flatten() )
	model.add( Activation("relu") )
	model.add( Dropout(0.5) )
	model.add( Dense(class_count, activation="softmax") )

	model.summary()

	plot_model(model, show_shapes=True, to_file="./akanechan_model/akanechan_model.png")

	# モデルのコンパイル
	model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=["accuracy"])

	# モデルの学習
	epochs = 38
	batch_size = 10
	history = model.fit(x_train, y_train,
	                    batch_size=batch_size,
						epochs=epochs,
						verbose=1,
						validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, verbose=0)
	print(f"テストデータの損失: {score[0]}")
	print(f"テストデータの精度: {score[1]}")
	print("save model...")
	json_string = model.to_json()
	open("./akanechan_model/cnn_model.json", "w").write(json_string)
	model.save_weights("./akanechan_model/akanechanCNN_model_weights.h5")

	# ここから描画タイム!
	font_prop = fm.FontProperties()
	font_prop.set_style("normal")
	font_prop.set_weight("light")
	font_prop.set_size("12")
	fp2 = font_prop.copy()
	fp2.set_size("20")
	fp2.set_family("serif")
	plt.figure(figsize=(14,10))
	plt.plot(history.history["acc"],
	         color="b",
	         linestyle="-",
	         linewidth=3,
	         path_effects=[path_effects.SimpleLineShadow(),
	                       path_effects.Normal()])
	plt.plot(history.history["val_acc"],
	         color="r",
	         linestyle="--",
	         linewidth=3,
	         path_effects=[path_effects.SimpleLineShadow(),
	                       path_effects.Normal()])
	plt.tick_params(labelsize=15)
	plt.title("Epoch-Accuracy-akanechan-CNN",fontsize=25,font_properties=fp2)
	plt.ylabel("Accuracy",fontsize=20,font_properties=fp2)
	plt.xlabel("Epoch",fontsize=20,font_properties=fp2)
	plt.legend(["Train", "Test"], loc="best", fontsize=20)
	plt.savefig('./akanechan_model/Epoch-Accuracy-akanechan-CNN.pdf')
	plt.figure(figsize=(14,10))
	plt.plot(history.history["loss"],
	         color="b",
	         linestyle="-",
	         linewidth=3,
	         path_effects=[path_effects.SimpleLineShadow(),
	                       path_effects.Normal()])
	plt.plot(history.history["val_loss"],
	         color="r",
	         linestyle="--",
	         linewidth=3,
	         path_effects=[path_effects.SimpleLineShadow(),
	                       path_effects.Normal()])

	plt.title("Epoch-Loss-akanechan-CNN",fontsize=25,font_properties=fp2)
	plt.ylabel("Loss",fontsize=20,font_properties=fp2)
	plt.xlabel("Epoch",fontsize=20,font_properties=fp2)
	plt.legend(["Train", "Test"], loc="best", fontsize=20)
	plt.savefig("./akanechan_model/Epoch-Loss-akanechan-CNN.pdf")
	actual = np.argmax(y_test, axis=1)
	pred_classes = np.argmax(model.predict(x_test), axis=1)
	cm = confusion_matrix(pred_classes, actual)
	classes = ["Seyana", "eenchau?", "ahokusa"]
	plot_confusion_matrix(cm, classes=classes, title="akanechan CNN")
	plt.savefig("./akanechan_model/confusionMatrix-akanechan-CNN.pdf")




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(12,9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=25)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=15)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize=15)
    plt.yticks(tick_marks, classes,fontsize=15)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center", fontsize=15,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=20)
    plt.xlabel('Predicted label',fontsize=20)


def loadData(jsonPath, TEST_DATA_COUNT=10) :
	with open(jsonPath) as file :
		texts = json.load(file)
		vectors = {}
		for k in texts.keys() :
			vectors[k] = []
			for t in texts[k] :
				vec = textToVector(t).tolist()
				if np.sum(vec) != 0 :
					vectors[k].append(vec)
	x_train = []
	y_train = []
	x_test  = []
	y_test  = []
	keyCount = 0
	for k in texts.keys() :
		for i in range(0, TEST_DATA_COUNT) :
			randIndex = np.random.randint(0, len(vectors[k]))
			indexVal  = vectors[k].pop(randIndex)
			x_test.append(indexVal)
			y_test.append(keyCount)
		x_train.extend(vectors[k])
		y_train.extend([ keyCount for i in vectors[k] ])
		keyCount += 1
	return (x_train, y_train), (x_test, y_test)





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

if __name__ == '__main__':
	main()
