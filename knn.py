import sklearn.datasets as dt

from sklearn.neighbors import KNeighborsClassifier as knn

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import accuracy_score as score

model = knn()

X, y = dt.make_moons()

xtr, xts, ytr, yts = tts(X, y)

model.fit(xtr, ytr)

print(score(yts, model.predict(xts)))
