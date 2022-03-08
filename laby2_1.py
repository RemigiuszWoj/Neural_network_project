#=========================Biblioteki=====================

import numpy
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import time

#=========================Biblioteki=====================


#=========================Parametry=====================

Algorytm='sgd'

WspolczynnikUczeniaSie=[0.1, 0.01, 0.001]

IloscWarstw=20

IloscProbek=60000

Iteracje=500

#=========================Parametry=====================


#=========================Wczytanie danych =====================

X_train, y_train = loadlocal_mnist(
        images_path='/Users/remik/Sieci_Neuronowe/train-images-idx3-ubyte',
        labels_path='/Users/remik/Sieci_Neuronowe/train-labels-idx1-ubyte')
X_test, y_test = loadlocal_mnist(
        images_path='/Users/remik/Sieci_Neuronowe/t10k-images-idx3-ubyte',
        labels_path='/Users/remik/Sieci_Neuronowe/t10k-labels-idx1-ubyte')
#=========================Wczytanie danych =====================


#========================standaryzacja===================================
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)


#========================standaryzacja===================================


#======================= kodowanie goracej jedynki ========================

enc=OneHotEncoder()
yy=y_train.reshape(-1,1)
yy.shape
enc.fit(yy)
yy_hot=enc.transform(yy).toarray()

#======================= kodowanie goracej jedynki ========================


#==================== Szukanie najlepszego klasyfikatora ===================



print("===============================================")


tablica = []
for LicznikWspolczynnkUzeniaSie in WspolczynnikUczeniaSie:

   mlp=MLPClassifier(hidden_layer_sizes=(IloscWarstw,), max_iter=Iteracje, solver=Algorytm, verbose=0,
                              random_state=1, learning_rate_init=LicznikWspolczynnkUzeniaSie)
   mlp.out_activation='softmax' # mapowanie z goraca jedynka
   print("Trening: warstwa: %d, eta: %.3f, algorytm: %s" % (IloscWarstw, LicznikWspolczynnkUzeniaSie, Algorytm))
   mlp.fit(X_train_std[:IloscProbek], y_train[:IloscProbek])
   print('Trening: warstwa: %d, eta: %.3f, algorytm: %s\n'
                'Dokladnosc dla treningowego: %f\n'
                'Dokladnosc dla testowego: %f\n'
                '================================\n'
   % (IloscWarstw, LicznikWspolczynnkUzeniaSie, Algorytm, mlp.score(X_train_std, y_train), mlp.score(X_test_std, y_test)))


   if Algorytm!= 'lbfgs':
       tablica.append(mlp.loss_curve_)

if  Algorytm!= 'lbfgs':
    plt.plot(tablica[0], label='eta=0.1')
    plt.plot(tablica[1], label='eta=0.01')
    plt.plot(tablica[2], label='eta=0.001')
    plt.ylabel('Strata obliczeniowa')
    plt.xlabel('Iteracja')
    plt.title('Alg: %s\nIlosc warstw: %d' % (Algorytm, IloscWarstw))
    plt.legend()
    plt.grid()
    plt.show()
    tablica.clear()
    plt.clf()


#==================== Szukanie najlepszego klasyfikatora ===================
