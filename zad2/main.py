"""
NNFIR   zmiana danych wejściowych -> usunięcie wartości y
        zmiana liczby wejść sieci z 4 na 2
NNARX   rozszerzenie NNFIR
        4 dane wejściowe -> u i y (poprawka na liczbie wejść sieci)
"""
import os
import numpy as np #operacje na macierzach podobnie jak w matlabie
import matplotlib.pyplot as plt #tworzenie wykresów bardzo podobnie jak w␣,→Matlabie
from tensorflow.keras.models import Sequential #keras - bardzo dobrze␣,→zaprojektowana biblioteka do sieci
from tensorflow.keras.layers import Dense

N = 10000
U = np.random.uniform(low=0.0, high=2.0, size=N) #POBUDZENIE (trzeba założyć␣,→dopuszczalną min. i max. wartość)
#rozklad jednostajny np.random.uniform - opis na https://numpy.org/doc/stable/,→reference/random/generated/numpy.random.uniform.html
#print (U)

def obiekt (u, add_noise=False, sigma=1.5):
    """
    Funkcja obliczajaca wyjscie obiektu
    u - wektor z wartosciami pobudzenia
    add_noise- czy do wyjścia obiektu dodać bialy szum (wart. oczekiwana␣
    ,→zakłócenia 0)
    sigma - odchylenie standardowe
    y - wartosci wyjscia obiektu
    """
    y=np.zeros (u.size) #rezerwacja pamieci
    #załóżmy, że obiekt jest opisany równaniem z zadania 6.
    for k in range (2,len(u)):
        y[k] = y[k-1] - 0.5*y[k-2] - 0.1*(y[k-2])**2 + u[k-1]+0.4*u[k-2]
    if add_noise:
        y += np.random.normal (0.0, sigma, u.size) #0.0 - wartość oczekiwana␣,→powinna byc zero, 1.5 - odchylenie standardowe
    return y

Y = obiekt(U) #obliczenie wyjscia obiektu
#odrzucenie pierwszych 10 probek u[10:]=u[10:len(u)]
U2=U[10:]
Y2=Y[10:]
plt.plot(Y2)
plt.ylabel('y')
plt.title('wyjście obiektu') #warto sprawdzic czy wartosc wyjscia nie dazy␣,→do +inf lub -inf (jezeli obiekt niestabilny prosze zmniejszyc wspolczynniki)
plt.show() #w ipython mozna pominac ta linijke

def dane_dla_sieci (u, y):
    """
    Przygotowanie zbioru danych dla sieci neuronowej
    u - wektor z wartosciami pobudzenia
    y - wektor z wartosciami wyjscia
    Funkcja zwraca:
    X - macierz z wartosciami wejsc sieci
    T - wektor z wartosciami wyjścia sieci (T od ang. target)
    W przypadku obiektu z zadania 6. sieć neuronowa dla modelu NNARX ma 4␣
    ,→wejścia: y(k-1), y(k-2), u(k-1), u(k-2)
    i jedno wyjscie y_hat(k)
    """
    print(u.size)
    assert u.size == y.size #zabezpieczenie przed niejednakowymi rozmiarami
    n=len(u)
    #X=[ y[1:-2], y[0:-3], u[1:-2], u[0:-3] ] #macierz z wartosciami wejscia␣,→sieci NNARX
    X=[ u[1:-2], u[0:-3]]   #NNFIR
    X=np.array (X)
    X=X.T #transpozycja
    T=y[2:-1] #pozadane wartosci wyjsc sieci
    T=np.array(T)
    return X, T

 # Utworzenie sieci
model = Sequential()
input_shape = (2,) #liczba wejsc sieci - uwaga na przecinek - w pythonie 4␣,→rozni sie od (4,) !!!
model.add(Dense(15, input_shape=input_shape, activation='tanh')) #15 neuronow␣,→z f.aktywacji tanh w pierwszej warstwie ukrytej
model.add(Dense(10, activation='tanh')) #10 neuronow z f.aktywacji tanh
model.add(Dense(1, activation='linear')) #1 neuron w warstwie wyjsciowej␣,→(liczba neuronow w tej warstwie jest rowna liczbie wyjsci sieci)
# loss - funkcja straty (celu) minimalizowana podczas treningu, optimizer␣,→oznacz alg. uczenia
model.compile(loss='mean_squared_error', optimizer='sgd',metrics=['mean_squared_error'])
#Utworzenie macierzy X i T
X,T = dane_dla_sieci (U2,Y2)
#trening sieci
history = model.fit(X, T, epochs=10, batch_size=100, verbose=1,validation_split=0.2)

model.summary()

plt.plot (history.history['loss'], label='train_loss')
plt.plot (history.history['val_loss'], label='val_loss')
#plt.ylim ([0, 10])
plt.xlabel ('Epoki')
plt.ylabel ('MSE')
plt.legend()
plt.grid (True)
plt.show ()

Y_hat = model.predict (X)

plt.plot (Y_hat[200:300],'r', label='wyjscie modelu')
plt.plot (T[200:300], label='wyjscie obiektu')
plt.legend()
plt.grid (True)
plt.show ()

T.shape #liczba danych
X.shape #liczba danych x liczba wejsc modelu
print (Y_hat.shape)
(Y_hat[:,0]).shape

errors=T-Y_hat[:,0]
plt.plot(errors)
plt.title('błędy predykcji (róznica miedzy wyjściem obiektu i wyjściem modelu)')
plt.show()

Y_hat[:,0]
MSE=np.mean(errors**2)
print(MSE)

#UWAGA - nie nalezy testowac na danych uzytych do treningu !!!
#mozna zbior danych podzielic na zbior uczący (treniningowy) i testowy
n=len(T)
n2 = n//2 #w pythonie operator // oznacza dzielenie bez reszty
#np. pierwsza polowa danych na zbior uczacy
T_train = T[:n2]
X_train = X[:n2,:] #podobnie jak w matlabie - przed przecinkiem wiersze, po␣,→przecinku kolumny
#np. druga polowa danych na zbior testujacy - jest on potrzebny, gdyz ocena␣,→modelu powinna zostac wykonana na danych NIE uzytych do treningu!
T_test = T[n2:]
X_test = X[n2:,:]

print ("T_Train: ", T_train.shape, '\nX_Train: ', X_train.shape)
print ("T_Test: ", T_test.shape, '\nX_Test: ', X_test.shape)
