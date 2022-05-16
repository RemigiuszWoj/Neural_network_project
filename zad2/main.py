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

def dane_dla_sieci (u, y, typ_sieci="NNARX"):
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
    # print(u.size)
    assert u.size == y.size #zabezpieczenie przed niejednakowymi rozmiarami
    n=len(u)
    # X=[ y[1:-2], y[0:-3], u[1:-2], u[0:-3] ] #macierz z wartosciami wejscia␣,→sieci NNARX
    # X=[ u[1:-2], u[0:-3]]   #NNFIR
    if typ_sieci == "NNARX":
        X=[ y[2:-1], y[1:-2], y[0:-3], u[2:-1], u[1:-2], u[0:-3] ] #macierz z wartosciami wejscia␣,→sieci NNARX
    elif typ_sieci == "NNFIR":   
        X=[ u[1:-2], u[0:-3]]   #NNFIR


    X=np.array (X)
    X=X.T #transpozycja
    T=y[2:-1] #pozadane wartosci wyjsc sieci
    T=np.array(T)
    return X, T

def build_network(X, T,typ_sieci="NNARX"):
    # Utworzenie sieci
    model = Sequential()
    if typ_sieci == "NNARX":
        input_shape = (6,) #liczba wejsc sieci - uwaga na przecinek - w pythonie 4␣,→rozni sie od (4,) !!!
    elif typ_sieci == "NNFIR":   
        input_shape = (2,) #liczba wejsc sieci - uwaga na przecinek - w pythonie 4␣,→rozni sie od (4,) !!!


    # input_shape = (4,) #liczba wejsc sieci - uwaga na przecinek - w pythonie 4␣,→rozni sie od (4,) !!!
    model.add(Dense(15, input_shape=input_shape, activation='tanh')) #15 neuronow␣,→z f.aktywacji tanh w pierwszej warstwie ukrytej
    model.add(Dense(10, activation='tanh')) #10 neuronow z f.aktywacji tanh
    model.add(Dense(1, activation='linear')) #1 neuron w warstwie wyjsciowej␣,→(liczba neuronow w tej warstwie jest rowna liczbie wyjsci sieci)
    # loss - funkcja straty (celu) minimalizowana podczas treningu, optimizer␣,→oznacz alg. uczenia
    model.compile(loss='mean_squared_error', optimizer='sgd',metrics=['mean_squared_error'])

    #trening sieci
    history = model.fit(X, T, epochs=10, batch_size=100, verbose=1,validation_split=0.2)
    model.summary()
    
    return model, history, X, T

def MSE_epoki(history, typ_sieci, save=True, run=0):
    plt.plot (history.history['loss'], label='train_loss')
    plt.plot (history.history['val_loss'], label='val_loss')
    #plt.ylim ([0, 10])
    plt.xlabel ('Epoki')
    plt.ylabel ('MSE')
    plt.legend()
    plt.grid (True)
    if save == True:
        filename = str(run) + os.sep + typ_sieci + os.sep + os.sep + "MSE_epoki" 
        plt.savefig(filename + '_plot.png')
        plt.close()
    else:
        plt.show()
    
def model_obiekt(Y_hat, T ,typ_sieci, save=True, run=0):
    plt.plot (Y_hat[200:300],'r', label='wyjscie modelu')
    plt.plot (T[200:300], label='wyjscie obiektu')
    plt.legend()
    plt.grid (True)
    if save == True:
        filename = str(run) + os.sep + typ_sieci + os.sep + os.sep + "model_obiekt" 
        plt.savefig(filename + '_plot.png')
        plt.close()
    else:
        plt.show()

def print_errors(errors, typ_sieci, save=True, run=0):
    plt.plot(errors)
    plt.title('błędy predykcji (róznica miedzy wyjściem obiektu i wyjściem modelu)')
    if save == True:
        filename = str(run) + os.sep + typ_sieci + os.sep + os.sep + "errors" 
        plt.savefig(filename + '_plot.png')
        plt.close()
    else:
        plt.show()

def print_hist(errors, typ_sieci, save=True, run=0):
    plt.hist(errors,50,)
    plt.title('Histogram błędów')
    if save == True:
        filename = str(run) + os.sep + typ_sieci + os.sep + os.sep + "hist" 
        plt.savefig(filename + '_plot.png')
        plt.close()
    else:
        plt.show()

def create_log(typ_sieci ,Y_hat, errors, T, X, run):
    #  logs
    Y_hat[:,0]
    MSE=np.mean(errors**2)

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

    my_data_file = open(str(run) + os.sep + typ_sieci + os.sep + "data_log" + ".txt", "+w")
    my_data_file.write("MSE: " + str(MSE) + "\n")
    my_data_file.write("T_Train: " + str(T_train.shape) + "\n")
    my_data_file.write("X_Train: " + str(X_train.shape) + "\n")

    my_data_file.write("T_Test: " + str(T_test.shape) + "\n")
    my_data_file.write("X_Test: " + str(X_test.shape) + "\n")

    # print("MSE:", MSE)
    # print ("T_Train: ", T_train.shape, '\nX_Train: ', X_train.shape)
    # print ("T_Test: ", T_test.shape, '\nX_Test: ', X_test.shape)

def badania(N, plots, logs, typ_sieci, run, save = True):
    U = np.random.uniform(low=0.0, high=2.0, size=N) #POBUDZENIE (trzeba założyć␣,→dopuszczalną min. i max. wartość)
    #rozklad jednostajny np.random.uniform - opis na https://numpy.org/doc/stable/,→reference/random/generated/numpy.random.uniform.html
    #print (U)

    Y = obiekt(U) #obliczenie wyjscia obiektu
        #odrzucenie pierwszych 10 probek u[10:]=u[10:len(u)]
    U2=U[10:]
    Y2=Y[10:]
    # plt.plot(Y2)
    # plt.ylabel('y')
    # plt.title('wyjście obiektu') #warto sprawdzic czy wartosc wyjscia nie dazy␣,→do +inf lub -inf (jezeli obiekt niestabilny prosze zmniejszyc wspolczynniki)
    # plt.show() #w ipython mozna pominac ta linijke

    #Utworzenie macierzy X i T
    X,T = dane_dla_sieci (U2, Y2, typ_sieci=typ_sieci)

    model, history, X, T = build_network(X=X, T=T, typ_sieci=typ_sieci)
    Y_hat = model.predict (X)
    T.shape #liczba danych
    X.shape #liczba danych x liczba wejsc modelu
    # print (Y_hat.shape)
    (Y_hat[:,0]).shape
    errors=T-Y_hat[:,0]

    if plots == True:
        #  plots
        MSE_epoki(history=history, typ_sieci=typ_sieci, save=save, run=run)
        model_obiekt(Y_hat=Y_hat, T=T, typ_sieci=typ_sieci, save=save , run=run)
        print_errors(errors=errors, typ_sieci=typ_sieci, save=save , run=run)
        print_hist(errors=errors, typ_sieci=typ_sieci, save=save , run=run)

    if logs == True:
        create_log(typ_sieci=typ_sieci, Y_hat=Y_hat, errors=errors, T=T, X=X, run=run)   





N = 10000

TYP_SIECI = ["NNARX", "NNFIR"]
# TYP_SIECI = TYP_SIECI[0]

RUNS = [1, 2, 3]
# RUNS = [0]

plots = True
logs = True
save = True # daj na False to nie bedzie tego bledu z os.sep() tylko wykresy beda sie plotowac

for run in RUNS:
    for typ_sieci in TYP_SIECI:
        print(run)
        print(typ_sieci)
        badania(N=N, plots=plots, logs=logs, typ_sieci=typ_sieci, run=run, save=save)
        