from data_loader import *

(X_train, y_train), (X_test, y_test) = load_gibbon_data()

if __name__ == '__main__':
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)