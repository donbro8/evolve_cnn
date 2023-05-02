from data_loader import *
import os

os.chdir('/Users/Donovan/Documents/Masters/masters-ed02/clean_code') # Remove when running from main.py


(X_train, y_train), (X_test, y_test) = load_gibbon_data()

# Error when running the above code. Test labels are in incorrect format. Do not know whether they are gibbon or not
# Need to figure out what is up with this.
# FileNotFoundError: [Errno 2] No such file or directory: '/Users/Donovan/Documents/Masters/masters-ed02/clean_code/data/raw_gibbon_data/test_labels/g_HGSM3B_0+1_20160306_055900.data'

if __name__ == '__main__':
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)