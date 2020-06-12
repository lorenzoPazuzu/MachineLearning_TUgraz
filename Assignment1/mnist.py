import numpy as np
import gzip
import os
import urllib.request

def bytes_to_int(bytes):
    return int.from_bytes(bytes,byteorder='big', signed=False)

def download_data(data_set):
    if not os.path.exists(os.path.join(os.getcwd(), data_set)):
        print('Download ', data_set)
        urllib.request.urlretrieve("http://yann.lecun.com/exdb/mnist/" + data_set, data_set)
    else:
        print('Use cached', data_set)
    
    return data_set

def load_data(data_set='train-images-idx3-ubyte.gz'):
    data_set = download_data(data_set)

    with gzip.open(data_set, 'rb') as f:
        magic_number = bytes_to_int(f.read(4))
        num_items = bytes_to_int(f.read(4))
        num_rows = bytes_to_int(f.read(4))
        num_cols = bytes_to_int(f.read(4))

        data = np.zeros((num_items, num_rows, num_cols), dtype='uint8')
        for item in range(num_items):
            data[item] = np.fromstring(f.read(num_rows * num_cols), dtype='uint8').reshape(num_rows, num_cols)

    return data

def load_labels(data_set='train-images-idx1-ubyte.gz'):
    data_set = download_data(data_set)

    with gzip.open(data_set, 'rb') as f:
        magic_number = bytes_to_int(f.read(4))
        num_items = bytes_to_int(f.read(4))

        labels = np.zeros(num_items, dtype='uint8')
        for item in range(num_items):
            labels[item] = np.fromstring(f.read(1), dtype='uint8')
                
    return labels
