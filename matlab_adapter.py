import scipy.io
import numpy as np

def mat_to_np(mat_file):
    data = scipy.io.loadmat(mat_file)
    boundary_data = np.array([data['xrub'], data['yrub']])
    return boundary_data.reshape(2, -1)

if __name__ == '__main__':
    print(mat_to_np('/home/local2/sunimali/research/TFM_prediction/Trial_4/Cell_2/Cell40/Cellboundary1.mat').shape)
