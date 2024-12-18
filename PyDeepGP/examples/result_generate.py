import numpy as np
import GPy
from pylab import *
from sys import path
np.random.seed(42)
import numpy as np
import pandas as pd

import deepgp
from sklearn.preprocessing import MinMaxScaler
# TODO: You might need to normalize the input and/or output data.
sheets = ['USD_JPY_Positive Volatility', 'USD_JPY_Negative Volatility', 'GBP_USD_Positive Volatility','GBP_USD_Negative Volatility',
          'EUR_CHF_Positive Volatility', 'EUR_CHF_Negative Volatility', 'EUR_USD_Positive Volatility', 'EUR_USD_Negative Volatility',]
# data= pd.read_excel('positive_negative_volatility_per_currency.xlsx',sheet_name = 'USD_JPY_Negative Volatility',header=None)
for j in range(len(sheets)):
    data = pd.read_excel('positive_negative_volatility_per_currency.xlsx', sheet_name=sheets[j],
                         header=None)
    data = data.dropna(axis=0, how='any')
    x = np.linspace(0, len(data) - 1, len(data) - 1)
    y = data.iloc[30:, 1].values.astype(float)
    y = (y - y.min()) / (y.max() - y.min())
    sample_length = 30
    sample_num = len(y) // sample_length
    x_all = []
    y_all = []
    # for i in range(sample_num):
    # x_all.append(x[i*sample_length:(i+1)*sample_length])
    # y_all.append(y[i*sample_length:(i+1)*sample_length])


    for i in range(sample_num - 1):
        x_all.append(y[i * sample_length:(i + 1) * sample_length])
        y_all.append(y[(i + 1) * sample_length:(i + 2) * sample_length])

    x_all = np.array(x_all)
    y_all = np.array(y_all)
    X_train = x_all[:int(len(x_all) * 0.6)]
    Y_train = y_all[:int(len(x_all) * 0.6)]
    X_test = x_all[int(len(x_all) * 0.6):]
    Y_test = y_all[int(len(x_all) * 0.6):]

    #--------- Model Construction ----------#
    # Number of latent dimensions (single hidden layer, since the top layer is observed)
    Q = 1
    # Define what kernels to use per layer
    kern1 = GPy.kern.RBF(Q,ARD=True) + GPy.kern.Bias(Q)
    kern2 = GPy.kern.RBF(X_train.shape[1],ARD=False) + GPy.kern.Bias(X_train.shape[1])
    # Number of inducing points to use
    num_inducing = 40
    # Whether to use back-constraint for variational posterior
    back_constraint = False
    # Dimensions of the MLP back-constraint if set to true
    encoder_dims=[[300],[150]]

    m = deepgp.DeepGP([Y_train.shape[1],Q,X_train.shape[1]],Y_train, X=X_train,kernels=[kern1, kern2], num_inducing=num_inducing, back_constraint=back_constraint)


    #--------- Optimization ----------#
    # Make sure initial noise variance gives a reasonable signal to noise ratio.
    # Fix to that value for a few iterations to avoid early local minima
    for i in range(len(m.layers)):
        output_var = m.layers[i].Y.var() if i==0 else m.layers[i].Y.mean.var()
        m.layers[i].Gaussian_noise.variance = output_var*0.01
        m.layers[i].Gaussian_noise.variance.fix()

    m.optimize(max_iters=10000, messages=True)
    # Unfix noise variance now that we have initialized the model
    for i in range(len(m.layers)):
        m.layers[i].Gaussian_noise.variance.unfix()

    m.optimize(max_iters=1500, messages=True)

    #--------- Inspection ----------#
    # Compare with GP
    m_GP = GPy.models.SparseGPRegression(X=X_train, Y=Y_train, kernel=GPy.kern.RBF(X_train.shape[1])+GPy.kern.Bias(X_train.shape[1]), num_inducing=num_inducing)
    m_GP.Gaussian_noise.variance = m_GP.Y.var()*0.01
    m_GP.Gaussian_noise.variance.fix()
    m_GP.optimize(max_iters=100, messages=True)
    m_GP.Gaussian_noise.variance.unfix()
    m_GP.optimize(max_iters=400, messages=True)

    def mse(predictions, targets):
        return ((predictions.flatten() - targets.flatten()) ** 2).mean()

    Y_pred = m.predict(X_test)[0]
    Y_pred_s = m.predict_withSamples(X_test, nSamples=500)[0]
    Y_pred_GP = m_GP.predict(X_test)[0]

    # DeepGP isn't expected to outperform GPs always (especially on simple problems like this one here)
    print(sheets[j])
    print('# RMSE DGP               : ' + str(mse(Y_pred, Y_test)))
    print('# RMSE DGP (with samples): ' + str(mse(Y_pred_s, Y_test)))
    print('# RMSE GP                : ' + str(mse(Y_pred_GP, Y_test)))
