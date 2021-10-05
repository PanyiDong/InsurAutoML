import math
import copy
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from scipy.optimize import fsolve
import sympy as sym
from sympy import *
from sympy import Abs, solveset, symbols, S
from sympy.functions import sign
import matplotlib
import matplotlib.pyplot as plt

#Linear Regression Model
# Using Matrix form calculation for least square
# Symbolic derivatives for gradient descent
class My_LinearRegression(object) :

    '''
    Linear Regression Models

    Parameters
    ----------
    solution: how to solve linear regression, default = 'LS',
    allows 'LS' (least square) and 'GS' (gradient descent)

    backtracking: whether using backtracking to select gradient descent steps, default = False

    method: how to achieve gradient descent, default = 'linear'
    'linear' uses pre-calculated gradient, 'symbolic' uses symbolic formula (takes time)

    ini_beta: initial linear regression coefficients, default = None
    'default' will be 0 vector, or use list/array as initial coeffcients

    plot: whether to plot loss during gradient descent, default = False

    delta: initial/fix gradient descent steps, default = 0.05

    max_iter: maximum number of iterations for gradient descent, contain computational time, default = 1000

    loss_limit: loss threshold to stop iteration, default = 10 ** (-5)

    backtracking_c: constant for setting Armijo–Goldstein Test in backtracking, default = 0.5

    backtracking_tau: every change on delta in backtracking, default = 0.5

    '''

    def __init__(self, solution = 'LS', backtracking = False, method = 'linear', ini_beta = None, plot = False, \
        delta = 0.05, max_iter = 1000, loss_limit = 10 ** (-5), backtracking_c = 0.5, backtracking_tau = 0.5) :
        self.solution = solution
        self.backtracking = backtracking
        self.method = method
        self.ini_beta = ini_beta
        self.plot = plot
        self.delta = delta
        self.max_iter = max_iter
        self.loss_limit = loss_limit
        self.backtracking_c = backtracking_c
        self.backtracking_tau = backtracking_tau

    def Loss_Cal(self, x, y, beta, include_gradient = False) :
        hat_y = np.dot(x, beta)
        err = y - hat_y
        loss = np.mean(err ** 2)
        if include_gradient == True :
            gradient = - 2 * np.mean(err.T * x.T, axis = 1).reshape((len(beta), 1))
            return (loss, gradient)
        else :
            return loss
    
    def Metric_Cal(self, x, y, beta) :
        hat_y = np.dot(x, beta)
        err = y - hat_y
        metric = np.mean(np.abs(err))
        return metric

    def fit(self, x, y) :
        if self.solution == 'LS' :
            self.Least_Square(x, y)
        elif self.solution == 'GS' :
            self.Gradient_Descent(x, y, self.backtracking, self.method, self.ini_beta, self.plot)
        else :
            raise ValueError('Not recognizing solution!')

    def Least_Square(self, x, y) :
        
        if len(y) != len(x) :
            raise ValueError('Arrays must have same sizes.')

        m = len(x)
        n = len(x[0])
        dof = m - n - 2 # degree of freedom
        
        #Calculate the properties of the Linear Model
        x_mean = [np.mean(x[:][i]) for i in range(n)] 
        y_mean = np.mean(y)   
        
        # Calculate estimates of regerssion coefficients with matrix calculation
        # b is estimates of true regression parameters \beta
        # (X' * X) * b = X' * y # X' for transpose of X
        x = np.array(np.insert(x, 0, 1, axis = 1), dtype = np.float64).tolist()
        A = np.array(np.dot(np.array(x).T, x), dtype = np.float64)
        b = np.dot(np.array(x).T, y)

        if np.linalg.det(A) != 0 :
            beta = np.linalg.solve(A, b)
        else :
            beta = np.linalg.lstsq(A, b, rcond = None)[0]
        regress = np.dot(x, beta)
        residual = y - regress
        residual_mean = np.mean(residual)

        loss = sum(item ** 2 for item in residual) / len(residual)
        metric = sum(np.abs(item) for item in residual) / len(residual)
        RSE = np.sqrt(sum(i ** 2 for i in residual) / dof) # Residual Standard Error, unbiased

        # 95 percent Confidence Interval on regression parameters
        # Standard error matrix
        # SE = s^{2} * (X' * X)^{-1}
        variance_matrix = (RSE ** 2) * np.array(np.linalg.inv(A), dtype = np.float64)
        SE = []
        for i in range(n + 1) :
            SE.append(np.sqrt(variance_matrix[i][i]))
        
        # 95% Confidence interval of estimated coefficients
        # with Normal distribution
        # stored as 2D tuple, [min, max] form
        CI = []
        for i in range(n + 1) :
            CI.append([beta[i] - scipy.stats.norm.ppf(0.975) * SE[i], beta[i] + scipy.stats.norm.ppf(0.975) * SE[i]])

        # Sum of Squares
        TSS = sum((y[i] - y_mean) ** 2 for i in range(m)) # Total Sum of Squares
        RSS = sum((y[i] - regress[i]) ** 2 for i in range(m)) # Residual Sum of Squares
        ESS = sum((regress[i] - y_mean) ** 2 for i in range(m)) # Explained Sum of Saqures
        R_squared = ESS / TSS

        # Estimate for test error
        #C_{p}
        hat_sigma_squared = sum((residual[i] - residual_mean) ** 2 for i in range(m)) / (m - 1)
        C_P = (RSS + 2 * n * hat_sigma_squared) / m
        #AIC
        AIC = (RSS + 2 * n * hat_sigma_squared) / (m * hat_sigma_squared)
        #BIC
        BIC = (RSS + math.log(m) * n * hat_sigma_squared) / m
        #Adjusted R^{2}
        Adjusted_R_squared = 1 - (RSS / (m - n - 1)) / (TSS / (m - 1))

        # Save these properties as attributes
        self.dof = dof
        self.x_mean = [round(item, 4) for item in x_mean]
        self.y_mean = round(y_mean, 4)
        self.beta = [item for item in beta]
        self.loss = loss
        self.metric = metric
        self.variance_matrix = variance_matrix
        self.CI = CI
        self.RSE = RSE
        self.TSS = TSS
        self.RSS = RSS
        self.ESS = ESS
        self.R_squared = R_squared
        self.C_P = C_P
        self.AIC = AIC
        self.BIC = BIC
        self.Adjusted_R_squared = Adjusted_R_squared

    def Gradient_Descent(self, x, y) :

        x = np.array(x)
        y = np.array(y)

        # Basic of the data to fit
        if len(y) != len(x) :
            raise ValueError('Arrays must have same sizes.')

        m = len(x)
        n = len(x[0])
        dof = m - n - 2 # degree of freedom
        
        #Calculate the properties of the Linear Model
        x_mean = [np.mean(x[:][i]) for i in range(n)] 
        y_mean = np.mean(y)   
        
        # Calculate estimates of regerssion coefficients with matrix calculation
        # b is estimates of true regression parameters \beta
        x = np.array(np.insert(x, 0, 1, axis = 1), dtype = np.float64)
        
        # initial states
        if self.ini_beta == None :
            beta = [[0] for i in range(n + 1)]
        else :
            beta = self.ini_beta
        iter_time = 0
        delta_loss = np.Infinity
        _delta = self.delta
        time_series = []
        loss_series = []
        metric_series = []

        if self.method == 'symbolic' : # symbolic differential calculation, much slower but more compatible 

            beta_vector = [[symbols('beta' + str(i))] for i in range(n + 1)]
            # loss function and differential in symbolic form
            hat_y = np.dot(x, beta_vector)
            loss = np.mean([(y[i][0] - hat_y[i][0]) ** 2 for i in range(len(hat_y))])
            loss = sym.simplify(loss)
            partial_loss = [[diff(loss, item[0])] for item in beta_vector]
            eval_partial_loss = lambdify([item[0] for item in beta_vector], partial_loss)

        while iter_time < self.max_iter and delta_loss > self.loss_limit :
            
            loss, gradient = self.Loss_Cal(x, y, beta, include_gradient = True)
            if self.method == 'symbolic' :
                grad = eval_partial_loss(*[item[0] for item in beta])
                new_beta = beta - _delta * np.array(grad) / m
            elif self.method == 'linear' :
                new_beta = beta - _delta * gradient
            new_loss = self.Loss_Cal(x, y, new_beta, include_gradient = False)
            # use backtracking to find optimal delta
            if self.backtracking == False :
                beta = new_beta
                new_metric = self.Metric_Cal(x, y, beta)
                delta_loss = np.abs(new_loss - loss)
                
                _delta = copy.deepcopy(self.delta) # reset delta
                iter_time = iter_time + 1
                time_series.append(iter_time)
                loss_series.append(new_loss)
                metric_series.append(new_metric)
            elif self.backtracking == True :
                # Armijo–Goldstein Test
                if new_loss <= loss - self.backtracking_c * _delta * np.sum(gradient ** 2) :
                    beta = new_beta
                    new_metric = self.Metric_Cal(x, y, beta)
                    delta_loss = np.abs(new_loss - loss)
                    
                    _delta = copy.deepcopy(self.delta) # reset delta
                    iter_time = iter_time + 1
                    time_series.append(iter_time)
                    loss_series.append(new_loss)
                    metric_series.append(new_metric)
                else :
                    _delta = _delta * self.backtracking_tau

        if self.plot == True :
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))
            ax1.scatter(time_series, loss_series, color = 'red')
            ax1.plot(time_series, loss_series, color = 'red', label = 'Loss')
            ax1.set_xlabel('Iteration Times')
            ax1.set_ylabel('Loss')
            ax1.legend()

            ax2.scatter(time_series, metric_series, color = 'red')
            ax2.plot(time_series, metric_series, color = 'red', label = 'Metric')
            ax2.set_xlabel('Iteration Times')
            ax2.set_ylabel('Metric')
            ax2.legend()
            plt.show()
        
        if delta_loss > self.loss_limit and iter_time >= self.max_iter :
            print('Maximum Iteration reached, last change in loss function is', delta_loss)
        
        self.beta = beta
        self.loss = new_loss
        self.metric = new_metric

    def predict(self, x) :
        if len(x[0]) != len(self.beta) - 1:
            raise ValueError('Length of inputs not matching regression model!')
        
        x0 = []
        for row in x:
            x0.append(np.insert(row, 0, 1).tolist())
        return [round(np.dot(np.array(item).T, self.beta)[0], 4) for item in x0] 

from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor