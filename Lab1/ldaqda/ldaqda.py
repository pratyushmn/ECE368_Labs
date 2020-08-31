import numpy as np
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    ### TODO: Write your code here

    male_indices = []
    female_indices = []

    for i in range(len(y)):
        if y[i] == 1:
            male_indices.append(i)
        elif y[i] == 2:
            female_indices.append(i)

    male_idx = np.asarray(male_indices)
    num_males = len(male_idx)

    female_idx = np.asarray(female_indices)
    num_females = len(female_idx)

    male_data = x[male_idx, :]
    female_data = x[female_idx, :]

    mu_male = (1/num_males)*np.sum(male_data, axis=0)
    mu_female = (1/num_females)*np.sum(female_data, axis=0)

    cov_male = 0

    for i in range(num_males):
        cov_male += ((x[male_idx[i], :] - mu_male)[:, np.newaxis]) @ (np.transpose(((x[male_idx[i], :] - mu_male)[:, np.newaxis])))

    cov_male = (1/(num_males))*cov_male

    cov_female = 0

    for i in range(num_males):
        cov_female += ((x[female_idx[i], :] - mu_female)[:, np.newaxis]) @ (np.transpose(((x[female_idx[i], :] - mu_female)[:, np.newaxis])))

    cov_female = (1/(num_females))*cov_female

    cov = (1/(num_females+num_males))*(num_females*cov_female + num_males*cov_male) 

    # Plotting LDA:
    num_h = 310
    num_w = 2010

    hvalues = np.linspace(50, 80, num = num_h)
    wvalues = np.linspace(80, 280, num = num_w)

    H_grid, W_grid = np.meshgrid(hvalues, wvalues)
    samples = np.column_stack((H_grid.flatten(), W_grid.flatten()))

    M_density = util.density_Gaussian(mu_male, cov, samples)
    M_density_grid = np.reshape(M_density, (num_w, num_h))

    F_density = util.density_Gaussian(mu_female, cov, samples)
    F_density_grid = np.reshape(F_density, (num_w, num_h))

    plt.figure(1)
    plt.scatter(male_data[:, 0], male_data[:, 1], c='b')
    plt.scatter(female_data[:, 0], female_data[:, 1], c='r')
    plt.contour(H_grid, W_grid, M_density_grid, cmap=plt.cm.winter)
    plt.contour(H_grid, W_grid, F_density_grid, cmap=plt.cm.spring)
    plt.contour(H_grid, W_grid, M_density_grid - F_density_grid, 0, cmap=plt.cm.copper)
    
    # Plotting QDA:
    M_density = util.density_Gaussian(mu_male, cov_male, samples)
    M_density_grid = np.reshape(M_density, (num_w, num_h))

    F_density = util.density_Gaussian(mu_female, cov_female, samples)
    F_density_grid = np.reshape(F_density, (num_w, num_h))

    decision_boundary = np.empty_like(M_density_grid)

    for i in range(num_w): # number of rows
        for j in range(num_h): # number of columns 
            x_vec = np.asarray([H_grid[i, j], W_grid[i, j]])

            decision_boundary[i, j] = ((np.math.log(0.5) - 0.5*np.transpose(mu_male)@np.linalg.inv(cov_male)@mu_male + np.transpose(mu_male)@np.linalg.inv(cov_male)@x_vec[:, np.newaxis]).flatten()[0]) - ((np.math.log(0.5) - 0.5*np.transpose(mu_female)@np.linalg.inv(cov_female)@mu_female + np.transpose(mu_female)@np.linalg.inv(cov_female)@x_vec[:, np.newaxis]).flatten()[0])
    
    plt.figure(2)
    plt.scatter(male_data[:, 0], male_data[:, 1], c='b')
    plt.scatter(female_data[:, 0], female_data[:, 1], c='r')
    plt.contour(H_grid, W_grid, M_density_grid, cmap=plt.cm.winter)
    plt.contour(H_grid, W_grid, F_density_grid, cmap=plt.cm.spring)
    plt.contour(H_grid, W_grid, M_density_grid - F_density_grid, 0, cmap=plt.cm.copper)
    plt.show()


    return (mu_male,mu_female,cov,cov_male,cov_female)
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here

    mu_male = mu_male[:, np.newaxis]
    mu_female = mu_female[:, np.newaxis]
    N = len(y)

    LDA_preds = []
    QDA_preds = []

    for i in range(N):
        p_male_LDA = (np.math.log(0.5) - 0.5*np.transpose(mu_male)@np.linalg.inv(cov)@mu_male + np.transpose(mu_male)@np.linalg.inv(cov)@x[i, :][:, np.newaxis]).flatten()[0]
        p_female_LDA = (np.math.log(0.5) - 0.5*np.transpose(mu_female)@np.linalg.inv(cov)@mu_female + np.transpose(mu_female)@np.linalg.inv(cov)@x[i, :][:, np.newaxis]).flatten()[0]

        p_male_QDA = np.math.log(0.5) - 0.5*np.math.log(np.linalg.det(cov_male)) - 0.5*np.transpose((x[i, :][:, np.newaxis] - mu_male))@np.linalg.inv(cov_male)@(x[i, :][:, np.newaxis] - mu_male)
        p_female_QDA = np.math.log(0.5) - 0.5*np.math.log(np.linalg.det(cov_female)) - 0.5*np.transpose((x[i, :][:, np.newaxis] - mu_female))@np.linalg.inv(cov_female)@(x[i, :][:, np.newaxis] - mu_female)       

        if p_male_LDA > p_female_LDA:
            LDA_preds.append(1)
        else:
            LDA_preds.append(2)

        if p_male_QDA > p_female_QDA:
            QDA_preds.append(1)
        else:
            QDA_preds.append(2)

    LDA_preds = np.asarray(LDA_preds)
    QDA_preds = np.asarray(QDA_preds)

    mis_lda = (N - np.count_nonzero(y == LDA_preds))/N
    mis_qda = (N - np.count_nonzero(y == QDA_preds))/N

    
    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    print(mis_LDA)
    print(mis_QDA)
    

    
    
    

    
