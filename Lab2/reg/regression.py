import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here

    mu = np.zeros(2)
    Cov = np.array([[beta, 0], [0, beta]])

    num_x = 100
    num_y = 100

    xvalues = np.linspace(-1, 1, num = num_x)
    yvalues = np.linspace(-1, 1, num = num_y)
    X_grid, Y_grid = np.meshgrid(xvalues, yvalues)

    samples = np.column_stack((X_grid.flatten(), Y_grid.flatten()))

    density = util.density_Gaussian(mu, Cov, samples)
    density_grid = np.reshape(density, (num_x, num_y))

    plt.figure(1)
    plt.title("Prior Distribution of α")
    plt.xlabel('$α_0$')
    plt.ylabel('$α_1$')
    plt.contour(X_grid, Y_grid, density_grid, cmap=plt.cm.winter)
    plt.scatter(-0.1, -0.5, c='r')
    plt.show()
    
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here

    # mean of posterior distribution is the MAP estimate of the weights a
    # tau^2(from notes) is beta

    extra_col = np.ones((x.shape[0], 1))
    x = np.append(extra_col, x, axis = 1)

    alpha_map = np.linalg.inv((np.transpose(x)@x + (sigma2/beta)*np.eye(2)))@(np.transpose(x)@z)
    mu = alpha_map

    Cov = np.linalg.inv((np.transpose(x)@x + (sigma2/beta)*np.eye(2)))*sigma2

    num_x = 100
    num_y = 100

    xvalues = np.linspace(-1, 1, num = num_x)
    yvalues = np.linspace(-1, 1, num = num_y)
    X_grid, Y_grid = np.meshgrid(xvalues, yvalues)

    samples = np.column_stack((X_grid.flatten(), Y_grid.flatten()))

    density = util.density_Gaussian(mu.squeeze(), Cov, samples)
    density_grid = np.reshape(density, (num_x, num_y))

    plt.figure(1)
    plt.title("Posterior Distribution of α Given 5 Data Points")
    plt.xlabel('$α_0$')
    plt.ylabel('$α_1$')
    plt.scatter(-0.1, -0.5, c='r')
    plt.contour(X_grid, Y_grid, density_grid, cmap=plt.cm.winter)
    plt.show()

    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    X = np.array(x)
    X = X[:, np.newaxis]
    extra_col = np.ones((X.shape[0], 1))
    X = np.append(extra_col, X, axis = 1)

    z_new = []
    error = []

    for x_new in X:
        z_new.append((x_new@mu).item())
        error.append(np.sqrt(abs(((np.transpose(x_new))@Cov@x_new).item() + sigma2)))

    plt.figure(1)
    plt.title("Predicted and Training Data")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([-4, 4, -4, 4])
    plt.plot(np.array(x), z_new, 'bo')
    plt.errorbar(np.array(x), z_new, yerr=error, c = 'b', ecolor='k')
    plt.scatter(x_train, z_train,c='r')
    plt.legend(['Predictions', 'Training Data'])
    plt.show()
    
    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns  = 5
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    


    
    
    

    
