import numpy as np

def gaussian_kernel(x):
    d = x.shape[0]
    return (1 / (2 * np.pi) ** (d / 2)) * np.exp(-0.5 * np.dot(x, x))

def multivariate_kde(X, H):
    n, d = X.shape
    Hinv = np.linalg.inv(H)
    Hdet = np.linalg.det(H)
    
    def kde(x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        N = x.shape[0]
        result = np.zeros(N)
        
        for i in range(N):
            diff = x[i] - X
            tdiff = np.dot(diff, Hinv)
            energy = np.sum(diff * tdiff, axis=1)
            result[i] = np.sum(np.exp(-0.5 * energy))
        
        result /= (n * Hdet ** 0.5)
        return result[0]
    
    return kde

def scotts_rule(X):
    n, d = X.shape
    sigma = np.std(X, axis=0)
    h = n ** (-1 / (d + 4)) * sigma
    return np.diag(h)

def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

def find_local_maxima(kde, X, num_restarts=10, max_iter=100, tol=1e-2, step_size=0.1):
    n, d = X.shape
    local_maxima = []
    
    for _ in range(num_restarts):
        x = X[np.random.choice(n)]
        for _ in range(max_iter):
            grad = numerical_gradient(kde, x)
            x_new = x + step_size * grad
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        local_maxima.append(x)
    
    return np.unique(np.array(local_maxima).round(decimals=-int(np.log10(tol))), axis=0)

def initiate_kde(X:np.array, n_neurons:int):
    kde = multivariate_kde(X, scotts_rule(X))
    local_max = find_local_maxima(kde, X)
    assert local_max.shape[0] > n_neurons, "Maximum number of neurons is {}".format(local_max.shape[0])
    return local_max[np.random.choice(len(local_max), size=n_neurons, replace=False)]