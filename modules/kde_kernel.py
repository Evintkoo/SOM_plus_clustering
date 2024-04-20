import numpy as np

def gaussian_kernel(x):
    d = x.shape[0]
    return (1 / (2 * np.pi) ** (d / 2)) * np.exp(-0.5 * np.dot(x, x))

def multivariate_kde(X, H):
    n, d = X.shape
    Hinv = np.linalg.inv(H)
    Hdet = np.linalg.det(H)
    Hdet_sqrt = Hdet ** 0.5
    const = 1 / (n * Hdet_sqrt)
   
    def kde(x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
       
        diff = x[:, np.newaxis, :] - X
        tdiff = np.einsum('ijk,kl->ijl', diff, Hinv)
        energy = np.einsum('ijk,ijk->ij', diff, tdiff)
        result = const * np.sum(np.exp(-0.5 * energy), axis=1)
       
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
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

def find_local_maxima(kde, X, num_restarts=100, max_iter=100, tol=1e-5, step_size=0.1):
    n, d = X.shape
    local_maxima = []
    decimals = -int(np.log10(tol))
    
    for _ in range(num_restarts):
        x = X[np.random.choice(n)]
        for _ in range(max_iter):
            grad = numerical_gradient(kde, x)
            x_new = x + step_size * grad
            if np.linalg.norm(x_new - x) < tol:
                local_maxima.append(x_new.round(decimals))
                break
            x = x_new
    
    return np.unique(local_maxima, axis=0)

def initiate_kde(X: np.array, n_neurons: int):
    kde = multivariate_kde(X, scotts_rule(X))
    local_max = find_local_maxima(kde, X)
    max_neurons = local_max.shape[0]
    if max_neurons <= n_neurons:
        raise ValueError(f"Maximum number of neurons is {max_neurons}")
    return local_max[np.random.choice(max_neurons, size=n_neurons, replace=False)]