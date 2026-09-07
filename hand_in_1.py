#%%
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

#%% a)

def muller_brown_potential(x, y):
    """
    Müller-Brown potential function.
    
    Parameters:
    x, y : float or array-like
        Coordinates in 2D space
    
    Returns:
    V : float or array-like
        Potential energy at (x, y)
    """
    # Müller-Brown parameters
    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])
    
    V = 0
    for i in range(4):
        V += A[i] * np.exp(a[i] * (x - x0[i])**2 + b[i] * (x - x0[i]) * (y - y0[i]) + c[i] * (y - y0[i])**2)
    
    return V

def gibbs_distribution(X):
    x, y = X
    return np.exp(-muller_brown_potential(x, y))

def metropolis_hastings(n_samples, X_0):
    samples = np.zeros((n_samples, 2))
    samples[0] = X_0

    for i in range(n_samples-1):
        prop_distribution = np.random.normal

        x = samples[i]
        x_new = prop_distribution(x, 0.5)  

        alpha = min(1, gibbs_distribution(x_new) / gibbs_distribution(x))

        u = np.random.uniform(0, 1)
        if u <= alpha:
            samples[i + 1] = x_new
        else:
            samples[i + 1] = x

    return samples


#%% b) 
X_0 = np.array([-0.5, 1.4])

samples = metropolis_hastings(10000, X_0)

#%% c)

j_vals = [1, 10, 100]

for j in j_vals:

    pear_corr = stats.pearsonr(samples[:-j], samples[j:])[0]

    print(f"{pear_corr}")

# %% d)
ref_mean =  -0.46, 1.25
ref_std = 0.38, 0.46

sample_mean = np.mean(samples, axis=0)
sample_std = np.std(samples, axis=0)

print(sample_mean)
print(sample_std)

# %%
corr_vals = []
for k in range(1, len(samples)-2):

    corr_vals.append(stats.pearsonr(samples[:-k], samples[k:])[0])
    if corr_vals[-1][0] < 0.1:
        print(f"k = {k}")
        break
# %%
print(len(corr_vals))

corr_mean_xy = np.mean(corr_vals, axis=1)
print(corr_mean_xy[0])
plt.plot(corr_mean_xy)
plt.show()


corr_sum = np.sum(corr_mean_xy)
# corr_sum = np.sum(corr_vals)

ess = len(samples) / np.abs((1 + 2 * corr_sum))
print(ess)

#%% 4
#
# In this exercise you will implement a regularized polynomial regression.
# You will need some basic numerical linear algebra to solve this problem,
# it can be solved in three lines of code.
#
# Hint: you rephrase it as a system of linear equations and solve it with
# numpys lstsq-function.
#
# fit a polynomial of degree k to data (x,y) with regularization parameter lamb
def get_reg_poly_model(x, y, k, lamb=0):
    X = np.vander(x, k+1)
    reg_matrix = lamb * np.eye(k + 1)
    coefficients = np.linalg.lstsq(X.T @ X / len(x) + reg_matrix, X.T @ y / len(x))[0]

    return coefficients


if __name__ == "__main__":
    #generate data
    x=np.linspace(-1,1,32)
    y=x**2 + np.random.normal(0,0.1,32)
    # fit unregularized model with numpy
    ureg = np.polyfit(x,y,24)
    # ureg = get_reg_poly_model(x,y,24, lamb=0)
    # fit regularized model
    reg_poly = get_reg_poly_model(x,y,24, lamb=.01)
    fig,ax=plt.subplots(1, figsize=(4,3))
    # build plot
    ax.plot(np.linspace(-1,1,1024), np.polyval(ureg, np.linspace(-1,1,1024)), label='unregularized')

    ax.plot(np.linspace(-1,1,1024), np.polyval([1,0,0], np.linspace(-1,1,1024)), label='true', linestyle='dashed', color='r')


    ax.plot(np.linspace(-1,1,1024), np.polyval(reg_poly, np.linspace(-1,1,1024)), label='regularized')

    ax.scatter(x,y, color='k', s=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_ylim(-0.5,1.5)
    plt.tight_layout()
# %%
