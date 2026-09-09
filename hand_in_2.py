#%%
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt



# %% Data generation process


def sample_fun():
    size = np.random.uniform(1, 100)

    angle = np.random.uniform(0, 2 * np.pi)
    u = np.random.uniform(-1, 1)

    theta = [np.sqrt(1-u**2) * np.cos(angle), np.sqrt(1-u**2) * np.sin(angle), u]

    return size, theta


def size_filter(size):
    mu = 14
    sigma = 0.5 
    p = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((size - mu) / sigma) ** 2)
    # print(size)
    # print(p)
    u = np.random.uniform(0, 1)
    if u <= p:
        return True
    return False

def orientation_filter(theta):
    a = np.array([1, 0, 0])

    p = np.exp(-np.abs(np.dot(a, theta)))
    u = np.random.uniform(0, 1)
    if u <= p:
        return True
    return False


# %%
n_samples = 10000000

sizes = []
# thetas = []
theta_x = []
theta_y = []
theta_z = []

for n in range(n_samples):
    size, theta = sample_fun()

    passed_size = size_filter(size)
    if passed_size:

        passed_orientation = orientation_filter(theta)

        if passed_orientation:
            sizes.append(size)
            # thetas.append(theta)
            theta_x.append(theta[0])
            theta_y.append(theta[1])
            theta_z.append(theta[2])

plt.hist(sizes, bins=30, density=True)
plt.xlabel('Size')
plt.ylabel('Density')
plt.title('Distribution of Filtered Sizes')
plt.show()

plt.hist(theta_x, bins=30, density=True)
plt.xlabel('Theta X')
plt.ylabel('Density')
plt.title('Distribution of Filtered Theta X')
plt.show()

plt.hist(theta_y, bins=30, density=True)
plt.xlabel('Theta Y')
plt.ylabel('Density')
plt.title('Distribution of Filtered Theta Y')
plt.show()

plt.hist(theta_z, bins=30, density=True)
plt.xlabel('Theta Z')
plt.ylabel('Density')
plt.title('Distribution of Filtered Theta Z')
plt.show()  

# size, theta = sample_fun()

# passed_size = size_filter(size)
# passed_orientation = orientation_filter(theta)

# print(passed_size)
# print(passed_orientation)


# theta = [0, 1, 0]
# print(theta)

# print(np.exp(-np.abs(np.dot([1, 0, 0], theta))))


# %%
