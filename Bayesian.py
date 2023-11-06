import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi2
import corner
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
import arviz
import pandas

#Set numpyro to use 4 CPU cores
numpyro.set_host_device_count(4)

#define the true function

def f(x):
    y = 3 * x - 5.5  #a = 3, b = -5.5
    return(y)

#compute values of f at linear x intervals

true_dist = np.zeros(20)
for i in range(len(true_dist)):
    true_dist[i] = f(i)
  
#create data normally distributed around true data points  
  
data = np.zeros(len(true_dist))
for i in range(len(true_dist)):
    data[i] = np.random.normal(true_dist[i], abs(4))   

print(data)

#generate error bars for data

y_err = np.zeros(len(true_dist))
for i in range(len(true_dist)):
    y_err[i] = abs(np.random.normal(abs(4), abs(0.5)))

#create x array for plotting

x = np.zeros(len(true_dist))
for i in range(len(true_dist)):
    x[i] = i
    
#plot intial data
    
plt.errorbar(x, data, yerr=y_err, fmt=".")
plt.show()

#define function to set up models

def model(a,b,x):
    y = a * x + b
    return(y)

#set the tested values of a
tests = 100

a_vals = np.zeros(tests)
for i in range(tests):
    a_vals[i] = i * 0.01 + 2.5 

#and b
b_vals = np.zeros(tests)
for i in range(tests):
    b_vals[i] = 0.1 * i - 10

#define a fitting function
def fit(a_in,b_in):
    chi2_list = np.zeros(len(true_dist))
    for i in range(len(true_dist)):
        m = model(a_in, b_in, x[i])
        chi2_list[i] = ((data[i] - m)**2) / (y_err[i]**2)
    chi_2 = np.sum(chi2_list)
    return(chi_2)

#set up an array containing the fit properties
fits = np.zeros([tests,tests])

#fit models for varying a and b
best_fit = [0, 0, 1000]
for i in range(tests):
    for j in range(tests):
        current_fit = [a_vals[i], b_vals[j], fit(a_vals[i], b_vals[j])]
        fits[j,i] = current_fit[2]
        if current_fit[2] < best_fit[2]:
            best_fit = current_fit
    ten_percent = tests / 10
    if (i % ten_percent) == 0:
        completion = i / ten_percent * 10
        print(str(completion) + "%")

#set up array of best fit model   
print("Brute force best fit:")         
print(best_fit)
best_m = np.zeros(len(true_dist))      
for i in range(len(true_dist)):
    best_m[i] = model(best_fit[0], best_fit[1], x[i])

#plot model with data
plt.errorbar(x, data, yerr=y_err, fmt=".")
plt.plot(x, best_m)
plt.show()

#plot contours
contour1 = chi2.ppf(0.68, 2) + best_fit[2]
contour2 = chi2.ppf(0.95, 2) + best_fit[2]
#contour3 = chi2.ppf(0.99, 2) + best_fit[2]
contours = [contour1, contour2]
print("Contours:")
print(contours)
a_best = [best_fit[0]]
b_best = [best_fit[1]]
plt.contour(a_vals, b_vals, fits, levels=contours, algorithm="threaded")
#plt.scatter(a_best, b_best)
plt.xlabel("a")
plt.ylabel("b")
plt.show()

#------------------------------------------------------------------------------

#Set up MCMC
steps = 10000
guess = np.zeros([steps+1, 3])
guess[0,0] = 17
guess[0,1] = 50
guess[0,2] = fit(guess[0,0], guess[0,1])
theta = guess[0]

#MCMC loop
for i in range(steps):
    guess[i+1,0] = theta[0] + np.random.normal(0, 0.1)
    guess[i+1,1] = theta[1] + np.random.normal(0, 1)
    guess[i+1,2] = fit(guess[i+1,0], guess[i+1,1])
    if guess[i+1,2] <= theta[2]:
        theta = guess[i+1]
    else:
        rho = np.exp(-1 * theta[2] / guess[i+1,2])
        r = np.random.uniform(0, 1)
        if r < rho:
            theta = guess[i+1]

#Find and output best fit values
MCMC_best = np.min(guess, axis=0)
min_i = np.argmin(guess, axis=0)
MCMC_best[0] = guess[min_i[2],0]
MCMC_best[1] = guess[min_i[2],1]
print("MCMC best fit:")
print(MCMC_best)

#Plot guess locations in range of grid search
guess = guess.transpose()
plt.scatter(guess[0], guess[1], marker=".")
plt.contour(a_vals, b_vals, fits, levels=contours, algorithm="threaded")
plt.xlabel("a")
plt.ylabel("b")
plt.xlim(2.5,3.5)
plt.ylim(-10,0)
plt.show()

#Plot all guess locations
plt.scatter(guess[0], guess[1], marker=".")
plt.xlabel("a")
plt.ylabel("b")
plt.show()

#Corner plot
corner_data = np.delete(guess, obj=2, axis=0)
corner_data = corner_data.transpose()
figure = corner.corner(corner_data, labels=["a","b"], range=[(2,4),(-15,5)])
plt.show()

#------------------------------------------------------------------------------
#Numpyro fitting following Jupyter notebook

#Mean and std of data
x_mean = x.mean()
x_std = x.std()
y_mean = data.mean()
y_std = data.std()

#Centre and scale data
x_center = jnp.array((x - x_mean) / x_std)
y_center = jnp.array((data - y_mean) / y_std)
sy_center = jnp.array(y_err / y_std)

#Order data
idx = x.argsort()

#Define function to uncentre data
def un_center(b0_prime, b1_prime, x_mean, x_std, y_mean, y_std):
    b0 = (b0_prime * y_std) - (b1_prime * x_mean * y_std/x_std) + y_mean
    x = x_mean - (x_std / b1_prime) * ((y_mean / y_std) + b0_prime)
    b1 = -b0 / x
    return b0, b1

#Plot centred data
plt.errorbar(x_center, y_center, sy_center, fmt=".")
plt.xlabel('Centered x')
plt.ylabel('Centered y')
plt.show()

#Define an OLS model
def mdl_ols():
    theta = numpyro.sample('theta', dist.Uniform(-0.5 * jnp.pi, 0.5 * jnp.pi))
    b_perp = numpyro.sample('b_perp', dist.Normal(0.0, 10.0))

    # convert to centered slope and intercept
    b1 = numpyro.deterministic('slope-centered', jnp.tan(theta))
    b0 = numpyro.deterministic('intercept-centered', b_perp / jnp.cos(theta))

    # un-center slope and intercept as Deterministic variables so we can plot them later
    b0_orig, b1_orig = un_center(b0, b1, x_mean, x_std, y_mean, y_std)
    numpyro.deterministic('intercept', b0_orig)
    numpyro.deterministic('slope', b1_orig)

    with numpyro.plate(f'vectors [{len(y_center)}]', len(y_center)):
        y_est_centered = numpyro.deterministic('y-centered', b0 + b1 * x_center)
        # un-center y_est for plotting later on
        numpyro.deterministic('y', (y_est_centered * y_std) + y_mean)

        # Likelihood
        numpyro.sample('likelihood', dist.Normal(y_est_centered, sy_center), obs=y_center)

numpyro.render_model(mdl_ols, render_distributions=True)

# Set a random seed
rng_key = jax.random.PRNGKey(0)

# Rotate the random seed
rng_key, rng_key_ = jax.random.split(rng_key)

#Run HMC
kernel = infer.NUTS(mdl_ols)
mcmc = infer.MCMC(kernel, num_warmup=1000, num_samples=2500, num_chains=4)
mcmc.run(rng_key_)

#Check for convergence
traces_ols = arviz.from_numpyro(mcmc)
print(f'divergences: {traces_ols.sample_stats.diverging.values.sum()}')
summary = arviz.summary(
    traces_ols,
    var_names=['intercept', 'slope']
)
print(summary)

#Plot traces
arviz.plot_trace(
    traces_ols,
    figsize=(15, 6),
    var_names=['intercept', 'slope']
);

#Corner plot for centred data
ax = arviz.plot_pair(
    traces_ols,
    var_names=['intercept-centered', 'slope-centered'],
    marginals=True,
    kind='kde',
    figsize=[6, 6]
)

#Corner plot for uncentred data
ax = arviz.plot_pair(
    traces_ols,
    var_names=['intercept', 'slope'],
    marginals=True,
    kind='kde',
    figsize=[6, 6]
)
plt.show()

#Set kwargs for error range
fill_kwargs = {
    'color': 'C0',
    'alpha': 0.3,
    'zorder': 1
}

# grab un-centered y-values from the sampler
y_est = traces_ols.posterior.y.data.reshape(-1, 20)

# plot original data
plt.figure(2, figsize=(10, 6))
plt.errorbar(x, data, y_err, fmt=".")

# find 2-sigma and median of best fit lines
y_est_minus_2_sigma, y_est_median, y_est_plus_2_sigma = np.percentile(
    y_est[:, idx],
    [2.5, 50, 97.5],
    axis=0
)

# plot the median of all best fit lines
plt.plot(x[idx], y_est_median, color='C3', lw=3, zorder=3)

# plot 2-sigma best fit region
plt.fill_between(x[idx], y_est_minus_2_sigma, y_est_plus_2_sigma, **fill_kwargs)

# plot a selection of best fit lines
plt.plot(x[idx], y_est[::800].T[idx], alpha=0.5, color='C7')

plt.xlabel('x')
plt.ylabel('y')
