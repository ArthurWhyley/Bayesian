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
    y = 3 * x - 5.5 + 6 * np.sin(x)  #a = 3, b = -5.5, c = 6
    return(y)

#compute values of f at linear x intervals

true_dist = np.zeros(20)
for i in range(len(true_dist)):
    true_dist[i] = f(i)
  
#create data normally distributed around true data points  
  
data = np.zeros(len(true_dist))
for i in range(len(true_dist)):
    data[i] = np.random.normal(true_dist[i], abs(3))   

print(data)

#generate error bars for data

y_err = np.zeros(len(true_dist))
for i in range(len(true_dist)):
    y_err[i] = abs(np.random.normal(abs(3), abs(0.5)))

#create x array for plotting

x = np.zeros(len(true_dist))
for i in range(len(true_dist)):
    x[i] = i
    
#plot intial data
    
plt.errorbar(x, data, yerr=y_err, fmt=".")
plt.show()

#define function to set up models

def model(a,b,c,x):
    y = a * x + b + c * np.sin(x)
    return(y)

#define a fitting function
def fit(a_in,b_in,c_in):
    chi2_list = np.zeros(len(true_dist))
    for i in range(len(true_dist)):
        m = model(a_in, b_in, c_in, x[i])
        chi2_list[i] = ((data[i] - m)**2) / (y_err[i]**2)
    chi_2 = np.sum(chi2_list)
    return(chi_2)

#Set up MCMC
steps = 10000
guess = np.zeros([steps+1, 4])
guess[0,0] = 17
guess[0,1] = 50
guess[0,2] = 1.5
guess[0,3] = fit(guess[0,0], guess[0,1], guess[0,2])
theta = guess[0]

#MCMC loop
for i in range(steps):
    guess[i+1,0] = theta[0] + np.random.normal(0, 0.1)
    guess[i+1,1] = theta[1] + np.random.normal(0, 1)
    guess[i+1,2] = theta[2] + np.random.normal(0, 0.1)
    guess[i+1,3] = fit(guess[i+1,0], guess[i+1,1], guess[i+1,2])
    if guess[i+1,3] <= theta[3]:
        theta = guess[i+1]
    else:
        rho = np.exp(-1 * theta[3] / guess[i+1,3])
        r = np.random.uniform(0, 1)
        if r < rho:
            theta = guess[i+1]

#Find and output best fit values
MCMC_best = np.min(guess, axis=0)
min_i = np.argmin(guess, axis=0)
MCMC_best[0] = guess[min_i[3],0]
MCMC_best[1] = guess[min_i[3],1]
MCMC_best[2] = guess[min_i[3],2]
print("MCMC best fit:")
print(MCMC_best)

#set up array of best fit model   
best_m = np.zeros(len(true_dist))      
for i in range(len(true_dist)):
    best_m[i] = model(MCMC_best[0], MCMC_best[1], MCMC_best[2], x[i])

#plot model with data
plt.errorbar(x, data, yerr=y_err, fmt=".")
plt.plot(x, best_m)
plt.show()

#Plot all guess locations
guess = guess.transpose()
plt.scatter(guess[0], guess[1], marker=".")
plt.xlabel("a")
plt.ylabel("b")
plt.show()

#Corner plot
corner_data = np.delete(guess, obj=3, axis=0)
corner_data = corner_data.transpose()
figure = corner.corner(corner_data, labels=["a","b", "c"], range=[(2,4),(-15,5), (3,9)])
plt.show()

#------------------------------------------------------------------------------

#Order data
idx = x.argsort()

#Model for numpuro fitting
def MCMC_mdl():
    a = numpyro.sample("a", dist.Normal(3.5, 1.0))
    b = numpyro.sample("b", dist.Uniform(-10, 0))
    c = numpyro.sample("c", dist.Normal(5, 2))
    with numpyro.plate("Vectors", len(true_dist)):
        y_est = numpyro.deterministic("y", a*x + b + c*np.sin(x))
        numpyro.sample("Likelihood", dist.Normal(y_est, y_err), obs=data)
        
numpyro.render_model(MCMC_mdl, render_distributions=True)

# Set a random seed
rng_key = jax.random.PRNGKey(0)

# Rotate the random seed
rng_key, rng_key_ = jax.random.split(rng_key)

#Run HMC
kernel = infer.NUTS(MCMC_mdl)
mcmc = infer.MCMC(kernel, num_warmup=2000, num_samples=3000, num_chains=4)
mcmc.run(rng_key_)

#Check for convergence
traces_ols = arviz.from_numpyro(mcmc)
print(f'divergences: {traces_ols.sample_stats.diverging.values.sum()}')
summary = arviz.summary(
    traces_ols,
    var_names=['a', 'b', "c"]
)
print(summary)

#Plot traces
arviz.plot_trace(
    traces_ols,
    figsize=(15, 6),
    var_names=['a', 'b', "c"]
);

#Corner plot
ax = arviz.plot_pair(
    traces_ols,
    var_names=['a', 'b', "c"],
    marginals=True,
    kind='kde',
    figsize=[8, 8]
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
