import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi2

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
print(contours)
a_best = [best_fit[0]]
b_best = [best_fit[1]]
plt.contour(a_vals, b_vals, fits, levels=contours, algorithm="threaded")
#plt.scatter(a_best, b_best)
plt.xlabel("a")
plt.ylabel("b")
plt.show()

#Set up MCMC
steps = 10000
guess = np.zeros([steps+1, 3])
guess[0,0] = 5
guess[0,1] = 0
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