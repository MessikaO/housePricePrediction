import math
import numpy as np
import matplotlib.pyplot as plt

# Use a nice plotting style
plt.style.use('seaborn-v0_8-darkgrid')

# Load the data set with more data points
x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])   # Features (house sizes)
y_train = np.array([300.0, 500.0, 700.0, 900.0, 1100.0, 1300.0])  # Target values (house prices)

# Function to calculate the cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost
    return total_cost

# Function to calculate the gradient
def compute_gradient(x, y, w, b): 
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

# Function to perform gradient descent
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        b -= alpha * dj_db
        w -= alpha * dj_dw
        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, J_history, p_history

# Initialize parameters
w_init = 0
b_init = 0
# Gradient descent settings
iterations = 10000
alpha = 1.0e-2
# Run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, alpha, 
                                                    iterations, compute_cost, compute_gradient)

print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

# Plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration (start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost');  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step');  ax2.set_xlabel('iteration step') 
plt.show(block=False)

# Function to predict house prices based on the trained model
def predict_price(size, w, b):
    return w * size + b

# Get user input for house size and predict its price
house_size = float(input("Enter the size of the house: "))

predicted_price = predict_price(house_size, w_final, b_final)
print(f"The predicted price for a house of size {house_size} is: ${predicted_price:.2f}")
