import numpy as np
import matplotlib.pyplot as plt

x = np.array([150, 160, 170, 180, 190])  # Height 
y = np.array([50, 60, 65, 70, 80])      # Weight 

# Calculate the mean of x and y
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate the slope (m) and intercept (c) manually
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
m = numerator / denominator
c = y_mean - m * x_mean

# Predicted y values based on the regression line
y_pred = m * x + c

# Calculate Mean Squared Error (MSE)
mse = np.mean((y - y_pred) ** 2)


# Display results
print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")
print(f"Mean Squared Error (MSE): {mse}")


# Plot the data points and the regression line
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs. Weight: Linear Regression')
plt.legend()
plt.show()
