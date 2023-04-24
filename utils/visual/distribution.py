#!/usr/bin/env python
# Created by "Thieu" at 10:36, 01/03/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import matplotlib.pyplot as plt
import numpy as np

# Generate some random data
error_data = np.random.normal(loc=0, scale=1, size=1000)

# Plot a histogram of the error distribution
plt.hist(error_data, bins=50, density=True, alpha=0.6, color='g')
plt.axvline(x=0, color='k', linestyle='--', label="Zero line")  # add a vertical line at zero
plt.xlabel('% Error')
plt.ylabel('% Instances')
plt.title('Error Distribution')

plt.legend()  # show the legend
plt.show()
