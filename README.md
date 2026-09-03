## Edited by: BASAL, EARL DAVID M.
## Section: 2ECE-B

## ===================PROGRAMMING ASSIGNMENT #2===================
# PA2_NUMERICAL_PYTHON

## 1. Overview & Objectives

This document provides a comprehensive technical overview and explanation of the Python code implemented for **Experiment 2: Numerical Python (NumPy)**. The objective of this assignment is to demonstrate vectorization, array manipulation, mathematical transformation, and file operations using NumPy without using Python explicit loops (`for`/`while`) or list comprehensions.

### Key Learning Outcomes Demonstrated:
1. **Reproducible Random Array Generation:** Utilizing pseudo-random seed generation (`np.random.seed`).
2. **Vectorized Mathematical Operations:** Performing standard normal distribution normalization ($Z$-score scaling) and element-wise arithmetic across $N$-dimensional arrays (`ndarray`).
3. **Array Reshaping & Sequence Construction:** Creating integer sequences and reshaping them into multi-dimensional matrices using `np.arange` and `.reshape()`.
4. **Boolean Indexing & Filtering:** Selecting elements based on element-wise conditions (modulo divisibility and scalar comparison).
5. **Array Persistence:** Saving computed binary arrays directly to disk in `.npy` format.

---

## 2. Environment & Dependencies

* **Language:** Python 3.x
* **Core Library:** `numpy` (Imported as `np`)
* **Constraints Followed:**
  * No Python `for`/`while` loops or list comprehensions.
  * No external libraries used other than `numpy`.
  * All variable names, matrix dimensions, and saved `.npy` filenames strictly adhere to assignment specifications.

---

## 3. Detailed Problem Solutions & Code Explanation

---

### Problem A: Reproducible Normalization Problem

#### Objective
Generate a reproducible $5 	x 5$ integer matrix $X$ with values in the range $[10, 100]$, normalize it to zero mean ($\mu = 0$) and unit standard deviation ($\sigma = 1$), and save the output as `X_normalized.npy`.

#### Mathematical Formulation
$$Z = rac{X -  ar{x}}{\sigma}$$

Where:
* $X$ is the input matrix of size $5 	x 5$.
* $ ar{x} = rac{1}{N} \sum_{i=1}^{N} X_i$ is the arithmetic mean of all 25 elements.
* $\sigma = \sqrt{rac{1}{N} \sum_{i=1}^{N} (X_i -  ar{x})^2}$ is the population standard deviation (NumPy default for `.std()`).

#### Implementation Code
```python
import numpy as np

# 1. Set seed for exact reproducibility across runs
np.random.seed(2112)

# 2. Generate a 5x5 matrix of random integers from 10 to 100 inclusive
X = np.random.randint(10, 101, size=(5, 5))

# 3. Calculate mean (x_bar) and standard deviation (sigma) of all elements
x_bar = X.mean()
sigma = X.std()

# 4. Perform element-wise vectorized Z-score normalization
X_normalized = (X - x_bar) / sigma

# 5. Display required verification checks
print("--- Problem A: Verification Checks ---")
print("Original Matrix X:
", X)
print("
Normalized Matrix X_normalized:
", X_normalized)
print("
Mean of X_normalized (Expected ~ 0.0):", X_normalized.mean())
print("Std Dev of X_normalized (Expected ~ 1.0):", X_normalized.std())

# 6. Save array to binary format
np.save("X_normalized.npy", X_normalized)
```

#### Code Logic Explanation
1. **`np.random.seed(2112)`**: Sets the state of the Mersenne Twister pseudo-random number generator. This guarantees that every execution of `np.random.randint` generates the exact same sequence of pseudo-random numbers.
2. **`np.random.randint(10, 101, size=(5, 5))`**: Draws discrete uniform random integers from the half-open interval $[10, 101)$ (which includes numbers 10 through 100) and arranges them into a 2D shape of $(5, 5)$.
3. **`X.mean()` and `X.std()`**: Computes the scalar mean ($ ar{x}$) and standard deviation ($\sigma$) over all 25 elements of matrix $X$.
4. **`(X - x_bar) / sigma`**: Executes element-wise broadcasting. `x_bar` and `sigma` are broadcast across every element of $X$, subtracting the mean and dividing by standard deviation in parallel without any Python loops.
5. **`np.save("X_normalized.npy", X_normalized)`**: Writes the normalized float array in binary `.npy` format.

#### Required Checks & Output Validation
* **Mean of `X_normalized`:** $0.0$ (floating-point precision exactness)
* **Standard Deviation of `X_normalized`:** $1.0$

---

### Problem B: Cubes Divisible by 4 Problem

#### Objective
Generate the cubes of the first 100 positive integers ($1^3, 2^3, \dots, 100^3$), reshape the $1 \mathrm{D}$ array into a $10 	imes 10$ matrix $C$, extract all elements divisible by 4 using Boolean indexing, and save the result as `div_by_4.npy`.

#### Implementation Code
```python
import numpy as np

# 1. Generate integers 1 to 100, cube each element, and reshape into 10x10 array
C = (np.arange(1, 101) ** 3).reshape(10, 10)

# 2. Apply Boolean conditioning to select elements divisible by 4
div_by_4 = C[C % 4 == 0]

# 3. Display required verification checks
print("--- Problem B: Verification Checks ---")
print("Shape of C (Expected: (10, 10)):", C.shape)
print("
Filtered Array div_by_4:
", div_by_4)
print("
Number of selected elements (Expected: 50):", div_by_4.size)
print("First element (Expected: 8):", div_by_4[0])
print("Last element (Expected: 1000000):", div_by_4[-1])

# 4. Save selected array to binary format
np.save("div_by_4.npy", div_by_4)
```

#### Code Logic Explanation
1. **`np.arange(1, 101)`**: Creates a 1D array of integers from 1 to 100 inclusive (`[1, 2, ..., 100]`).
2. **`** 3`**: Element-wise exponentiation cubes each integer in the array ($1, 8, 27, \dots, 1000000$).
3. **`.reshape(10, 10)`**: Re-organizes the 100 flattened elements into a 2D matrix of 10 rows and 10 columns in row-major (C-style) order.
4. **`C % 4 == 0`**: Computes element-wise remainder when divided by 4 and returns a $10 	imes 10$ Boolean array (`True` where remainder is 0, `False` otherwise).
5. **`C[...]`**: Evaluates the Boolean mask against array $C$. It extracts only the values corresponding to `True` indices, preserving NumPy's normal row-major order and returning a 1D array `div_by_4`.
6. **`np.save("div_by_4.npy", div_by_4)`**: Saves the resulting 1D array.

#### Required Checks & Output Validation
* **Shape of $C$:** `(10, 10)`
* **Number of Selected Elements:** $50$
* **First Element:** $8$ ($2^3$)
* **Last Element:** $1,000,000$ ($100^3$)

---

### Problem C: Above-Mean Squares Problem

#### Objective
Construct a $6 	x 6$ matrix $S$ containing the squares of the first 36 positive integers ($1^2, 2^2, \dots, 36^2$). Compute the mean of all elements ($S_{	ext{mean}}$), select elements strictly greater than $S_{	ext{mean}}$, and save the output as `above_mean.npy`.

#### Implementation Code
```python
import numpy as np

# 1. Generate squares of integers 1 to 36 and reshape into 6x6 array
S = (np.arange(1, 37) ** 2).reshape(6, 6)

# 2. Compute the arithmetic mean of all 36 elements
S_mean = S.mean()

# 3. Apply Boolean filtering for elements strictly greater than S_mean
above_mean = S[S > S_mean]

# 4. Display required verification checks
print("--- Problem C: Verification Checks ---")
print("Array S:
", S)
print("
Mean of S (S_mean):", S_mean)
print("
Filtered Array above_mean:
", above_mean)
print("
Number of selected elements (Expected: 15):", above_mean.size)
print("First element (Expected: 484):", above_mean[0])
print("Last element (Expected: 1296):", above_mean[-1])

# 5. Save selected array to binary format
np.save("above_mean.npy", above_mean)
```

#### Code Logic Explanation
1. **`np.arange(1, 37) ** 2`**: Generates integers 1 through 36 and squares each term in a vectorized pass ($1, 4, 9, \dots, 1296$).
2. **`.reshape(6, 6)`**: Converts the 36-element sequence into a $6  x 6$ grid in row-major order.
3. **`S.mean()`**: Sums all 36 squared values ($\sum_{k=1}^{36} k^2 = rac{36 	x 37 	x 73}{6} = 16206$) and divides by 36, yielding $S_{	ext{mean}} = rac{16206}{36} = 450.1666666666667$.
4. **`S > S_mean`**: Evaluates scalar comparison against every element of $S$, generating a $6 	x 6$ Boolean matrix (`True` for $k^2 > 450.167$).
5. **`S[...]`**: Boolean indexing filters out all elements where the condition is `False`, returning a 1D array starting at $22^2 = 484$ up to $36^2 = 1296$.
6. **`np.save("above_mean.npy", above_mean)`**: Saves the 1D array to disk.

#### Required Checks & Output Validation
* **$S_{	ext{mean}}$:** $450.1666666666667$
* **Number of Selected Elements:** $15$
* **First Element:** $484$ ($22^2$)
* **Last Element:** $1,296$ ($36^2$)

---

## 4. Summary of Output Files

When executing the Jupyter Notebook, three `.npy` files are written to the working directory:

| Filename | Matrix/Array Description | Shape | Data Type |
| :--- | :--- | :--- | :--- |
| **`X_normalized.npy`** | Z-score normalized $5 	x 5$ random matrix | $(5, 5)$ | `float64` |
| **`div_by_4.npy`** | Cubes of $1..100$ divisible by $4$ | $(50,)$ | `int64` |
| **`above_mean.npy`** | Squares of $1..36$ strictly greater than $S_{	ext{mean}}$ | $(15,)$ | `int64` |

### How to Load and Verify Files
To load and inspect any of the saved files, run the following code:
```python
import numpy as np

# Load files
x_norm = np.load("X_normalized.npy")
div4 = np.load("div_by_4.npy")
above_m = np.load("above_mean.npy")

print("Loaded X_normalized shape:", x_norm.shape)
print("Loaded div_by_4 shape:", div4.shape)
print("Loaded above_mean shape:", above_m.shape)
```

