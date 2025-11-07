# #1.  Accept list of numbers from user
# numbers = list(map(int, input("Enter numbers separated by space: ").split()))

# # Find largest, smallest, and average
# largest = max(numbers)
# smallest = min(numbers)
# average = sum(numbers) / len(numbers)

# # Sort the list
# ascending = sorted(numbers)
# descending = sorted(numbers, reverse=True)

# # Display results
# print("Numbers:", numbers)
# print("Largest number:", largest)
# print("Smallest number:", smallest)
# print("Average value:", average)
# print("Ascending order:", ascending)
# print("Descending order:", descending)






# #2. Accept two integers
# a = int(input("Enter first number: "))
# b = int(input("Enter second number: "))

# # Arithmetic operations
# print("\n--- Arithmetic Operations ---")
# print("Addition:", a + b)
# print("Subtraction:", a - b)
# print("Multiplication:", a * b)
# print("Division:", a / b)
# print("Floor Division:", a // b)
# print("Modulus:", a % b)
# print("Power:", a ** b)

# # Relational operations
# print("\n--- Relational Operations ---")
# print("a > b:", a > b)
# print("a < b:", a < b)
# print("a == b:", a == b)
# print("a != b:", a != b)
# print("a >= b:", a >= b)
# print("a <= b:", a <= b)

# # Logical operations
# print("\n--- Logical Operations ---")
# print("(a > 0) and (b > 0):", (a > 0) and (b > 0))
# print("(a > 0) or (b > 0):", (a > 0) or (b > 0))
# print("not(a > 0):", not (a > 0))

# # Data types
# print("\n--- Data Types ---")
# print("Type of a:", type(a))
# print("Type of b:", type(b))

# # Type casting
# print("\n--- Type Casting ---")
# f = float(a)       # int - float
# s = str(f)         # float - string

# print("Integer:", a, "| Type:", type(a))
# print("After casting to float:", f, "| Type:", type(f))
# print("After casting to string:", s, "| Type:", type(s))







# #3. Input a string from the user
# text = input("Enter a string: ")

# # Initialize counters
# vowels = 0
# consonants = 0
# digits = 0
# special_chars = 0

# # Define vowel set
# vowel_set = "aeiouAEIOU"

# # Loop through each character
# for ch in text:
#     if ch.isalpha():                     # check if alphabet
#         if ch in vowel_set:
#             vowels += 1
#         else:
#             consonants += 1
#     elif ch.isdigit():                   # check if digit
#         digits += 1
#     else:                                # everything else = special char
#         special_chars += 1

# # Display results
# print("\n--- Character Count ---")
# print("Vowels:", vowels)
# print("Consonants:", consonants)
# print("Digits:", digits)
# print("Special Characters:", special_chars)






# 4. Import numpy library
# import numpy as np

# # Create 1D and 2D arrays
# arr1 = np.array([10, 20, 30, 40, 50])
# arr2 = np.array([[1, 2, 3], [4, 5, 6]])

# print("1D Array:\n", arr1)
# print("2D Array:\n", arr2)

# # Arithmetic operations (element-wise)
# print("\n--- Arithmetic Operations ---")
# print("Addition:", arr1 + 5)
# print("Subtraction:", arr1 - 5)
# print("Multiplication:", arr1 * 2)
# print("Division:", arr1 / 2)

# # Statistical functions
# print("\n--- Statistics on 1D Array ---")
# print("Maximum:", np.max(arr1))
# print("Minimum:", np.min(arr1))
# print("Mean:", np.mean(arr1))
# print("Standard Deviation:", np.std(arr1))

# print("\n--- Statistics on 2D Array ---")
# print("Maximum:", np.max(arr2))
# print("Minimum:", np.min(arr2))
# print("Mean:", np.mean(arr2))
# print("Standard Deviation:", np.std(arr2))






# 5.
# import pandas as pd
# import matplotlib.pyplot as plt

# # Create a DataFrame with student marks
# data = {
#     'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
#     'Maths': [85, 70, 95, 60, 80],
#     'Science': [90, 75, 85, 65, 88],
#     'English': [88, 72, 91, 58, 84]
# }

# df = pd.DataFrame(data)

# # Calculate total and average
# df['Total'] = df['Maths'] + df['Science'] + df['English']
# df['Average'] = df['Total'] / 3

# # Function to assign grades
# def get_grade(avg):
#     if avg >= 90:
#         return 'A+'
#     elif avg >= 80:
#         return 'A'
#     elif avg >= 70:
#         return 'B'
#     elif avg >= 60:
#         return 'C'
#     else:
#         return 'F'

# # Apply the grade function to each row
# df['Grade'] = df['Average'].apply(get_grade)

# # Display the DataFrame
# print("\n--- Student Marks DataFrame ---")
# print(df)

# # Plot: Student Name vs Average Marks
# plt.bar(df['Name'], df['Average'], color='skyblue')
# plt.title("Student Average Marks")
# plt.xlabel("Student Name")
# plt.ylabel("Average Marks")
# plt.show()






# 6. 
# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Step 1: Load dataset (you can use your own CSV file)
# # For demo, let's reuse the student_marks.csv from previous experiment
# df = pd.read_csv("student_marks.csv")

# print("\n--- Original Dataset ---")
# print(df)

# # Step 2: Handle missing/null values
# # (Assume some missing values — fill with mean or median)
# df.fillna(df.mean(numeric_only=True), inplace=True)

# print("\n--- After Handling Missing Values ---")
# print(df)

# # Step 3: Normalize data using Min-Max normalization
# # Formula: (x - min) / (max - min)
# normalized_df = df.copy()
# for column in ['Maths', 'Science', 'English', 'Total', 'Average']:
#     min_val = df[column].min()
#     max_val = df[column].max()
#     normalized_df[column] = (df[column] - min_val) / (max_val - min_val)

# print("\n--- Normalized Data ---")
# print(normalized_df)

# # Step 4: Split into training (80%) and testing (20%)
# train, test = train_test_split(normalized_df, test_size=0.2, random_state=42)

# print("\n--- Training Set ---")
# print(train)
# print("\n--- Testing Set ---")
# print(test)






# 7. 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_squared_error

# # Step 1: Create a sample dataset (student study hours vs marks)
# data = {
#     'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'Scores': [35, 40, 50, 55, 60, 65, 70, 75, 80, 85]
# }

# df = pd.DataFrame(data)

# # Step 2: Split dataset into features (X) and target (y)
# X = df[['Hours']]  # independent variable
# y = df['Scores']   # dependent variable

# # Step 3: Split into training (80%) and testing (20%)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 4: Train Linear Regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Step 5: Get slope (m) and intercept (c)
# m = model.coef_[0]
# c = model.intercept_

# print(f"Regression Equation: y = {m:.2f}x + {c:.2f}")

# # Step 6: Predict and visualize
# y_pred = model.predict(X_test)

# plt.scatter(X, y, color='blue', label='Actual Data')
# plt.plot(X, model.predict(X), color='red', label='Best Fit Line')
# plt.title("Linear Regression: Study Hours vs Scores")
# plt.xlabel("Hours Studied")
# plt.ylabel("Scores")
# plt.legend()
# plt.show()

# # Step 7: Evaluate model
# r2 = r2_score(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# print(f"R² Score: {r2:.4f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")






# 8.
# Import necessary libraries
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# import pandas as pd

# # Load the Iris dataset
# iris = load_iris()
# X = iris.data
# y = iris.target

# # Split dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # ---- DECISION TREE MODEL ----
# dt_model = DecisionTreeClassifier(random_state=42)
# dt_model.fit(X_train, y_train)
# dt_pred = dt_model.predict(X_test)

# # ---- KNN MODEL ----
# knn_model = KNeighborsClassifier(n_neighbors=3)
# knn_model.fit(X_train, y_train)
# knn_pred = knn_model.predict(X_test)

# # ---- ACCURACY ----
# dt_accuracy = accuracy_score(y_test, dt_pred)
# knn_accuracy = accuracy_score(y_test, knn_pred)

# # ---- CONFUSION MATRICES ----
# dt_cm = confusion_matrix(y_test, dt_pred)
# knn_cm = confusion_matrix(y_test, knn_pred)

# # ---- PREDICTION FOR A NEW DATA POINT ----
# # Example data point: [sepal length, sepal width, petal length, petal width]
# new_data = [[5.1, 3.5, 1.4, 0.2]]

# dt_new_pred = iris.target_names[dt_model.predict(new_data)[0]]
# knn_new_pred = iris.target_names[knn_model.predict(new_data)[0]]

# # ---- DISPLAY RESULTS ----
# print("===== Decision Tree Results =====")
# print(f"Accuracy: {dt_accuracy:.2f}")
# print("Confusion Matrix:\n", dt_cm)
# print(f"Prediction for {new_data}: {dt_new_pred}")

# print("\n===== KNN Results =====")
# print(f"Accuracy: {knn_accuracy:.2f}")
# print("Confusion Matrix:\n", knn_cm)
# print(f"Prediction for {new_data}: {knn_new_pred}")





# 9. 
# Import necessary libraries
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     roc_curve, auc, roc_auc_score
# )

# # Load the Breast Cancer dataset (binary classification)
# data = load_breast_cancer()
# X = data.data
# y = data.target

# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Create and train Logistic Regression model
# model = LogisticRegression(max_iter=10000)
# model.fit(X_train, y_train)

# # Predict test data
# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)[:, 1]  # Probability estimates

# # ---- Calculate Evaluation Metrics ----
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# auc_score = roc_auc_score(y_test, y_prob)

# # ---- ROC Curve ----
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# # ---- Display Results ----
# print("===== Logistic Regression Evaluation =====")
# print(f"Accuracy:  {accuracy:.3f}")
# print(f"Precision: {precision:.3f}")
# print(f"Recall:    {recall:.3f}")
# print(f"F1-Score:  {f1:.3f}")
# print(f"AUC Score: {auc_score:.3f}")

# # ---- Plot ROC Curve ----
# plt.figure(figsize=(7, 5))
# plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.3f})')
# plt.plot([0, 1], [0, 1], color='red', linestyle='--')
# plt.title("ROC Curve - Logistic Regression")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend()
# plt.grid(True)
# plt.show()






# 10. 
# Import required libraries
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# # Load the Iris dataset
# iris = load_iris()
# X = iris.data
# y = iris.target
# target_names = iris.target_names

# # Step 1: Standardize the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Step 2: Apply PCA (reduce to 2 components for visualization)
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# # Step 3: Display explained variance ratio
# print("Explained Variance Ratio for each component:")
# print(pca.explained_variance_ratio_)

# # Step 4: Plot before PCA (original 4D reduced to first 2 features)
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis')
# plt.title("Before PCA (Original Data)")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")

# # Step 5: Plot after PCA transformation
# plt.subplot(1, 2, 2)
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
# plt.title("After PCA (Reduced Data)")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")

# plt.tight_layout()
# plt.show()

