import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. Load the CSV Dataset
# ==========================================

df = pd.read_csv("Iris.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# ==========================================
# 2. Dataset Information
# ==========================================

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

# ==========================================
# 3. Remove ID Column and Duplicate Rows
# ==========================================

df = df.drop(columns=["Id"])

df = df.drop_duplicates().copy()

print("\nShape After Removing Duplicates:")
print(df.shape)

print("\nDuplicate Rows After Removal:")
print(df.duplicated().sum())

# ==========================================
# 4. Rename CSV Columns
# ==========================================

df = df.rename(columns={
    "SepalLengthCm": "sepal length (cm)",
    "SepalWidthCm": "sepal width (cm)",
    "PetalLengthCm": "petal length (cm)",
    "PetalWidthCm": "petal width (cm)",
    "Species": "species"
})

# ==========================================
# 5. Convert Species Names into Numbers
# ==========================================

species_mapping = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}

species_names = [
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
]

df["species"] = df["species"].map(species_mapping)

print("\nSpecies Distribution:")
print(df["species"].value_counts().sort_index())

# ==========================================
# 6. Handle Missing Values
# ==========================================

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

df = df.dropna().copy()

print("\nShape After Cleaning:")
print(df.shape)

# ==========================================
# 7. Separate Features and Target
# ==========================================

X = df.drop("species", axis=1)
y = df["species"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)

# ==========================================
# 8. Split Training and Testing Data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining and Testing Data:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# ==========================================
# 9. Train Logistic Regression Model
# ==========================================

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

print("\nLogistic Regression Model Trained Successfully!")

# ==========================================
# 10. Predict and Evaluate the Model
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Accuracy:", accuracy)
print("Accuracy Percentage:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=species_names,
    zero_division=0
))

# ==========================================
# 11. Create and Display Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2]
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=species_names
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Iris Flower Classification - Confusion Matrix")
plt.tight_layout()
plt.show()

# ==========================================
# 12. Visualize Flower Measurements
# ==========================================

plt.figure(figsize=(8, 5))

for species_id, species_name in enumerate(species_names):

    species_data = df[df["species"] == species_id]

    plt.scatter(
        species_data["sepal length (cm)"],
        species_data["petal length (cm)"],
        label=species_name
    )

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Sepal Length vs Petal Length by Species")

plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# 13. Visualize Species Distribution
# ==========================================

species_counts = df["species"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.bar(
    species_names,
    [species_counts.get(i, 0) for i in range(3)]
)

plt.xlabel("Species")
plt.ylabel("Number of Flowers")
plt.title("Iris Flower Species Distribution")

plt.tight_layout()
plt.show()

# ==========================================
# 14. Predict a New Iris Flower
# ==========================================

new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

predicted_species = model.predict(new_flower)

print("\nNew Flower Prediction:")
print(
    "Predicted Species:",
    species_names[predicted_species[0]]
)

# ==========================================
# 15. Actionable Insights
# ==========================================

print("\nActionable Insights:")

print("1. Flower measurements help classify Iris species.")
print("2. Petal length and petal width are useful for species identification.")
print("3. Logistic Regression learns patterns from training data.")
print("4. The confusion matrix shows correct and incorrect predictions.")
print("5. Machine Learning can automate Iris flower classification.")

# ==========================================
# 16. Final Project Summary
# ==========================================

print("\nFinal Project Summary:")

print("Project: Iris Flower Classification using Machine Learning")
print("Dataset: Iris.csv")
print("Algorithm: Logistic Regression")
print("Number of Records Used:", len(df))
print("Number of Features:", X.shape[1])
print("Number of Species:", y.nunique())
print("Test Accuracy:", round(accuracy * 100, 2), "%")

print("\nProject completed successfully!")
