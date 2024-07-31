import pandas as pd
from termcolor import cprint
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


csv_file = "Sales.csv"
data = pd.read_csv(csv_file)

columns = data.columns
print(columns[0])

rows_to_drop = []

for index, row in data.iterrows():
    for column in columns:
        if pd.isnull(row[column]):
            cprint(f"Row {index} has None", "red", attrs=["bold"])
            rows_to_drop.append(index)
            break

data = data.drop(rows_to_drop)
print(data)

target = "Original Price"

X = data.drop(target, axis=1)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

model.compile(optimizer="adam", loss="mean_squared_error")

history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

model.save("model.h5")
