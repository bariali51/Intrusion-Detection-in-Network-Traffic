import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

# Load the NSL-KDD dataset
url = r"C:\Users\bariali51\PycharmProjects\ML\data.txt"

column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "attack_type"
]

# Load dataset into a DataFrame
data = pd.read_csv(url, names=column_names)

# Display the first few rows of the dataset and the column names
print(data.head())
print("Columns in the dataset:", data.columns.tolist())

# Check if expected columns are in the dataset
expected_columns = ['protocol_type', 'service', 'flag', 'attack_type']
missing_columns = [col for col in expected_columns if col not in data.columns]
if missing_columns:
    print(f"Missing columns in the dataset: {missing_columns}")
else:
    # Proceed with encoding if all columns are present
    label_encoders = {}
    for column in expected_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))  # Convert to string before encoding
        label_encoders[column] = le

# Check for object data types in the features
for col in data.columns:
    if data[col].dtype == 'object':
        print(f"Column '{col}' still has non-numeric values.")

# Split features and target variable
X = data.drop("attack_type", axis=1)
y = data["attack_type"].values  # Ensure y is a NumPy array for compatibility

# Ensure all features are numeric
if not np.issubdtype(X.values.dtype, np.number):
    print("Non-numeric values detected in X. Converting all features to numeric types.")
    X = X.apply(pd.to_numeric, errors='coerce')  # Convert all features to numeric, forcing errors to NaN

# Check for missing values
if X.isnull().values.any():
    print("Missing values detected. Filling missing values with column mean.")
    X.fillna(X.mean(), inplace=True)

# Ensure X and y have the same number of rows
print(f"Shape of X: {X.shape}, Length of y: {len(y)}")

# Standardize features
scaler = StandardScaler()

try:
    # Scale the features
    X_scaled = scaler.fit_transform(X)
    print(f"Shape of X_scaled: {X_scaled.shape}, Length of y: {len(y)}")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print("Data split successful!")

    # Ensure that y_train is defined before using it
    if 'y_train' in locals() and 'y_test' in locals():  # Checking if y_train and y_test are defined
        # Build the MLP model
        num_classes = len(np.unique(y_train))

        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Define input shape
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(num_classes, activation='softmax')  # Adjust for number of classes in y_train
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Print classification report
        print(classification_report(y_test, y_pred_classes))

        # Confusion matrix
        confusion_mtx = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(10, 7))
        plt.matshow(confusion_mtx, cmap='Blues', fignum=1)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        # Plotting training & validation loss and accuracy
        plt.figure(figsize=(12, 4))

        # Plotting Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plotting Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    else:
        print("y_train or y_test is not defined. Please check the data processing steps.")

except Exception as e:
    print(f"An error occurred during train_test_split or scaling: {e}")
