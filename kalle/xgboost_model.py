import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# Load and encode labels
data_dict = pickle.load(open('kalle/mit_data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Convert class labels (e.g., 'A', 'B', ...) to integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.1, shuffle=True, stratify=labels
)

# Initialize XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the model
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Evaluate accuracy
score = accuracy_score(y_test, y_predict)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the model
with open('kalle/mit_xgboost.p', 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': label_encoder}, f)
