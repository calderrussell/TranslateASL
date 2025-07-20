import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('calder/data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

output_dir = os.path.join('.', 'calder')
os.makedirs(output_dir, exist_ok=True)

# Save model to pickle in the Calder folder
model_path = os.path.join(output_dir, 'model.p')
with open(model_path, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"\n✅ Model saved to {model_path}.")