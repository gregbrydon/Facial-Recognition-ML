import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from keras.preprocessing import image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


tqdm.pandas()

target_size = (224, 224)
def getImagePixels(file):
    img = image.load_img(file, grayscale=True, target_size=target_size)
    x = image.img_to_array(img).reshape(1, -1)[0]
    return x

df = pd.read_csv('labels.csv')
df = df[:10000]
df['file'] = 'fairface/' + df['file']
df['pixels'] = df['file'].progress_apply(getImagePixels)

print('convert to numpy')
train_features = []
for i in range(0, df.shape[0]):
    train_features.append(df['pixels'].values[i])

train_features = np.array(train_features)
print('reshape numpy')
train_features = train_features.reshape(train_features.shape[0], 224 * 224)
print('normalize numpy')
train_features = train_features / 255.0

train_label = df[['race']]
races = df['race'].unique()

for j in range(len(races)): #label encoding
    current_race = races[j]
    print("replacing ",current_race," to ", j+1)
    train_label['race'] = train_label['race'].replace(current_race, str(j+1))
train_label = train_label.astype({'race': 'int32'})

# train_target = pd.get_dummies(train_label['race'], prefix='race')
train_target = np.array(train_label['race'])
train_target = train_target.reshape(train_target.shape[0])


print("splitting test/train data")
X_train, X_test, y_train, y_test = train_test_split(train_features, train_target, test_size=0.20, random_state=27)


print('x data save')
np.save('x-train-data.npy', X_train)
np.save('x-test-data.npy', X_test)
print('y data save')
np.save('y-train-data.npy', y_train)
np.save('y-test-data.npy', y_test)
