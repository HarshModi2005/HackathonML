import cv2
import tensorflow as tf
import pickle
model_pkl_file = "model.pkl"  

with open(model_pkl_file, 'wb') as file:  
    model = pickle.load(file)

image_path = ''#enter your file path here

# function to predict whether image at the given path is real or fake
def predict_image(image_path):
    # load the image and resize it to 32x32
    img = cv2.imread(image_path)
    img = tf.image.resize(img, (32, 32))
    
    # predict the class
    y_prob = model.predict(np.expand_dims(img, 0))
    return 'REAL' if y_prob[0]>0.5 else 'FAKE'

print(predict_image(image_path))
