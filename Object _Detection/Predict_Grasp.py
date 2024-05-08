import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model


class MobileNetPredictor:
    def __init__(self, model_path, class_labels):
        self.model = load_model(model_path)
        self.class_labels = class_labels

    def predict_image(self, image_path):
        image = load_img(image_path, target_size=(224, 224))
        image_array = img_to_array(image) / 255.0
        image_array = tf.expand_dims(image_array, 0)

        predictions = self.model.predict(image_array)
        predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
        predicted_class_label = self.class_labels[predicted_class_index]

        return predicted_class_label


if __name__ == '__main__':
    
    model_path = 'mobilenetv3_grasp_model.h5'
    class_labels = ['Palmar_Wrist_Pronated', 'Palmar_wrist_neutral', 'Pinch', 'Tripod']

    predictor = MobileNetPredictor(model_path, class_labels)


    predictor = MobileNetPredictor(model_path, class_labels)

    image_path = 'object_image.jpg'
    predicted_class = predictor.predict_image(image_path)

    print("Predicted Class:", predicted_class)