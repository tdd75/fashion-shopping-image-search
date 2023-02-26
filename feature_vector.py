import os
import tensorflow as tf
from keras.models import Model
from keras.applications.resnet import ResNet50
from keras.losses import cosine_similarity
from PIL import Image
import psycopg2
import pickle

import logging
_logger = logging.getLogger(__name__)


class FeatureVector:
    def __init__(self) -> None:
        # connect database
        conn = psycopg2.connect(
            dbname=str(os.getenv('DB_NAME')),
            user=str(os.getenv('DB_USER')),
            password=str(os.getenv('DB_PASSWORD')),
            port=os.getenv('DB_PORT'),
            host=str(os.getenv('DB_HOST')),
        )
        conn = psycopg2.connect(
            'dbname=fashion_shopping user=postgres password=duytd123 port=5432 host=db')
        cursor = conn.cursor()
        # get feature vector from database
        cursor.execute(
            "SELECT id, feature_vector FROM products_product WHERE feature_vector IS NOT NULL")
        # define model
        resnet_model = ResNet50(weights='imagenet')
        self.model = Model(inputs=resnet_model.input,
            outputs=resnet_model.get_layer('avg_pool').output)
        self.dnn_features = {}
        for (id, feature_vector) in cursor.fetchall():
            self.dnn_features[id] = pickle.loads(bytes.fromhex(feature_vector))

    def calculate_vector(self, image_path):
        image = Image.open(image_path)
        image = self._transform_image(image)
        result = self.model.predict(image)[0]
        return result

    def calculate_distances(self, query):
        distances = []
        for id, feature in self.dnn_features.items():
            distances.append((id, cosine_similarity(query, feature)))
        distances.sort(key=lambda x: x[1])
        id_list = [prod[0] for prod in distances[:20]]
        return id_list
    
    def _transform_image(self, image):
        image = image.resize((224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.resnet.preprocess_input(image)
        image = tf.expand_dims(image, axis=0)
        return image

feature_vector = FeatureVector()