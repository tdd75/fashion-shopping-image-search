import os
import tensorflow as tf
from keras.models import Model
from keras.applications.resnet import ResNet50
from keras.losses import cosine_similarity
from PIL import Image
import psycopg2
import pickle
import hashlib

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
        cursor = conn.cursor()
        # get feature vector from database
        cursor.execute(
            "SELECT id, feature_vector FROM products_product WHERE feature_vector IS NOT NULL")
        # define model
        resnet_model = ResNet50(weights='imagenet')
        self.model = Model(inputs=resnet_model.input,
            outputs=resnet_model.get_layer('avg_pool').output)
        self.cached_query = {}
        self.dnn_features = {}
        for (pid, feature_vector) in cursor.fetchall():
            self.dnn_features[pid] = pickle.loads(bytes.fromhex(feature_vector))

    def calculate_vector(self, image_path, is_dumped=False):
        image = Image.open(image_path)
        image = self._transform_image(image)
        vector = self.model.predict(image)[0]
        if is_dumped:
            return pickle.dumps(vector).hex()
        return vector

    def calculate_distances(self, query):
        # # try load cache
        # hashed_query = self._hash_vector(query)
        # cached = self.cached_query.get(hashed_query)
        # if cached:
        #     return cached
        # calculate
        distances = [(pid, cosine_similarity(query, feature)) for pid, feature in self.dnn_features.items()]
        distances.sort(key=lambda x: x[1])
        id_list = [prod[0] for prod in distances[:20]]
        # # caching
        # self.cached_query[hashed_query] = id_list
        return id_list
    
    def _transform_image(self, image):
        image = image.convert('RGB')
        image = image.resize((224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.resnet.preprocess_input(image)
        image = tf.expand_dims(image, axis=0)
        return image
    
    def _hash_vector(self, vector):
        hash_object = hashlib.sha256(pickle.dumps(vector))
        hash_value = hash_object.hexdigest()
        return hash_value

feature_vector = FeatureVector()