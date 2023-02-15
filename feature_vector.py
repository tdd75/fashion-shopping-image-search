import warnings
from vendor.utils_cv.similarity import model, metrics
from fastai.vision import load_learner
import psycopg2
import pickle
import os

warnings.filterwarnings("ignore")


class FeatureVector:
    def __init__(self) -> None:
        self.learner = load_learner('./weight')
        self.dnn_features = {}
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
        for (id, feature_vector) in cursor.fetchall():
            self.dnn_features[id] = pickle.loads(bytes.fromhex(feature_vector))

    def calculate_feature_vector(self, path):
        return model.compute_feature(path, self.learner, self.learner.model[1][-2])

    def calculate_distances(self, query_feature):
        return metrics.compute_distances(query_feature, self.dnn_features)


feature_vector = FeatureVector()
