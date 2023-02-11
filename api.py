from feature_vector import FeatureVector
from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import base64

from . import helpers

app = FastAPI()


class FileSchema(BaseModel):
    file: str | None


feature_vector = FeatureVector()


@app.post('/api/v1/query-image/')
def query_image(file: FileSchema):
    base64.decode(file.file, file_path)
    file_path = helpers.save_to_tmp_image(file.file)
    vector = feature_vector.calculate_feature_vector(file_path)
    distances = feature_vector.calculate_distances(vector)
    distances.sort(key=lambda x: x[1])
    id_list = [prod[0] for prod in distances[:20]]
    helpers.remove_tmp_image(file_path)
    return {
        'results': id_list
    }


@app.post('/api/v1/get-vector/')
def get_vector(file: FileSchema):
    file_path = helpers.save_to_tmp_image(file.file)
    vector = feature_vector.calculate_feature_vector(file_path)
    helpers.remove_tmp_image(file_path)
    return {
        'vector': pickle.dumps(vector).hex()
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('api:app', host='0.0.0.0', port=8001, reload=True)
