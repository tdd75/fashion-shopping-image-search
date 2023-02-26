from fastapi import FastAPI
import pickle
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

import helpers
from feature_vector import FeatureVector

app = FastAPI()


class FileSchema(BaseModel):
    file: str


feature_vector = FeatureVector()


@app.post('/api/v1/query-image/')
def query_image(file: FileSchema):
    file_path = helpers.save_to_tmp_image(file.file)

    vector = feature_vector.calculate_vector(file_path)
    result_ids = feature_vector.calculate_distances(vector)

    helpers.remove_tmp_image(file_path)
    return {
        'results': result_ids
    }


@app.post('/api/v1/get-vector/')
def get_vector(file: FileSchema):
    file_path = helpers.save_to_tmp_image(file.file)

    vector = feature_vector.calculate_vector(file_path)

    helpers.remove_tmp_image(file_path)
    return {
        'vector': pickle.dumps(vector).hex()
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('api:app', host='0.0.0.0', port=8001, reload=True)
