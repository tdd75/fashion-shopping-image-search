import os
import uuid


tmp_dir = '/tmp/image_search/'


def save_to_tmp_image(base64_encoded):
    random_file_name = str(uuid.uuid4())
    file_path = os.path.join(tmp_dir, random_file_name)
    with open(file_path, 'w') as f:
        f.write(base64_encoded)

    return file_path


def remove_tmp_image(file_name):
    file_path = os.path.join(tmp_dir, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
