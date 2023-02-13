import os
import uuid
import base64

tmp_dir = '/tmp/image_search/'
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)


def save_to_tmp_image(encoded_image):
    decoded_image = base64.b64decode(encoded_image)
    random_file_name = str(uuid.uuid4())
    file_path = os.path.join(tmp_dir, random_file_name)
    with open(file_path, 'wb') as f:
        f.write(decoded_image)
    return file_path


def remove_tmp_image(file_name):
    file_path = os.path.join(tmp_dir, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
