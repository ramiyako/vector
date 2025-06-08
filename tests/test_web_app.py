import os
import io
import sys
import werkzeug

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import web_app

if not hasattr(werkzeug, "__version__"):
    werkzeug.__version__ = "0"


def test_secure_filename(tmp_path):
    client = web_app.app.test_client()
    web_app.app.config['UPLOAD_FOLDER'] = str(tmp_path)
    web_app.UPLOAD_FOLDER = str(tmp_path)
    data = {
        'file': (io.BytesIO(b'data'), '../../evil.mat')
    }
    client.post('/', data=data, content_type='multipart/form-data')
    assert not (tmp_path / '../../evil.mat').exists()
    assert (tmp_path / 'evil.mat').exists()


def test_invalid_inputs(tmp_path):
    client = web_app.app.test_client()
    web_app.app.config['UPLOAD_FOLDER'] = str(tmp_path)
    response = client.post('/generate', data={'vector_length': 'abc', 'packet_count': 'xyz'})
    assert response.status_code == 200
    assert b'Create Vector' in response.data

