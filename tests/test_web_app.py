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


def test_spectrogram_and_analyze(tmp_path):
    from scipy.io import savemat
    import numpy as np

    client = web_app.app.test_client()
    web_app.app.config['UPLOAD_FOLDER'] = str(tmp_path)
    web_app.UPLOAD_FOLDER = str(tmp_path)

    file_path = tmp_path / 'test.mat'
    savemat(file_path, {'Y': np.ones(2048, dtype=np.complex64), 'xDelta': 1 / 56e6})

    resp = client.get(f'/spectrogram/{file_path.name}')
    assert resp.status_code == 200
    assert (tmp_path / f'{file_path.name}_spec.png').exists()

    resp = client.get(f'/analyze/{file_path.name}')
    assert resp.status_code == 200
    assert (tmp_path / f'{file_path.name}_analysis.png').exists()


def test_extract_packet(tmp_path):
    from scipy.io import savemat, loadmat
    import numpy as np

    client = web_app.app.test_client()
    web_app.app.config['UPLOAD_FOLDER'] = str(tmp_path)
    web_app.UPLOAD_FOLDER = str(tmp_path)

    file_path = tmp_path / 'test.mat'
    savemat(file_path, {'Y': np.arange(100, dtype=np.complex64), 'xDelta': 1/56e6})

    resp = client.post(f'/extract/{file_path.name}', data={'start': '10', 'end': '20'})
    assert resp.status_code == 200
    saved = tmp_path / 'test_extract.mat'
    assert saved.exists()
    mat = loadmat(saved)
    assert mat['Y'].flatten().shape[0] == 10
    assert (tmp_path / f'{file_path.name}_extract.png').exists()


def test_packet_cache(tmp_path):
    from scipy.io import savemat
    import numpy as np

    client = web_app.app.test_client()
    web_app.app.config['UPLOAD_FOLDER'] = str(tmp_path)
    web_app.UPLOAD_FOLDER = str(tmp_path)
    web_app.PACKET_CACHE.clear()

    file_path = tmp_path / 'cache.mat'
    savemat(file_path, {'Y': np.ones(50, dtype=np.complex64), 'xDelta': 1/56e6})

    client.get(f'/spectrogram/{file_path.name}')
    key = os.path.abspath(file_path)
    assert key in web_app.PACKET_CACHE
    obj1 = web_app.PACKET_CACHE[key]['signal']

    client.get(f'/analyze/{file_path.name}')
    obj2 = web_app.PACKET_CACHE[key]['signal']
    assert obj1 is obj2


def test_file_save_error(tmp_path):
    client = web_app.app.test_client()
    web_app.app.config['UPLOAD_FOLDER'] = str(tmp_path)
    # Try to save to a non-existent directory
    web_app.UPLOAD_FOLDER = str(tmp_path / 'nonexistent')
    data = {
        'file': (io.BytesIO(b'data'), 'test.mat')
    }
    response = client.post('/', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert b'Error saving file' in response.data


def test_vector_creation_error(tmp_path):
    client = web_app.app.test_client()
    web_app.app.config['UPLOAD_FOLDER'] = str(tmp_path)
    # Try to create vector with invalid file
    response = client.post('/generate', data={
        'vector_length': '1',
        'packet_count': '1',
        'file': 'nonexistent.mat'
    })
    assert response.status_code == 200
    assert b'Error creating vector' in response.data

