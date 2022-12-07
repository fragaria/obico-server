#!/usr/bin/env python

import flask
from flask import request, jsonify
from os import path, environ
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
import cv2
import numpy as np
import requests

from lib.detection_model import load_net, detect

THRESH = 0.08  # The threshold for a box to be considered a positive detection
SESSION_TTL_SECONDS = 60*2

# Sentry
if environ.get('SENTRY_DSN'):
    sentry_sdk.init(
        dsn=environ.get('SENTRY_DSN'),
        integrations=[FlaskIntegration(), ],
    )

app = flask.Flask(__name__)

# SECURITY WARNING: don't run with debug turned on in production!
app.config['DEBUG'] = environ.get('DEBUG') == 'True'

model_dir = path.join(path.dirname(path.realpath(__file__)), 'model')
net_main, meta_main = load_net(path.join(model_dir, 'model.cfg'), path.join(model_dir, 'model.weights'), path.join(model_dir, 'model.meta'))

app.logger.debug('\n\n================== STARTING ==============================')


@app.route('/p/', methods=['GET'])
def get_p():
    app.logger.debug('\n\n\n\n\n>>>>>>>>>>>>>>>> NEW ANALYZE REQUEST >>>>>>>>>>>>>>>>')

    imgUrl = request.args.get('img', None)

    if imgUrl:
        try:
            resp = requests.get(imgUrl, stream=True, timeout=(0.1, 5))
            status_code = resp.status_code
            content_type = resp.headers.get('content-type', None)

            app.logger.debug('Analyze URL: %s' % (imgUrl,))
            app.logger.debug('Status code: %s' % (status_code,))
            app.logger.debug('Content type: %s' % (content_type,))

            # check HTTP reponse code is 200
            if status_code != 200:
                return '', status_code

            # check response for correct content type (image)
            allowed_contentypes = ['image/jpeg', 'image/jpg',]
            
            if content_type not in allowed_contentypes:
                return 'Bad content type on your URL: %s. Expecting %s' % (content_type, allowed_contentypes), 400

            img_array = np.array(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1)
            detections = detect(net_main, meta_main, img, thresh=THRESH)

            app.logger.debug('Detections: %s' % (detections,))

            return jsonify({'detections': detections})

        except Exception as e:
            app.logger.debug('ERROR: %s' % (e,))
            return 'ERROR: %s' % (str(e),), 500

    else:
        app.logger.warn('ERROR: no image URL provided.')
        return 'No image URL provided in your request.', 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3333, threaded=False)
