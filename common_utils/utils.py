import json
from common_utils.logging_util import *
from flask import jsonify

logger_app = setup_logging()

# A shorthand function for sending an error message to indicate something wrong with client request, 400
def send_error(msg, code=400):
    logger_app.error('Exception: '+ str(msg))
    res = jsonify(status='error', message=msg)
    res.status_code = code
    return res

def validateJSON(jsonData):
    try:
        json.loads(jsonData)
    except ValueError as err:
        return False
    return True

