"""
This module contains functions used for the content API
Keep it light! (don't load big files)
"""

import sys, requests, json, os
from gixante.utils.parsing import log

# load config file
print(os.path.join(*(['/'] + __file__.split('/')[:-1] + ['config.json'])))
cfg = json.load(open(os.path.join(*(['/'] + __file__.split('/')[:-1] + ['config.json']))))

def _requestWrap(reqType, url, data=None, **kwargs):
    """
    A wrapper to avoid crashes if the API is misbehaving
    """
    try:
        if reqType == 'get':
            req = requests.get(url, data=data, **kwargs)
        elif reqType == 'put':
            req = requests.put(url, data=data, **kwargs)
        else:
            raise NotImplementedError("reqType must be one of 'get' or 'put'")
            
        msg = "API status on {0} is {1} with message: {2}".format(url, req.status_code, req.reason)
        if req.status_code == 200:
            log.debug(msg)
            out = req.json()
            out.update({'APIcode': req.status_code})
        else:
            log.error(msg)
            out = {'APIerror': req.reason, 'APIcode': req.status_code}
    except requests.exceptions.ConnectionError as e:
        log.error("Cannot connect to API on {0}".format(url))
        out = {'APIerror': "The API does not seem to be up", 'APIcode': 503}
    except:
        code = req.status_code if 'req' in locals() else 0
        out = {'APIerror': "Internal error: {0}".format(sys.exc_info().__str__()), 'APIcode': code}
    
    return(out)
    
def APIheartbeat():
    hb = _requestWrap('get', apiRoot + '/heartbeat')
    if hb['APIcode'] == 200:
        return('OK')
    elif hb['APIcode'] == 404:
        return('API is up, but cannot check heartbeat') # perhaps it's not implemented?
    else:
        return(hb['APIerror'])

def get(url, data=None, **kwargs):
    return _requestWrap('get', url, data=data, **kwargs)

def put(url, data=None, **kwargs):
    return(_requestWrap('put', url, data=data, **kwargs))
    