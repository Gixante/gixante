"""
This module contains functions used for the content API
Keep it light! (don't load big files)
"""

import sys, requests, json, os, time, re
from gixante.utils.parsing import log, cfg

###
class HeartbeatError(Exception):
    def __init__(self, url, lag):
        self.url = url
        self.lag = lag

class HeartbeatDead(HeartbeatError):
    def __init__(self, url): HeartbeatError.__init__(self, url, None)
    
    def __str__(self):
        return("No heartbeat detected from '{0}'".format(self.url))
        
class HeartbeatUndetected(Exception):
    def __init__(self, url): HeartbeatError.__init__(self, url, None)
    
    def __str__(self):
        return("'{0}' looks alive but I cannot detect any heartbeat from it; perhaps no heartbeat was implemented in the API?".format(self.url))

class HeartbeatSlow(HeartbeatError):
    def __str__(self):
        return("No heartbeat detected from '{0}' for {1:,} seconds".format(self.url, self.lag))

def checkHeartbeat(url, guessHeartBeatURL=True):
    
    if guessHeartBeatURL:
        hbURL = ''.join(re.split('(/)', url)[:5] + ['/heartbeat'])
    else:
        hbURL = url
    
    try:
        hbReq = requests.get(hbURL)
    except:
        log.debug(sys.exc_info().__str__())
        raise HeartbeatDead(url)
    
    hbCode = hbReq.status_code
    if hbCode == 200:
        hb = hbReq.json()
        lag = (time.time() - hb['lastBeat'])
        if  lag > hb['period'] + hb['runtime']:
            raise HeartbeatSlow(url, int(lag))
        else:
            log.debug("'{0}' was alive and kickin' {1} seconds ago".format(url, int(lag)))
    else:
        raise HeartbeatUndetected(url)

def _requestWrap(reqType, url, data=None, **kwargs):
    """
    A wrapper to avoid crashes if the API is misbehaving
    NOTE: a heartbeat GET is expected as a dictionary containing 'lastBeat' (epoch timestamp), 'period' (seconds), 'runtime' (seconds)
    """
    try:
        checkHeartbeat(url)
        
        if reqType == 'GET':
            req = requests.get(url, data=data, **kwargs)
        elif reqType == 'PUT':
            req = requests.put(url, data=data, **kwargs)
        else:
            raise NotImplementedError("reqType must be one of 'GET' or 'PUT'")
            
        msg = "API status on {0} is {1} with message: {2}".format(url, req.status_code, req.reason)
        if req.status_code == 200:
            log.debug(msg)
            out = req.json()
            out.update({'APIcode': req.status_code})
        else:
            log.error(msg)
            out = {'APIerror': req.reason, 'APIcode': req.status_code}
    except HeartbeatError as hbe:
        log.error(hbe.__str__())
        out = {'APIerror': "The API does not seem to be up (no heartbeat)", 'APIcode': 503}
    except requests.exceptions.ConnectionError as e:
        log.error("Cannot connect to API on {0}".format(url))
        out = {'APIerror': "The API does not seem to be up (no connection)", 'APIcode': 503}
    except:
        code = req.status_code if 'req' in locals() else 0
        out = {'APIerror': "Internal error: {0}".format(sys.exc_info().__str__()), 'APIcode': code}
    
    return(out)

def get(url, data=None, **kwargs):
    return(_requestWrap('GET', url, data=data, **kwargs))

def put(url, data=None, **kwargs):
    if data is not None: data.update({'createdTs': time.time()})
    return(_requestWrap('PUT', url, data=data, **kwargs))
