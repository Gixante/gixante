"""
This module contains functions used for the content API
Keep it light! (don't load big files)
"""

import sys, requests, json, os, time, re
from gixante.utils.common import cfg

# configure a heartbeat as a global variable
global heartbeat, rootUrl

def configForCollection(collectionName):
    global heartbeat, rootUrl
    rootUrl = 'http://{0}:{1}'.format(cfg[ collectionName + 'ApiIP' ], cfg[ collectionName + 'ApiPort' ])
    heartbeat = Heartbeat(rootUrl, cfg[ collectionName + 'ApiHeartbeatPeriod' ], cfg[ collectionName + 'ApiHeartbeatRuntime' ], True)

class Heartbeat:
    def __init__(self, url, period=None, runtime=0, guessHBUrl=False):
        """
        If period is not defined, will attempt to set it from a retrieved heartbeat
        Set runtime > 0 if the API heartbeat function takes a while to complete (set from a retrieved heartbeat if no period is set)
        status:
            unknown (never tried) = was never detected
            alive                 = everything is within the parameters
            slow                  = hartbeat was detected and is lagging behind the max lag (period + runtime)
            unknown (timed out)   = request was timed out before being able to detect a heartbeat
            dead                  = no connection, no heartbeat or any other API error
        """
        self.hbURL = self.guessHeartBeatURL(url) if guessHBUrl else url
        self.beatTs = 0
        self._status = 'unknown (never tried)'
        self.httpCode = None
        self.updateTs = 0
        
        if not period:
            fetchedHB = self.fetch()
            period = fetchedHB['period']
            runtime = fetchedHB.get('runtime', 0)
        
        self.period = period
        self.maxLag = period + runtime
    
    def __repr__(self):
        printees = []
        for attr, val in self.toDict().items():
            if type(val) is float:
                val = time.strftime("%H:%M", time.localtime(val))
            printees.append("{0}: {1}".format(attr, val))
                
        return('' + ', '.join(printees))
        
    def __str__(self):
        if self._status == 'unknown (never tried)':
            return("As of {2}, {0} is {1}".format(self.hbURL, self._status, time.strftime("%H:%M:%S", time.localtime(time.time()))))
        else:
            self.isSlow()
            updateTimeStr = time.strftime("%H:%M:%S", time.localtime(self.updateTs))
            
            if self.beatTs > 0:
                beatTimeStr = time.strftime("%H:%M:%S", time.localtime(self.beatTs))
                return("As of {3}, {0} is {1} (code {2}): the last heartbeat was at {4} (lag={5:.2f} / {6})".format(self.hbURL, self._status, self.httpCode, updateTimeStr, beatTimeStr, self.lag(), self.maxLag))
            else:
                return("As of {3}, {0} is {1} (code {2}): no heartbeat was ever detected".format(self.hbURL, self._status, self.httpCode, updateTimeStr)) 
    
    def guessHeartBeatURL(self, url):
        return(''.join(re.split('(/)', url)[:5] + ['/heartbeat']))
    
    def toDict(self):
        out = dict()
        for attr in self.__dir__():
            v = self.__getattribute__(attr)
            if not attr.startswith('_') and type(v) in [str, float, int]:
                out[attr] = v
        return(out)
    
    def lag(self):
        return(time.time() - self.beatTs)
    
    def isSlow(self):
        _isSlow = ('unknown' not in self._status and self.lag() > self.maxLag)
        if _isSlow: self._status = 'slow'
        return(_isSlow)
    
    def fetch(self):
        
        self.updateTs = time.time()
        
        for timeout in [ 0.01, 0.05, 0.1 ]:
            try:
                hbRes = requests.get(self.hbURL, timeout=timeout)
                self._status = "alive (timeout={0})".format(timeout)
                self.httpCode = hbRes.status_code
                if self.httpCode == 200:
                    hbDict = hbRes.json() 
                    hbGuess = [ v for k,v in hbDict.items() if 'beat' in k.lower() and type(v) is float ]
                    if hbGuess:
                        self.beatTs = min(hbGuess)
                        self.error = None
                    else:
                        raise RuntimeError("No valid field containing a timestamp found at {0}".format(self.hbURL))
                else:
                    hbDict = {}
                    self.error = "API internal error"
                
                return(hbDict)
            
            except requests.exceptions.Timeout as te:
                self._status = "unknown (timed out)"
                self.httpCode = 408
            except:
                self.error = sys.exc_info().__str__()
                self._status = 'dead'
                self.httpCode = 404
        
        return({})
    
    def status(self, maxRetries=1):
        if self._status.startswith('alive'):
            return('OK')
        elif maxRetries > 0:
            # update it just in case
            tmp = self.fetch()
            return(self.status(maxRetries=(maxRetries-1)))
        else:
            return("Maximim number of retries exceeded; last status available is {0}".format(self._status))
            
    # one for jinja
    def isOK(self):
        return(self.status() == "OK")

def _requestWrap(reqType, url, data=None, **kwargs):
    """
    A wrapper to avoid crashes if the API is misbehaving
    """
    global heartbeat
    
    out = {'APImessage': "Heartbeat {0}".format(heartbeat.status()), 'APIcode': heartbeat.httpCode}
    if heartbeat.isOK():
        try:
            if reqType == 'GET':
                req = requests.get(url, data=data, timeout=25, **kwargs)
            elif reqType == 'PUT':
                req = requests.put(url, data=data, timeout=5, **kwargs)
            else:
                raise NotImplementedError("reqType must be one of 'GET' or 'PUT'")
            
            out = {'APIcode': req.status_code, 'APImessage': req.reason}
            if req.status_code == 200: out.update(req.json())
        
        except:
            if 'req' not in locals():
                # something went wrong before connecting
                out = {'APImessage': "Runtime error: {0}".format(sys.exc_info().__str__()), 'APIcode': None, 'requestFailed': True}
    
    else:
        out.update({'requestFailed': True})
    
    # can't use log - not visible in uwsgi
    print("API status on {0} is {1} with message: {2}".format(url, out['APIcode'], out['APImessage']))
    return(out)

def get(relativeUrl, data=None, **kwargs):
    return(_requestWrap('GET', os.path.join(rootUrl, relativeUrl), data=data, **kwargs))

def put(relativeUrl, data=None, **kwargs):
    if data is not None: data.update({'createdTs': time.time()})
    return(_requestWrap('PUT', os.path.join(rootUrl, relativeUrl), data=data, **kwargs))
