import sys, pickle, pika, time, json

from gixante.utils.rabbit import hat, cfg
from gixante.utils.common import urlFeat, dodgyFeatures, knownErrors, log, stripURL
from gixante.utils.arango import addErrors, missing
from numpy import log as ln

# runtime args
if len(sys.argv) < 2: sys.argv.append("news")

collectionName = sys.argv[1]
maxLinksInQ = cfg['bufferBlock']
addDodgiesToErrors = False

# CONFIG
dodgyDir = '/home/bean/dodgyModel/'

# LOAD MODEL AND DATA
dodgyM, relevTokens = pickle.load(open(dodgyDir + '/model.pkl', 'rb'))

###
while True:
    
    buffer = hat.consumeN(collectionName+'-links', cfg['bufferBlock'])
    
    if hat.nInQ(collectionName) > cfg['bufferBlock']: # enough links, filter these ones
        docs = []
        for m, d, b in buffer:
            doc = json.loads(b.decode('utf-8'))
            doc.update(urlFeat(doc['URL']))
            docs.append(doc)
        
        nURLsInQ = hat.nInQ(collectionName)
        # the model optimal threshold is 0.5 - relax it if the system is not busy
        dodgyThreshold = max([0.5, 1 / (1 + ln(1 + buffer[-1][0].message_count / cfg['bufferBlock']))])
        log.debug("Dodgy threshold set to {0:.3f}".format(dodgyThreshold))
        
        # run the model to detect dodgy docs
        if dodgyThreshold <= 1:
            # Only keep URLs if p(dodgy) < dodgyThreshold
            pDodgy = dodgyM.predict_proba(dodgyFeatures(docs, relevTokens))[:,1]
            docs = [ doc for doc, p in zip(docs, pDodgy) if p <= dodgyThreshold or 'reuters' in doc['URL'].lower() ] # PATCH: model not configured for reuters yet
            
        if addDodgiesToErrors: # slower but useful for model performance debugging
            dodgyDocs = [ {**doc, **{'errorCode': 'dodgy'}} for doc, p in zip(docs, pDodgy) if p >= dodgyThreshold ]
            addErrors(dodgyDocs, collectionName)
                    
        if nURLsInQ > maxLinksInQ:
            # collection q is juicy; re-deliver to -links and wait a bit (not to saturate the CPU)
            publishSuffix = '-links'
            waitSec = 60 * len(docs) / cfg['bufferBlock']
            #log.info("Queue '{0}' is juicy; will wait {1:.1f} seconds...".format(collectionName, waitSec))
            time.sleep(waitSec)
        else:
            publishSuffix = ''
        
        if nURLsInQ > 0:
            # check if URLs are in the collection
            nonErrorSkinnies = missing('skinnyURL', '^%sErrors$' % collectionName, [ doc['skinnyURL'] for doc in docs ])
            notExistURLs = missing('URL', '^{0}$|^{0}Newbies$'.format(collectionName), [ doc['URL'] for doc in docs ])
            validDocs = [ doc for doc in docs if doc['URL'] in notExistURLs and doc['skinnyURL'] in nonErrorSkinnies ]
        else:
            validDocs = [ doc for doc in docs ]
        
        # publish the other ones to publishQ
        log.info("Publishing {0} docs to '{1}{2}'...".format(len(validDocs), collectionName, publishSuffix))
        hat.multiPublish(collectionName+publishSuffix, [ json.dumps(doc) for doc in validDocs ])
        
        # send acks
        hat.multiAck([ m.delivery_tag for m, d, b in buffer ])
    
    else:
        log.info("Queue '{0}' is not juicy enough; will shovel {1:,} links directly in there".format(collectionName, len(buffer)))
        hat.multiPublish(collectionName, [ b for m, d, b in buffer ])
        hat.multiAck([ m.delivery_tag for m, d, b in buffer ])
    
    if len(buffer) < cfg['bufferBlock']:
        log.info("Queue '{0}-links' is not juicy enough; will wait a minute".format(collectionName))
        hat.close()
        time.sleep(60)
