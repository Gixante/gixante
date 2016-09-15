import sys, time, urllib3, certifi, json
from tqdm import tqdm

from gixante.utils.common import log, checkTemperature, cfg
from gixante.utils.rabbit import hat
from gixante.utils.arango import addErrors, addDocs, getRandomLinks, missingFromAll
import gixante.utils.parsing as parsing

# runtime args
if len(sys.argv) < 2: sys.argv.append("news")
if len(sys.argv) < 3: sys.argv.append(int(cfg['bufferBlock']) / 100)
if len(sys.argv) < 4: sys.argv.append(0)

collectionName = sys.argv[1]
bufferLength = int(sys.argv[2])
nRandomDocs = int(sys.argv[3])

log.info("Starting a HTTP connection...")
urlPoolMan = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
parsing.configForCollection(collectionName)
parser = parsing.Parser(urlPoolMan)

### scrape!

while True:
    
    checkTemperature()
    buffer = hat.consumeN(collectionName, bufferLength)
        
    if len(buffer) > 0:
        
        log.info("Downloading and parsing docs...")
        docs = [ parser.strip(parser.parseDoc(json.loads(b.decode()))) for m, d, b in tqdm(buffer) ]
        docs = [ doc for doc in docs if parser.isValid(doc) ]
        
        if len(docs) < len(buffer): log.warning("WARNING: Found {0} / {1} docs without a valid URL!".format(len(buffer)-len(docs), len(buffer)))
                
        log.info("Publishing links...")
        linkCounts = []
        for doc in tqdm(docs):
            checkTemperature()
            newLinks = [ x[0] for x in missingFromAll(doc.get('links', []), collectionName) ]
            print([ l for l in newLinks if 'reuters' in l ])
            linkCounts.append(hat.publishLinks(newLinks, doc['URL']))
        
        if linkCounts: log.info("Published {0} / {1} new links".format(*[ sum(x) for x in zip(*linkCounts) ]))
        
        res = addDocs([ doc for doc in docs if doc['errorCode'] == 'allGood' ], collectionName + 'Newbies')
        err = addErrors(docs, collectionName)
        
        hat.multiAck([ m.delivery_tag for m, d, b in buffer ])
    
    else:
        log.info("Queue %s is empty - will wait a minute" % collectionName)
        [ hat.publishLinks(x['links'], x['ref'], routingKeySuffix='') for x in getRandomLinks(collectionName, nRandomDocs) ]
        time.sleep(60)
