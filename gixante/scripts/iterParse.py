import sys, time, urllib3, certifi, json, re
from tqdm import tqdm
from collections import defaultdict

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
urlPoolMan = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where(), headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'})
parsing.configForCollection(collectionName)
parser = parsing.Parser(urlPoolMan)
dom = parsing.Domain()

### scrape!

while True:
    
    checkTemperature()
    buffer = hat.consumeN(collectionName, bufferLength)
        
    if len(buffer) > 0:
        
        log.info("Downloading and parsing docs...")
        docs = [ parser.strip(parser.parseDoc(json.loads(b.decode()))) for m, d, b in tqdm(buffer) ]
        
        errors = addErrors(docs, collectionName)
        res = addDocs([ doc for doc in docs if doc['errorCode'] == 'allGood' ], collectionName + 'Newbies')
                
        log.info("Publishing links...")
        allLinks = [ re.sub("'", "%27", l.rstrip('\\')) for doc in docs for l in doc.get('links', []) if len(l) <= 250 ]
        newSkinnies = dict(missingFromAll(allLinks, 'news'))
        
        linksByCollection = defaultdict(list)
        for doc in docs:
            for l in doc.get('links', []):
                if l in newSkinnies:
                    linkDoc = dom.add({'URL': l, 'refURL': doc['URL'], 'skinnyURL': newSkinnies[l]})[0]
                    linksByCollection[ parsing.domain2coll[linkDoc['domain']] ].append(json.dumps(linkDoc))
                    newSkinnies.pop(l)
        
        res = dict([ (routingKey, sum(hat.multiPublish(routingKey, bodies))) for routingKey, bodies in linksByCollection.items() ])
        log.info("Published: {0} (out of {1})".format(res, len(allLinks)))
                
        hat.multiAck([ m.delivery_tag for m, d, b in buffer ])
    
    else:
        log.info("Queue %s is empty - will wait a minute" % collectionName)
        #[ hat.publishLinks(x['links'], x['ref'], routingKeySuffix='') for x in getRandomLinks(collectionName, nRandomDocs) ]
        time.sleep(60)
