import sys, time, urllib3, certifi, json
from tqdm import tqdm

from gixante.utils.rabbit import hat, cfg, log
from gixante.utils.arango import addErrors, addDocs, getRandomLinks, missingFromAll
import gixante.utils.parsing as parsing

# runtime args
if len(sys.argv) < 2: sys.argv.append("news")
if len(sys.argv) < 3: sys.argv.append(int(cfg['bufferBlock']) / 100)

collectionName = sys.argv[1]
bufferLength = int(sys.argv[2])

log.info("Starting a HTTP connection...")
urlPoolMan = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
parsing.configForCollection(collectionName)
parser = parsing.Parser(urlPoolMan)

### scrape!

while True:
    
    buffer = hat.consumeN(collectionName, bufferLength)
        
    if len(buffer) == bufferLength:
        
        nTot = len(buffer)
        # parse all docs
        docs = [ parser.strip(parser.parseDoc(json.loads(b.decode()))) for m, d, b in tqdm(buffer) ]
        docs = [ doc for doc in docs if 'URL' in doc ]
        
        nValid = len(docs)
        if nValid < nTot: log.warning("WARNING: Found {0} / {1} docs without a valid URL!".format(nTot-nValid, nTot))
               
        # add errors to <collectionName>Errors
        errURLs = set([ err['URL'] for err in addErrors(docs, collectionName) ])
            
        # add valid docs to <collectionName>Newbies
        res = addDocs([ doc for doc in docs if doc['URL'] not in errURLs ], collectionName + 'Newbies')
            
        # publish links
        linkCounts = [ hat.publishLinks([ x[0] for x in missingFromAll(doc.get('links', []), collectionName) ], doc['URL']) for doc in docs ]
        if linkCounts: log.info("Published {0} / {1} new links".format(*[ sum(x) for x in zip(*linkCounts) ]))
        
        hat.multiAck([ m.delivery_tag for m, d, b in buffer ])
    
    else:
        log.info("Queue %s is empty - will wait a minute" % collectionName)
        [ hat.publishLinks(x['links'], x['ref']) for x in getRandomLinks(collectionName) ]
        hat.sleep(60)

###
"""
### older version
while True:
    method, properties, body = rabbitConsumeChannel.basic_get(collectionName)
    #method, properties, body = rabbitConsumeChannel.basic_get('test')
    
    if method:
        docs.append(body.decode('utf-8'))
        
        acks.append(method.delivery_tag)
    
        if len(docs) >= bufferLength:
            
            # check how many links are in the '-links' queue
            nLinksInQ = rabbitConsumeChannel.queue_declare(queue=collectionName+'-links', durable=True, passive=True).method.message_count
            
            # parse new URLs
            log.info("Downloading and parsing URLs...")
            if forceReParse or method.message_count < bufferBlock:
                # there are not many URLs left; re-parse all the URLs
                docs = [ parser.parseDoc({'URL': url}) for url in tqdm(URLs) ]
            else:
                # there are quite a bit of links left; only parse missing
                docs = [ parser.parseDoc({'URL': url, 'skinnyURL': skinny}) for url, skinny in tqdm(missingFromAll(URLs, collectionName)) ]
            
            
            # add errors to <collectionName>Errors
            errURLs = set([ err['URL'] for err in addErrors(docs, collectionName) ])
            
            # add valid docs to <collectionName>Newbies
            res = addDocs([ doc for doc in docs if doc['URL'] not in errURLs ], collectionName + 'Newbies')
            
            # publish links
            links = set([ l for doc in docs if 'links' in doc for l in doc['links'] ])
            
            if nLinksInQ > bufferBlock * 10:
                # the shovel is the bottleneck; filter links to make its life easier
                
            
            # send acks
            [ rabbitConsumeChannel.basic_ack(delivery_tag = t) for t in acks ]
            
            # reset buffers
            URLs = []
            acks = []
    
    else:
        log.info("Queue %s is empty - will wait a minute" % collectionName)
        publishLinks(getRandomLinks(collectionName))
        time.sleep(60)
"""