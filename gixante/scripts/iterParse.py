import sys, time, urllib3, certifi, json, subprocess, re
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
hostname = subprocess.Popen(["cat", "/etc/hostname"], stdout=subprocess.PIPE).communicate()[0].decode().strip()

def checkTemperature():
    
    # to avoid Raspberry Pi burnout
    if hostname == 'pibuntu':
        tmp = subprocess.Popen(["vcgencmd", "measure_temp"], stdout=subprocess.PIPE).communicate()
        temp = float(re.sub("[^0-9\.]*", "", tmp[0].decode()))
        while temp > 65:
            log.warning("CPU temperature is {0:.1f} C: will take five!".format(temp))
            time.sleep(5)
            tmp = subprocess.Popen(["vcgencmd", "measure_temp"], stdout=subprocess.PIPE).communicate()
            temp = float(re.sub("[^0-9\.]*", "", tmp[0].decode()))

### scrape!

while True:
    
    checkTemperature()
    buffer = hat.consumeN(collectionName, bufferLength)
        
    if len(buffer) == bufferLength:
        
        nTot = len(buffer)
        # parse all docs
        docs = [ parser.strip(parser.parseDoc(json.loads(b.decode()))) for m, d, b in tqdm(buffer) ]
        docs = [ doc for doc in docs if 'URL' in doc ]
        
        nValid = len(docs)
        if nValid < nTot: log.warning("WARNING: Found {0} / {1} docs without a valid URL!".format(nTot-nValid, nTot))
                
        log.info("Publishing links...")
        linkCounts = []
        for doc in docs:
            checkTemperature()
            newLinks = [ x[0] for x in missingFromAll(doc.get('links', []), collectionName) ]
            checkTemperature()
            linkCounts.append(hat.publishLinks(newLinks, doc['URL']))
        
        if linkCounts: log.info("Published {0} / {1} new links".format(*[ sum(x) for x in zip(*linkCounts) ]))
        
        errURLs = set([ err['URL'] for err in addErrors(docs, collectionName) ])
        res = addDocs([ doc for doc in docs if doc['URL'] not in errURLs ], collectionName + 'Newbies')
        
        hat.multiAck([ m.delivery_tag for m, d, b in buffer ])
    
    else:
        log.info("Queue %s is empty - will wait a minute" % collectionName)
        [ hat.publishLinks(x['links'], x['ref']) for x in getRandomLinks(collectionName) ]
        hat.sleep(60)
