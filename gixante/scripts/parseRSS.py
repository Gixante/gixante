import sys, json, time
import pandas as pd
from feedparser import parse as rssParse

import gixante.utils.parsing as parsing
from gixante.utils.rabbit import hat, log

# runtime args
if len(sys.argv) < 2: sys.argv.append("news")

collectionName = sys.argv[1]

parsing.configForCollection(collectionName)
parser = parsing.Parser()
parser.requiredKeys = set(['URL', 'skinnyURL', 'title', 'domain', 'parsedTs', 'createdTs'])

usefulKeys = set([ k for f in parser.fields.values() for k in f.requiredKeys(justCurrent=False) if k and not k.startswith('__')])
usefulKeys.update(set(['title', 'author']))
stripRgx = '^link$|^published.*$|^update.*$|.*_detail$|^id$|^tags$'

parserConfig = pd.read_csv(parsing.configFile, sep='\t', header=0)
parserConfig = parserConfig[ parserConfig.collection == collectionName ]

feedUrls = dict([ (row.domain, [u for u in row.rssFeeds.split(',')]) for row in parserConfig.itertuples() ])
feedLastDownloaded = dict([ (dom, 0) for dom in feedUrls.keys() ])
docLastInQTime = dict()

###

while True:
    urlInRSS = []
    t0 = time.time()
    
    for domain in feedUrls.keys():
        for url in feedUrls[domain]:
            print(url)
            p = rssParse(url)
            
            for doc in p.entries:
                # can't do jack with no URL!
                if 'link' not in doc and 'feedburner_origlink' not in doc:
                    log.warning("Malformed RSS feed for {0}".format(url))
                    continue
                
                urlInRSS.append(doc['link'])
                
                # get the last update time (use 'published' if not available)
                refTs = 0
                for timeF in ['updated', 'published']:
                    if timeF + '_parsed' in doc:
                        refTs = time.mktime(doc[timeF + '_parsed'])
                        break
                    elif timeF in doc:
                        hms = re.sub(".*([0-9][0-9]:[0-9][0-9]:[0-9][0-9]).*", r'\1', doc[timeF])
                        timeStr = time.strftime("%Y-%m-%d ", time.localtime(recogniseDate(doc[timeF]).timestamp())) + hms
                        refTs = time.mktime(time.strptime(timeStr, "%Y-%m-%d %H:%M:%S"))
                        break
                
                # only process if there's a new update
                if refTs > docLastInQTime.get(doc['link'], 0):
                    doc = dict([ (k, v) for k, v in doc.items() if k in usefulKeys ])
                    doc = parser.strip(parser.parseDoc(doc), stripRgx, stripPrivate=False)
                    if doc['errorCode'] == 'allGood':
                        #log.info("New version of {0} by {1} seconds".format(doc['URL'], refTs - docLastInQTime.get(doc['URL'], 0)))
                        hat.simplePublish(collectionName, json.dumps(doc))
                        docLastInQTime[doc['URL']] = refTs
                    
    docLastInQTime = dict([ (url, docLastInQTime.get(url, 0)) for url in urlInRSS ])
    
    if time.time()-t0 < 300:
        log.info("Taking a nap...")
        time.sleep(300 + t0 - time.time())

