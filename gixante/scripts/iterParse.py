import sys, time
from tqdm import tqdm

from gixante.utils.arango import addErrors, addDocs, missingFromAll, getRandomLinks
from gixante.utils.http import parseHTML
from gixante.utils.rabbit import rabbitConsumeChannel, publishLinks, bufferBlock
from gixante.utils.parsing import log, stripURL

# runtime args
if len(sys.argv) < 2: sys.argv.append("news")
if len(sys.argv) < 3: sys.argv.append(bufferBlock / 50)

collectionName = sys.argv[1]
bufferLength = int(sys.argv[2])
forceReParse = False

# configurations for different collections (TODO: put in a config file)
f1 = ['source', 'ampLink', 'canonicalLink', 'title', 'createdTs', 'links', 'metas']
f2 = ['sentences', 'contentLength', 'errorCode']

collArgs = {
    'news': {'fields': f1 + ['body'] + f2, 'useForSentences': 'body'},
    'recipes': {'fields': f1 + ['ingredients', 'method'] + f2, 'useForSentences': 'method'},
    'otherBBCfood': {'fields': f1 + ['ingredientDetails'] + f2, 'useForSentences': 'ingredientDetails'},
}

def parser(URL, skinny):
    doc = parseHTML(URL, **collArgs[collectionName])
    doc.update({'skinnyURL': skinny})
    return(doc)

URLs = []
acks = []
### scrape!


while True:
    method, properties, body = rabbitConsumeChannel.basic_get(collectionName)
    
    if method:
        URLs.append(body.decode('utf-8'))
        acks.append(method.delivery_tag)
    
        if len(URLs) >= bufferLength:
            
            # check how many links are in the '-links' queue
            nLinksInQ = rabbitConsumeChannel.queue_declare(queue=collectionName+'-links', durable=True, passive=True).method.message_count
            
            # parse new URLs
            log.info("Downloading and parsing URLs...")
            if forceReParse or method.message_count < bufferBlock:
                # there are not many URLs left; re-parse all the URLs
                docs = [ parser(url, stripURL(url)) for url in tqdm(URLs) ]
            else:
                # there are quite a bit of links left; only parse missing
                docs = [ parser(url, skinny) for url, skinny in tqdm(missingFromAll(URLs, collectionName)) ]
                        
            # add errors to <collectionName>Errors
            errURLs = set([ err['URL'] for err in addErrors(docs, collectionName) ])
            
            # add valid docs to <collectionName>Newbies
            res = addDocs([ doc for doc in docs if doc['URL'] not in errURLs ], collectionName + 'Newbies')
            
            # publish links
            links = set([ l for doc in docs if 'links' in doc for l in doc['links'] ])
            
            if nLinksInQ > bufferBlock * 10:
                # the shovel is the bottleneck; filter links to make its life easier
                links = [ link for link, skinny in missingFromAll(links, collectionName) ]
            publishLinks(links)
            
            # send acks
            [ rabbitConsumeChannel.basic_ack(delivery_tag = t) for t in acks ]
            
            # reset buffers
            URLs = []
            acks = []
    
    else:
        log.info("Queue %s is empty - will wait a minute" % collectionName)
        publishLinks(getRandomLinks(collectionName))
        time.sleep(60)
