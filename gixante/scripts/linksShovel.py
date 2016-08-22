import sys, pickle, pika, time

from gixante.utils.rabbit import rabbitConsumeChannel, rabbitPublishChannel, urlXchgName, bufferBlock
from gixante.utils.parsing import urlFeat, dodgyFeatures, knownErrors, log
from gixante.utils.arango import addErrors, missing
from numpy import exp

# runtime args
if len(sys.argv) < 2: sys.argv.append("news")

collectionName = sys.argv[1]
maxLinksInQ = bufferBlock
addDodgiesToErrors = False

# CONFIG
dodgyDir = '/home/bean/dodgyModel/'

# LOAD MODEL AND DATA
dodgyM, relevTokens = pickle.load(open(dodgyDir + '/model.pkl', 'rb'))
featURLs = []
acks = []
###

while True:
    method, properties, body = rabbitConsumeChannel.basic_get(collectionName + '-links')
    
    if method:
        featURLs.append(urlFeat(body.decode('utf-8')))
        acks.append(method.delivery_tag)
        
        if len(featURLs) >= bufferBlock:
            nURLsInQ = rabbitConsumeChannel.queue_declare(queue=collectionName, durable=True, passive=True).method.message_count
            dodgyThreshold = exp(-method.message_count / (bufferBlock*10))
            log.debug("Dodgy threshold set to {0:.3}".format(dodgyThreshold))
            
            # run the model to detect dodgy featURLs
            if dodgyThreshold <= 1:
                # Only keep URLs below the theshold ( p(dodgy) < threshold )
                pDodgy = dodgyM.predict_proba(dodgyFeatures(featURLs, relevTokens))[:,1]
                featURLs = [ f for f, p in zip(featURLs, pDodgy) if p < dodgyThreshold ]
                
                if addDodgiesToErrors: # slower but useful for model performance debugging
                    dodgyDocs = [ {**f, **{'errorCode': 'dodgy'}} for f, p in zip(featURLs, pDodgy) if p >= dodgyThreshold ]
                    addErrors(dodgyDocs, collectionName)
                        
            if nURLsInQ > maxLinksInQ:
                # collection q is juicy; re-deliver to -links and wait a bit (not to saturate the CPU)
                publishQ = collectionName + '-links'
                waitSec = 60 * len(featURLs) / bufferBlock
                #log.info("Queue '{0}' is juicy; will wait {1:.1f} seconds...".format(collectionName, waitSec))
                #time.sleep(waitSec)
            else:
                publishQ = collectionName
            
            if nURLsInQ > 0:
                # check if URLs are in the collection
                nonErrorSkinnies = missing('skinnyURL', '^%sErrors$' % collectionName, [ f['skinnyURL'] for f in featURLs ])
                notExistURLs = missing('URL', '^{0}$|^{0}Newbies$'.format(collectionName), [ f['URL'] for f in featURLs ])
                validURLs = [ f['URL'] for f in featURLs if f['URL'] in notExistURLs and f['skinnyURL'] in nonErrorSkinnies ]
            else:
                validURLs = [ f['URL'] for f in featURLs ]
            
            # publish the other ones to publishQ
            log.info("Publishing {0} URLs to '{1}'...".format(len(validURLs), publishQ))
            [ rabbitPublishChannel.basic_publish(exchange=urlXchgName, routing_key=publishQ, body=url.encode(), properties=pika.BasicProperties(delivery_mode=2)) for url in validURLs ]
            
            # send acks
            [ rabbitConsumeChannel.basic_ack(delivery_tag = t) for t in acks ]
            
            # reset buffers
            featURLs = []
            acks = []
    
    else:
        # links queue is empty: just pass the URLs on to be processed
        [ rabbitPublishChannel.basic_publish(exchange=urlXchgName, routing_key=collectionName, body=f['URL'].encode(), properties=pika.BasicProperties(delivery_mode=2)) for f in featURLs ]
        [ rabbitConsumeChannel.basic_ack(delivery_tag = t) for t in acks ]
        featURLs = []
        acks = []
        log.info("Queue '{0}-links' is empty; will wait a minute...".format(collectionName))
        time.sleep(60)

###
exit(0)
"""
shovel = LinksShovel('news', bufferLength, maxLinksInQ)
shovel.start()

# CLASSES
class LinksShovel():
    def __init__(self, collectionName, bufferLength, maxLinksInQ):
        self.buffer = list()
        self.acks = list()
        self.collName = collectionName
        self.bufferLength = bufferLength
        self.maxLinksInQ = maxLinksInQ
        
    def flushBuffer(self, publishQ, dodgyThreshold=1):
        
        # run the model to detect dodgy featURLs
        if dodgyThreshold == 1:
            p = dodgyM.predict(dodgyFeatures(self.buffer, relevTokens))
        else:
            p = dodgyM.predict_proba(dodgyFeatures(self.buffer, relevTokens))[:,1]
        
        dodgyDocs = [ {**f, **{'errorCode': 'dodgy'}} for f, p in zip(self.buffer, p) if p >= dodgyThreshold ]
        # add the dodgy featURLs to the Errors collection
        addErrors(dodgyDocs, self.collName)
        
        # check if the non dodgy URLs are in the collection
        featURLs = [ f for f, p in zip(self.buffer, p) if p < dodgyThreshold ]
        nonErrorSkinnies = missing('skinnyURL', '^%sErrors$' % self.collName, [ f['skinnyURL'] for f in featURLs ])
        notExistURLs = missing('URL', '^%s$' % self.collName, [ f['URL'] for f in featURLs ])
        
        # publish the other ones to <self.consume_q>
        validURLs = [ f['URL'] for f in featURLs if f['URL'] in notExistURLs and f['skinnyURL'] in nonErrorSkinnies ]
        log.info("Publishing {0} URLs to '{1}'...".format(len(validURLs), collectionName))
        [ rabbitPublishChannel.basic_publish(exchange=urlXchgName, routing_key=publishQ, body=url.encode(), properties=pika.BasicProperties(delivery_mode=2)) for url in validURLs ]
        self.buffer = list()
        
        # send acks
        [ rabbitConsumeChannel.basic_ack(delivery_tag = t) for t in self.acks ]
        self.acks = list()
        
    def callback(self, ch, method, properties, body):
        
        self.buffer.append(urlFeat(body.decode('utf-8')))
        self.acks.append(method.delivery_tag)
            
        if len(self.buffer) >= self.bufferLength:
            nURLsInQ = rabbitConsumeChannel.queue_declare(queue=self.collName, durable=True, passive=True).method.message_count
            
            if nURLsInQ > self.maxLinksInQ:
                # collection q is juicy; re-deliver to -featURLs and wait a bit (not to saturate CPU)
                #TODO: make threshold and sleep time dynamic (based on q pressure)
                self.flushBuffer(self.collName + '-featURLs')
                waitSec = 5
                log.debug("Queue '{0}' is juicy; will wait {1} seconds...".format(self.collName, waitSec))
                time.sleep(waitSec)
            else:
                self.flushBuffer(self.collName)
    
    def start(self):
        rabbitConsumeChannel.basic_consume(self.callback, queue=self.collName + '-featURLs')
        rabbitConsumeChannel.start_consuming()

dodgyThreshold=1
publishQ = 'news-featURLs'

for k in range(10):
    method, properties, body = rabbitConsumeChannel.basic_get('news-featURLs')
    out = urlFeat(body.decode('utf-8'))
    self.acks.append(method.delivery_tag)
    self.buffer.append(out)
###

import sys
sys.path.append('/home/bean/Code/Python')

from htmlParser import urlFeat, dodgyM, colOrder, dodgyFeatures, knownErrors, handleErrors, rabbitChannel

cleaner = RabbitMQBufferedBridge(consumeCh, 'news', 'URLs', publishCh, urlFeat, bufferFun, 10000)
cleaner.start()

### remove dodgies from Arango

def callback(ch, method, properties, body):
    URL = body.decode()
    database.execute_query("for doc in news filter doc.URL == '%s' filter doc.partition == -101 remove doc in news" % URL)
    ch.basic_ack(method.delivery_tag)
    
rabbitChannel.basic_consume(callback, queue='dodgies')
rabbitChannel.start_consuming()

###

import json, pika, sys
sys.path.append('/home/bean/Code/Python')

from htmlParser import *

def callback(ch, method, properties, body):
    errUrl = json.loads(body.decode())
    if 'not defined' in errUrl['error'].lower():
        ch.basic_publish(exchange='URLs', routing_key='news', body=errUrl['URL'], properties=pika.BasicProperties(delivery_mode=2))
        ch.basic_ack(method.delivery_tag)
    else:
        print(errUrl['error'])

rabbitChannel.basic_consume(callback, queue='errors')
rabbitChannel.start_consuming()
"""



