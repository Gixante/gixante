# this file contains all functions that need a RabbitMQ connection
# keep it light! (don't load big files)

import pika, sys, os, re, json

from gixante.utils.parsing import log, domain2coll, domain

# load config file
cfg = json.load(open(os.path.join(*(['/'] + __file__.split('/')[:-1] + ['config.json']))))
rabbitIP = cfg['rabbitIP']
urlXchgName = cfg['urlXchgName']
bufferBlock = cfg['bufferBlock']

# STARTUP
# connect to RabbitMQ
log.info("Connecting to RabbitMQ...")
rmq_credentials = pika.credentials.PlainCredentials(cfg['rabbitUser'], cfg['rabbitPwd'])
rmq_connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitIP, port=5672, credentials=rmq_credentials))
rabbitConsumeChannel = rmq_connection.channel()
rabbitPublishChannel = rmq_connection.channel()
rabbitConsumeChannel.exchange_declare(exchange=urlXchgName, exchange_type='direct', durable=True)

# declare all queues
for coll in [ coll for coll in set(domain2coll.values()) if coll ]:
    rabbitConsumeChannel.queue_declare(queue=coll, durable=True)
    rabbitConsumeChannel.queue_bind(coll, urlXchgName, routing_key=coll)
    rabbitConsumeChannel.queue_declare(queue=coll+'-links', durable=True)
    rabbitConsumeChannel.queue_bind(coll+'-links', urlXchgName, routing_key=coll+'-links')

# FUNCTIONS
def publishLinks(links, routKeySuffix='-links', linkMaxLegth=250):
    if not links: return None
    
    # remove quotes; ignore long links
    links = set([ re.sub("'", "%27", l.rstrip('\\')) for l in links ])
    links = list(links - set([ l for l in links if len(l) > linkMaxLegth ]))
    
    # which collection should they go to?
    colls = [ domain2coll[domain(l)] for l in links ]
    
    # publish to <collName>-links queue (directly)
    validLinksColls = [ (c, l) for c, l in zip(colls, links) if c ]
    log.info("Publishing {0} links...".format(len(validLinksColls)))
    [ rabbitPublishChannel.basic_publish(exchange=urlXchgName, routing_key=c + appendToQ, body=l, properties=pika.BasicProperties(delivery_mode=2)) for c, l in validLinksColls ]
            
