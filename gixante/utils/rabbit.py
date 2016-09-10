# this file contains all functions that need a RabbitMQ connection
# keep it light! (don't load big files)

import pika, sys, re, json

from gixante.utils.common import log, cfg
import gixante.utils.parsing as parsing
from tqdm import tqdm

parsing.configForCollection()
parsing.requiredFields = [ 'URL', 'skinnyURL', 'domain' ]
basicParser = parsing.Parser()

class RabbitHat:
    def __init__(self, rmqParams):
        self.connParams = rmqParams
        self.exchangeName = cfg['urlXchgName']
        self._durable = pika.BasicProperties(delivery_mode=2)
        self._reconnect()
        
        # declare the exchange
        self._publishCh.exchange_declare(exchange=self.exchangeName, exchange_type='direct', durable=True)
        # declare all queues
        for coll in set(parsing.domain2coll.values()):
            self._publishCh.queue_declare(queue=coll, durable=True)
            self._publishCh.queue_bind(coll, self.exchangeName, routing_key=coll)
            self._publishCh.queue_declare(queue=coll+'-links', durable=True)
            self._publishCh.queue_bind(coll+'-links', self.exchangeName, routing_key=coll+'-links')
    
    def _reconnect(self):
        self._connection = pika.BlockingConnection(self.connParams)
        self._publishCh = self._connection.channel()
        self._consumeCh = self._connection.channel()
        
    def sleep(self, timeSec):
        self._connection.sleep(timeSec)
    
    def _pullAlive(self, direction):
        assert direction =='publish' or direction=='consume'
        
        channelName = "_{0}Ch".format(direction)
        
        # make sure the _connection is open
        try:
            self._connection.sleep(1e-10)
            #print('connection was open')
            assert self.__getattribute__(channelName).is_open
        
        except (pika.exceptions.ConnectionClosed, AssertionError) as e:
            #print('connection / channel was closed')
            self._reconnect()
        
        return(self.__getattribute__(channelName))
        
        # now look at the channel
        if connWasOpen and self.__getattribute__(channelName).is_open:
            return(self.__getattribute__(channelName))
        else:
            self.__setattr__(channelName, self._connection.channel(channel_number=self.__getattribute__(channelName).channel_number))
            return(self.__getattribute__(channelName))
    
    def nInQ(self, Q):
        return(self._pullAlive('consume').queue_declare(queue=Q, durable=True, passive=True).method.message_count)
        
    def declareQ(self, **kwargs):
        self._pullAlive('publish').queue_declare(**kwrgs)
    
    def bindQ(self, **kwargs):
        self._pullAlive('publish').queue_bind(**kwrgs)
        
    def consumeN(self, Q, N):
        ch = self._pullAlive('consume')
        
        out = []
        
        while len(out) == 0 or (len(out) < N and method):
            method, properties, body = ch.basic_get(Q)
            out.append((method, properties, body))
        
        return(out)
    
    def multiPublish(self, routing_key, bodies):
        ch = self._pullAlive('publish')
        [ ch.basic_publish(exchange=self.exchangeName, routing_key=routing_key, body=body, properties=self._durable) for body in bodies ]
    
    def multiAck(self, deliveryTags):
        ch = self._pullAlive('consume')
        [ ch.basic_ack(delivery_tag = t) for t in deliveryTags ]
    
    def publishLinks(self, links, refURL=None, routingKeySuffix='-links', linkMaxLength=250):
        if not links: return (0, 0)
        
        ch = self._pullAlive('publish')
        
        # remove quotes; ignore long links
        nTot = len(links)
        links = set([ re.sub("'", "%27", l.rstrip('\\')) for l in links ])
        links = list(links - set([ l for l in links if len(l) > linkMaxLength ]))
        
        nPub = 0
        for l in links:
            d = basicParser.strip(basicParser.parseDoc({'URL': l}))
            if d['errorCode'] == 'allGood': # using "if 'domain' in doc" will also publish unkown domains
                if refURL: d['refURL'] = refURL
                nPub += ch.basic_publish(exchange=self.exchangeName, routing_key=parsing.domain2coll[d['domain']]+routingKeySuffix, body=json.dumps(d), properties=self._durable)
        
        return(nPub, nTot)

# STARTUP
# connect to RabbitMQ
log.info("Connecting to RabbitMQ...")
rmqCredentials = pika.credentials.PlainCredentials(cfg['rabbitUser'], cfg['rabbitPwd'])
rmqParams = pika.ConnectionParameters(host=cfg['rabbitIP'], port=5672, credentials=rmqCredentials)
hat = RabbitHat(rmqParams)
