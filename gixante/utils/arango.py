# this file contains all functions that need a connection to ArangoDB

import sys, re, time, pickle, json, os
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain, product
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from random import sample
from collections import defaultdict, Counter # Counter is only used for debug
from lxml import etree # only used if not using rabbit

from gixante.pyrango import Arango
from gixante.utils.parsing import log, knownErrors, stripURL, emptyField, cfg
from gixante.utils.http import addAll, urlPoolMan # only used if not using rabbit
#from gixante.utils.rabbit import publishLinks one for the future

# STARTUP
log.info("Connecting to ArangoDB...")
database = Arango(host=cfg['arangoIP'], port=cfg['arangoPort'], password=cfg['arangoPwd']).db('catapi')

# FUNCTIONS
# Error handler
def addErrors(errorDocs, collectionName):
    # errorDocs is a list of dictionaries containing (at least): 'URL', 'skinnyURL' (to check for missing), 'errorCode' (a key from knownErrors)
    
    newErrSkinnies = missing('skinnyURL', '^%sErrors' % collectionName, [ e['skinnyURL'] for e in errorDocs ])
    
    errors = []
    for doc in errorDocs:
        if doc['errorCode'] == 'allGood' or doc['skinnyURL'] not in newErrSkinnies: # i.e. no error or already in DB
            continue
            
        if doc['errorCode'] not in knownErrors:
            doc['otherErrorMessage'] = "Invalid error code: " + doc['errorCode']
            doc['errorCode'] = 'otherError'
        
        err = {
            'raisedTs': int(time.time()), 
            'URL': doc['URL'], 
            'skinnyURL': doc['skinnyURL'], 
            'errorCode': doc['errorCode'],
            }
        
        # some codes require extra fields:
        if doc['errorCode'] == 'parsingError' and 'parserLog' in doc:
            err['parserLog'] = doc['parserLog']
            
        if doc['errorCode'] == 'duplicated' and 'dupliURL' in doc:
            err['dupliURL'] = doc['dupliURL']
            
        if doc['errorCode'] in ['empty', 'duplicated'] and 'usedForBody' in doc:
            err['usedForBody'] = doc['usedForBody']
        
        if doc['errorCode'] == 'otherError' and 'otherErrorMessage' in doc:
            err['otherMessage'] = doc['otherErrorMessage']
        
        errors.append(err)
    
    # TODO: use batches!
    if len(errors) > 0:
        q = re.sub('None', 'NULL', "FOR err in {0} INSERT err IN {1}Errors OPTIONS {{ ignoreErrors: true }}".format(errors, collectionName))
        res = database.execute_query(q)
    
    log.info("Added {0} errors to {1}Errors".format(len(errors), collectionName)) 
    return(errors)   

def addDocs(docs, collectionName):
    # TODO: use batches!
    res = []
    if len(docs):
        log.info("Adding {0} documents to {1}...".format(len(docs), collectionName))
        collection = database.col(collectionName)
        for doc in tqdm(docs):
            try:
                res.append(collection.create_document(doc))
            except:
                res.append(collection.update_by_example({'URL': doc['URL']}, doc))
    
    return(res)

def delDocs(docs, collectionName):
    res = []
    # TODO: use batches!
    URLs = exist('URL', '^%s$' % collectionName, [ doc['URL'] for doc in docs])
    if len(URLs):
        q = "FOR doc in {0} FILTER doc.URL in {1} REMOVE doc IN {0} OPTIONS {{ ignoreErrors: true }}".format(collectionName, list(URLs))
        res = database.execute_query(q)
    
    log.info("Deleted {0} documents from {1}".format(len(URLs), collectionName))
    
    return(res)

def getRandomLinks(collectionName, nDocs=25):
    randomQ = "FOR doc in {0} FILTER doc.partition > 0 FILTER doc.links != NULL LIMIT {1} SORT RAND() LIMIT {2} RETURN doc.links"
    linkIter = database.execute_query(randomQ.format(collectionName, nDocs**2, nDocs))
    return([ l for ll in linkIter for l in ll ])

def exist(what, collRgx, stringIter):
    
    # remove quotes and duplicates
    strings = [ re.sub("'", "%27", s.rstrip('\\')) for s in stringIter ]
    
    collToCheck = [ coll for coll in database.collections['user'] if re.match(collRgx, coll) ]
    existQ = "FOR doc in {{0}} filter doc.{1} in {0} RETURN doc.{1}".format(strings, what)
    
    out = set()
    for coll in collToCheck:
        out.update(set(database.execute_query(existQ.format(coll))))
    
    return(out)

def missing(what, collRgx, stringIter):
    
    # remove quotes and duplicates
    strings = set([ re.sub("'", "%27", s.rstrip('\\')) for s in stringIter ])
    
    collToCheck = [ coll for coll in database.collections['user'] if re.match(collRgx, coll) ]
    existQ = "FOR doc in {{0}} filter doc.{1} in {0} RETURN doc.{1}".format(list(strings), what)
    
    for coll in collToCheck:
        strings = strings.difference(set(database.execute_query(existQ.format(coll))))
    
    return(strings)
    
def missingFromAll(URLs, collectionName):
    URLs = set(URLs)
    skinnyURLs = [ stripURL(url) for url in URLs ]
    nonErrorSkinnies = missing('skinnyURL', '^%sErrors$' % collectionName, skinnyURLs)
    notExistURLs = missing('URL', '^{0}$|^{0}Newbies$'.format(collectionName), URLs)
    return(set([ (url, skinny) for url, skinny in zip(URLs, skinnyURLs) if url in notExistURLs and skinny in nonErrorSkinnies ]))

def count(collectionName, fast=True):
    if fast:
        return(database.col(collectionName).statistics['alive']['count'])
    else:
        q = "FOR doc in %s COLLECT WITH COUNT INTO c RETURN c"
        return(next(database.execute_query(q % collectionName)))

def cleanupTests(collectionNames):
    cleanupQ = "FOR doc IN {0} FILTER doc._flag == 'test' REMOVE doc IN {0}"
    res = [ list(database.execute_query(cleanupQ.format(cName))) for cName in collectionNames ]
    return(dict(zip(collectionNames, res)))

def getCollection(collectionName, withPivots=True):
    
    if collectionName not in database.collections['user']:
        database.create_collection(collectionName)
    
    collection = database.col(collectionName)
    # TODO: set indices based on kwargs
    collection.create_hash_index(['URL'], unique=True, sparse=False)
    collection.create_hash_index(['partition'], unique=False, sparse=False)
    collection.create_geo_index(['tsne'])

    if withPivots:
        pivotCollName = collectionName +'Pivots'
        if pivotCollName not in database.collections['user']:
            database.create_collection(pivotCollName)
        
        pivotColl = database.col(pivotCollName)
        if pivotColl.statistics['alive']['count'] == 0:
            splitPartition(0, database, collection, partitionSize=250, fast=False)
        
        pivotColl.create_hash_index(['URL'], unique=True, sparse=False)
        pivotColl.create_hash_index(['partition'], unique=True, sparse=False)
    
    return(collection)

def createPivot(doc, vec, pivotColl, partitionId, nDocs):
    pivot = {'URL': doc['URL']}
    pivot['partition'] = partitionId
    pivot['nDocs'] = nDocs
    pivot['embedding'] = [ float(v) for v in vec ]
    pivot['createdTs'] = int(time.time())
    pivotColl.create_document(pivot)
    log.debug("Created a new pivot: partition={0}, nDocs: {1}".format(partitionId, nDocs))

def getPivots(collectionName, cacheFile='/tmp/pivots.pkl'):
    
    pivCollName = collectionName + 'Pivots'
    
    if os.path.exists(cacheFile):
        try:
            log.info("Getting pivots from %s - the fast way" % pivCollName)
            existPids, existVecs, _ = pickle.load(open('/tmp/pivots.pkl', 'rb'))
            allPivotQ = "FOR piv in %s RETURN {'partition': piv.partition, 'nDocs': piv.nDocs}" % pivCollName
            embedQ = "FOR piv in %s FILTER piv.partition == {0} RETURN piv.embedding" % pivCollName
            pidCounts = list(database.execute_query(allPivotQ))
            
            outList = []
            for pc in tqdm(pidCounts):
                if pc['partition'] in existPids:
                    ix = np.where(existPids == pc['partition'])[0][0]
                    outList.append( (existVecs[ix], pc['partition'], pc['nDocs']) )
                else:
                    newVec = np.array(next(database.execute_query(embedQ.format(pc['partition']))))
                    outList.append( (newVec, pc['partition'], pc['nDocs']) )
            
            tmp = zip(*outList)
            pivotVecs = np.vstack(next(tmp)).astype(np.float32)
            pivotIds = np.array(next(tmp))
            counts = np.array(next(tmp))
            loadFromScratch = False
        except Exception as e:
            log.debug(sys.exc_info().__str__())
            log.warning("Initialisation from cache failed - will reload from database")
            loadFromScratch = True
    
    if loadFromScratch:
        log.info("Getting pivots from %s - from scratch" % pivCollName)
        allPivotQ = "FOR piv in %s RETURN {'embedding': piv.embedding, 'partition': piv.partition, 'nDocs': piv.nDocs}" % pivCollName
        pivotDocs = list(database.execute_query(allPivotQ))
        pivotVecs = np.vstack([ p['embedding'] for p in pivotDocs ]).astype(np.float32)
        pivotIds = np.array([ p['partition'] for p in pivotDocs ])
        counts = [ p['nDocs'] for p in pivotDocs ]
    
    pivotCounts = dict(zip(pivotIds, counts))
    pickle.dump((pivotIds, pivotVecs, pivotCounts), open('/tmp/pivots.pkl', 'wb'))
    return(pivotVecs, pivotIds, pivotCounts)

def checkPivotCount(collectionName, nRand=100):
    log.info("Checking that the count on {0} random pids is correct...".format(nRand))
    pColl = database.col(collectionName + 'Pivots')
    allPivotQ = "FOR piv in %s RETURN {'partition': piv.partition, 'nDocs': piv.nDocs}" % pColl.name
    pidCounts = list(database.execute_query(allPivotQ))
    
    reportList = []
    for pc in tqdm(sample(pidCounts, nRand)):
        actualCount = next(database.execute_query("FOR doc in news filter doc.partition == {0} collect with count into c return c".format(pc['partition'])))
        if actualCount != pc['nDocs']:
            pColl.update_by_example({'partition': pc['partition']}, {'nDocs': actualCount})
            pc.update({'actual': actualCount})
            reportList.append(pc)
    
    report = pd.DataFrame(reportList)
    pctOff = len(report) / nRand
    meanErr = ((report['nDocs'] - report['actual']).abs() / report['actual']).mean()
    log.info("{0:.1f}% were off - mean abs err = {1:.2f}%".format(100*pctOff, 100*meanErr))
    
    return(report, pctOff, meanErr)

def getPivotIds(collectionName):
    return(np.array(list(database.execute_query("FOR p in %sPivots RETURN p.partition" % collectionName))))

def grouper(x):
    nCore = round(len(x)/2)
    avg = x.mean(0)
    rank = x.dot(avg) / (np.abs(x-avg) / x.std(0)).mean(1)
    return(x[ np.argpartition(-rank, nCore)[:nCore] ].mean(0))

def scoreBatch(docBatch, voc, weights, scoreType='zscores', verbose=True):
    """
    Reads a batch of docs (list of deafaultdicts, expecting 'sentences' as a key containing the text)
    Returns a tuple of lists to be expanded into indices to build sparse matrices
    Done by sentence
    """
    ct = tqdm if verbose else list
    
    if verbose: log.info("Parsing {0:,} documents...".format(len(docBatch)))
    docN = 0
    sentN = 0
    ixBySentence = []
    for doc in ct(docBatch):
        for sent in doc['sentences']:
            wordIndices = [ voc[w] for w in sent.split() if w in voc ]
            ixBySentence.append(([np.divide(1, len(wordIndices))], [sentN], wordIndices, [docN]))
            sentN += 1
        
        docN += 1
    
    if len(ixBySentence) == 0: return((docBatch, np.array([]).reshape(0,0)))
    
    # now expand each sentence word by word
    ixByWord = list(zip( *chain( *[ product( *ix ) for ix in ixBySentence ] ) ))
    if len(ixByWord) == 0: return((docBatch, np.array([]).reshape(0,0)))
    
    # create a sparse matrix
    M = csr_matrix( (ixByWord[0], (ixByWord[1], ixByWord[2])), shape=(sentN, len(weights)), dtype=np.float32 )
    
    # score all the sentences
    if verbose: log.info("Scoring {0:,} documents, {1:,} sentences and {2:,} words...".format(docN, sentN, len(ixByWord[0])))
    scores = M.dot(weights)
    norms = np.sqrt((scores**2).sum(axis=1)).reshape(-1,1)
    norms[ norms==0 ] = 1 # just to avoid NaNs later - TODO: is this needed?
    scores = scores / norms
    
    docIndex = np.array([ ix[3][0] for ix in ixBySentence ])
    scoresByDoc = np.zeros( (docN, weights.shape[1]) )
    
    if scoreType == 'zscores':
        # group them by docs (numpy faster than pandas if using z-scores)
        counts = np.bincount(docIndex)
        longDocs = np.where(counts > 1)[0]
        shortDocs = np.where(counts == 1)[0]
        
        if len(longDocs): scoresByDoc[ longDocs, : ] = np.vstack([ grouper(scores[ docIndex == d, : ]) for d in longDocs ])
        if len(shortDocs): scoresByDoc[ shortDocs, : ] = scores[ [ np.where(docIndex == d)[0][0] for d in shortDocs ], : ]
    elif scoreType == 'mean':
        scoresByDoc[ np.unique(docIndex), : ] = pd.DataFrame(scores).groupby(docIndex).mean().values
    else:
        log.error("Value '%s' for 'scoreType' not understood." % scoreType)
 
    norms2 = np.sqrt((scoresByDoc**2).sum(axis=1)).reshape(-1,1)
    return(docBatch, scoresByDoc / norms2)

def iterKMeans(X, partitionSize):
    
    labels = np.tile(0, len(X))
    labelCount = np.bincount(labels)
    centres = np.zeros((1, X.shape[1]))
    
    while labelCount.max() > partitionSize:
        splittee = np.argmax(labelCount)
        subIx = np.where(labels == splittee)[0]
        m = KMeans(n_clusters=2).fit(X[ subIx, : ])
        newLabels = m.labels_
        labels[ subIx[ newLabels>0 ] ] = len(labelCount)
        centres[splittee,:] = m.cluster_centers_[0,:]
        centres = np.vstack([centres, m.cluster_centers_[1,:]])
        labelCount = np.bincount(labels)
    
    # euclidean distance
    D = ((centres**2).sum(1) - 2 * X.dot(centres.T)).T + (X**2).sum(1)
    centroids = D.argmin(1)
    
    return(labels, centroids)

def getFastDocs(shortQuery, voc, weights, collectionName, maxNSent=10, scoreType='mean'):
    """
    See getCleanDocs, without cleaning and passing some options for speed
    """
    log.info("Quickly retrieving documents from %s..." % collectionName)
    q = shortQuery + " {'_key': doc._key, 'sentences': slice(doc.sentences, 0, %d)}" % maxNSent
    return(scoreBatch([ defaultdict(list, doc) for doc in database.execute_query(q) ], voc, weights, scoreType=scoreType))

def getCleanDocs(shortQuery, voc, weights, collectionName):
    """
    - Complete and execute an AQL read query
    
    - Weed out all the errors, empties and so on. Current error partitions:
        -1: no sentences in doc or empty
        -2: duplicates (will keep one)
        -3: cannot calculate vector embedding
        -4: parser failed
    
    - Update them in Arango
    - Score the good ones
    - Return good docs and scores
    
    Inputs: 
    - shortQuery: partial query to return 'doc' collection elements, until 'RETURN'
        ex: 'FOR doc in news FILTER doc.partition == 0 LIMIT 5000 RETURN'
    
    Outputs: docs and vecs
    """
    # build the query (NOTE: order is important! Same as collArgs['fields'] in iterParse)
    retFields = [ '_key', 'URL', 'domain', 'parserLog', 'createdTs', 'parsedTs', 'sentences', 'contentLength', 'errorCode' ]
    ret = ", ".join([ "'{0}': doc.{0}".format(f) for f in retFields ])
    query = shortQuery + ' {%s}' % ret
    
    log.info("Carefully retrieving documents...")
    docList = list(database.execute_query(query))
    
    if len(docList) == 0:
        return(docList, np.ndarray((0, weights.shape[1])))
    
    # Integrity checks: add new fields in retFields and they will be lazily added (without re-downloading the doc) or ...
    log.info("Checking for missing fields...")
    nDocWithMiss = 0
    # don't parse known errors again
    for doc in tqdm([ d for d in docList if d['errorCode'] == 'allGood' or not d['errorCode'] ]):
        missingFields = [ f for f in retFields if emptyField(doc, f) ]
        if missingFields:
            nDocWithMiss = nDocWithMiss =+ 1
            doc, _ = addAll(doc, None, missingFields, None)
            stillMissing = [ f for f in retFields if emptyField(doc, f) ]
            if stillMissing:
                print(stillMissing)
                # TODO: instead, re-queue the docs with a 'forceUpdate' flag
                try:
                    tree = etree.ElementTree(etree.HTML(urlPoolMan.request('GET', URL).data))
                    doc, _ = addAll(doc, tree, stillMissing, None)
                except:
                    doc['errorCode'] = 'cannotDownload'
    
    log.debug("Found {0} docs with missing fields".format(nDocWithMiss))
       
    log.debug(Counter([ doc['errorCode'] for doc in docList ]))
    
    # score 'em all
    docList2, vecs = scoreBatch([ doc for doc in docList if doc['errorCode'] == 'allGood' ], voc, weights)
    log.info("Flagging incomprehensible documents...")
    naIx = np.isnan(vecs).any(1)
    for doc, isNaN in zip(docList2, naIx):
        if isNaN: doc['errorCode'] = 'cannotUnderstand'
    
    # find duplicates (in vector space)
    vecs = vecs[ ~naIx, : ]
    docList3 = [ doc for doc in docList2 if doc['errorCode'] == 'allGood' ]
    log.info("Flagging duplicates...")
    s = vecs.dot(vecs.T) # TODO: just compute the upper tri!
    dupes = (np.round(np.triu(s,1), decimals=12) == 1)
    dupeIndices = np.where(dupes)
    allDupes = pd.DataFrame({'origIx': dupeIndices[0], 'dupeIx': dupeIndices[1]})
    # dupeIx is repeated many times (it's transitive!)
    uniDupes = allDupes[ ~np.array(allDupes.duplicated(subset='dupeIx')) ]
    
    for ix in uniDupes.itertuples(index=False):
        docList3[ix.dupeIx]['errorCode'] = 'duplicated'
        docList3[ix.dupeIx]['dupliURL'] = docList3[ix.origIx]['URL']
    
    log.debug(Counter([ doc['errorCode'] for doc in docList ]))
        
    #errorDocs = [ {**doc, **{'skinnyURL': stripURL(doc['URL'])}} for doc in docList if doc['errorCode'] != 'allGood' ]
    errorDocs = []
    for doc in docList:
        if doc['errorCode'] != 'allGood':
            doc['skinnyURL'] = stripURL(doc['URL'])
            errorDocs.append(doc)
    
    if len(errorDocs) > 0:
        # add errors to <collectionName>Errors
        res = addErrors(errorDocs, collectionName)
        # remove errors from main collection and from Newbies
        res = delDocs(errorDocs, collectionName)
        res = delDocs(errorDocs, collectionName + 'Newbies')
    
    outZ = zip(*[ (doc, vec) for doc, vec in zip(docList2, vecs) if doc['errorCode'] == 'allGood' ])
    docs = list(next(outZ))
    vecs = np.vstack(next(outZ))
        
    return(docs, vecs)

def assignBatchToPartitions(shortQuery, voc, weights, collectionName, coordModel):
    
    log.info("Assigning documents to partitions:")
    docs, vecs = getCleanDocs(shortQuery, voc, weights, collectionName)
    
    log.info("Assigning geo index...")
    xx = coordModel['rfx'].predict(vecs)
    yy = coordModel['rfy'].predict(vecs)
    
    log.info("Getting pivots...")
    pivotVecs, pivotPartitionIds, pivotCounts = getPivots(collectionName)
    # find most similar pivot
    newPids = pivotPartitionIds[ np.argmax(vecs.dot(pivotVecs.T), axis=1) ]
    modPids = np.unique(newPids)
    addedPidCounts = np.bincount(modPids)
    newCounts = dict([ (int(pid), int(pivotCounts[pid] + addedPidCounts[pid])) for pid in modPids ])
    
    log.info("Adding documents to %s..." % collectionName)
    for doc, pid, x, y in zip(docs, newPids, xx, yy):
        doc['partition'] = int(pid)
        doc['tsne'] = [ x, y ]
    
    res = addDocs(docs, collectionName)
    res = delDocs(docs, collectionName + 'Newbies')
        
    pivotColl = database.col(collectionName + 'Pivots')
    log.info("Updating pivots in %s..." % pivotColl.name)
    # NOTE: without checking 'countExact' is faster BUT has concurrency issues!
    res = [ pivotColl.update_by_example({'partition': pid}, {'nDocs': count}) for pid, count in tqdm(newCounts.items()) ]
    #countExact = next(database.execute_query("FOR doc IN {0} FILTER doc.partition == {1} COLLECT WITH COUNT INTO c RETURN c".format(collection.name, pid)))
    
    return(newCounts)

def splitPartition(partitionId, voc, weights, collectionName, partitionSize=250):

    log.info("Splitting partition %d in %s (aiming at partitionSize=%d)..." % (partitionId, collectionName, partitionSize))
    shortQuery = "FOR doc in {0} FILTER doc.partition == {1} RETURN".format(collectionName, partitionId) 
    
    docs, vecs = getCleanDocs(shortQuery, voc, weights, collectionName)
    
    if len(docs) < partitionSize*2:
        log.info("Partition {0} contains {1} docs - no need to split".format(partitionId, len(docs)))
        database.col(collectionName + 'Pivots').update_by_example({'partition': int(partitionId)}, {'nDocs': len(docs)})
        return(None)
    
    X = TSNE(init='pca', verbose=1, n_components=3).fit_transform(vecs)
    
    pivotPartitionIds = getPivotIds(collectionName)
    newPartitionsIds, centroids = iterKMeans(X, partitionSize)
    
    # don't recycle partition ids
    if len(pivotPartitionIds) > 0:
        startFrom = pivotPartitionIds.max() + 1
    else:
        startFrom = 1
    
    # update Arango with the new partitions
    # TODO: use batches!
    log.info("Updating documents in %s..." % collectionName)
    collection = database.col(collectionName)
    res = [ collection.update_document(docs[k]['_key'], {'partition': int(pid+startFrom)}) for k, pid in tqdm(list(enumerate(newPartitionsIds))) ]
    
    # move away old pivot (pivot collection is uniquely indexed on URL)
    database.execute_query("FOR piv in {0}Pivots FILTER piv.partition == {1} UPDATE piv WITH {{'URL': CONCAT('old-', piv.URL)}} IN {0}Pivots".format(collection.name, partitionId))
    
    # create new pivots
    log.info("Updating pivots in %sPivots..." % collectionName)
    pivotColl = database.col(collection.name + 'Pivots')
    res = [ createPivot(docs[cix], vecs[cix,:], pivotColl, int(ix+startFrom), int((newPartitionsIds==ix).sum())) for ix, cix in tqdm(list(enumerate(centroids))) ]
    
    # delete old pivot
    database.execute_query("FOR piv in {0}Pivots FILTER piv.partition == {1} REMOVE piv IN {0}Pivots".format(collection.name, partitionId))
