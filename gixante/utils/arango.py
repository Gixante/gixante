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
from collections import defaultdict, Counter

from gixante.pyrango import Arango
from gixante.utils.common import log, knownErrors, stripURL, cfg
from gixante.utils.rabbit import hat

# STARTUP
log.info("Connecting to ArangoDB...")
database = Arango(host=cfg['arangoIP'], port=cfg['arangoPort'], password=cfg['arangoPwd']).db('catapi')

# FUNCTIONS
# Error handler
def addErrors(errorDocs, collectionName):
    # errorDocs is a list of dictionaries containing at least: 'URL' and 'errorCode' (a key from knownErrors)
    
    validDocs = [ doc for doc in errorDocs if 'errorCode' in doc and doc['errorCode'] != 'allGood' and 'URL' in doc ]
    
    for doc in validDocs:
        if 'skinnyURL' not in doc: doc['skinnyURL'] = stripURL(doc['URL'])
    
    # check if errors are already in DB
    newErrSkinnies = missing('skinnyURL', '^%sErrors' % collectionName, [ e['skinnyURL'] for e in errorDocs ])
    
    errors = []
    for doc in [ d for d in validDocs if d['skinnyURL'] in newErrSkinnies ]:
        err = dict([ (k, v) for k, v in doc.items() if k in doc and k in ['URL', 'skinnyURL', 'errorCode', 'parserLog', 'dupliURL', 'otherErrorMessage', 'usedForBody' ] ])
        err['raisedTs'] = time.time()
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
    URLs = exist('URL', '^%s$' % collectionName, [ doc['URL'] for doc in docs ])
    if len(URLs):
        q = "FOR doc in {0} FILTER doc.URL in {1} REMOVE doc IN {0} OPTIONS {{ ignoreErrors: true }}".format(collectionName, list(URLs))
        res = database.execute_query(q)
    
    log.info("Deleted {0} documents from '{1}'".format(len(URLs), collectionName))
    
    return(res)

def delErrors(errors, collectionName):
    res = []
    # TODO: use batches!
    skinnyURLs = exist('skinnyURL', '^%sErrors$' % collectionName, [ err['skinnyURL'] if 'skinnyURL' in err else stripURL(err['URL']) for err in errors])
    if len(skinnyURLs):
        q = "FOR doc in {0}Errors FILTER doc.skinnyURL in {1} REMOVE doc IN {0}Errors OPTIONS {{ ignoreErrors: true }}".format(collectionName, list(skinnyURLs))
        res = database.execute_query(q)
    
    log.info("Deleted {0} errors from '{1}Errors'".format(len(skinnyURLs), collectionName))
    
    return(res)

def delEverywhere(docs, collectionName):
    appendices = ['', 'Newbies', 'Errors']
    collToCheck = [ c for c in [ collectionName + apx for apx in appendices ] if c in database.collections['user'] ]
    res = []
    for col in collToCheck:
        if col.endswith('Errors'):
            res.extend(delErrors(docs, collectionName))
        else:
            res.extend(delDocs(docs, col))
    return(res)

def getRandomLinks(collectionName, nDocs=25):
    randomQ = "FOR doc in {0} FILTER doc.partition > 0 FILTER doc.links != NULL LIMIT {1} SORT RAND() LIMIT {2} RETURN {{'links': doc.links, 'ref': doc.URL}}"
    return(database.execute_query(randomQ.format(collectionName, nDocs**2, nDocs)))

def exist(what, collRgx, stringIter):
    
    # remove quotes and duplicates
    strings = [re.sub("'", "%27", s.rstrip('\\')) for s in stringIter if len(re.findall('{|}', s)) == 0]
    
    collToCheck = [ coll for coll in database.collections['user'] if re.match(collRgx, coll) ]
    existQ = "FOR doc in {{0}} filter doc.{1} in {0} RETURN doc.{1}".format(strings, what)
    
    out = set()
    for coll in collToCheck:
        out.update(set(database.execute_query(existQ.format(coll))))
    
    return(out)

def missing(what, collRgx, stringIter):
    
    # remove quotes and duplicates
    strings = set([ re.sub("'", "%27", s.rstrip('\\')) for s in stringIter if len(re.findall('{|}', s)) == 0 ])
    
    collToCheck = [ coll for coll in database.collections['user'] if re.match(collRgx, coll) ]
    # don't use .format: curly brackets mess it up!
    existQ = " FILTER doc.{1} in {0} RETURN doc.{1}".format(list(strings), what)
    
    for coll in collToCheck:
        strings = strings.difference(set(database.execute_query("FOR doc in "+coll+existQ)))
    
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
    loadFromScratch = True
    
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
    Reads a batch of docs (list of dicts, expecting 'sentences' as a key containing the text)
    Returns an array of embeddings (one per doc)
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
    
    if len(ixBySentence) == 0: return(np.array([]).reshape(0,0))
    
    # now expand each sentence word by word
    ixByWord = list(zip( *chain( *[ product( *ix ) for ix in ixBySentence ] ) ))
    if len(ixByWord) == 0: return(np.array([]).reshape(0,0))
    
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
    return(scoresByDoc / norms2)

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

def validate(docs, parser, restrictValidationToFields=None):
    if not parser: return(docs, [])
    
    valid = []
    invalid = []
    
    log.info("Checking for missing fields...")
    for doc in tqdm(docs):
        isValid = parser.isValid(doc, restrictValidationToFields)
        if not isValid:
            # reparse the doc
            doc = parser.strip(parser.parseDoc(doc))
            isValid = parser.isValid(doc, restrictValidationToFields)
        
        if isValid:
            valid.append(doc)
        else:
            invalid.append(doc)
    
    return(valid, invalid)

def validateNPartitions(collectionName, nPartitions, parser):
    pids = list(database.execute_query("FOR piv IN {0}Pivots return piv.partition".format(collectionName)))
    for pid in sample(pids, nPartitions):
        getValidDocs("FILTER doc.partition == {0}".format(pid), collectionName, returnFields=[], parser=parser)

def getValidDocs(queryFilter, collectionName, returnFields=[], parser=None):
    
    if returnFields:
        ret = ", ".join([ "'{0}': doc.{0}".format(f) for f in returnFields ])
        query = "FOR doc in {0} {1} RETURN {{{2}}}".format(collectionName, queryFilter, ret)
    else:
        query = "FOR doc in {0} {1} RETURN doc".format(collectionName, queryFilter)
    
    docList = list(database.execute_query(query))
    
    validDocs, invalid = validate(docList, parser, returnFields)
    
    if invalid:
        baseQ = re.sub("[A-Z].*", "", collectionName)
        log.debug("Found {0} docs (out of {1}) with missing fields (will re-queue to '{2}' and remove from database)".format(len(invalid), len(docList), baseQ))
        
        usableDocs = []
        for doc_ in invalid:
            if 'URL' in doc_:
                usableDocs.append({'URL': doc_['URL']})
                for f in ['skinnyURL', 'domain', 'refURL']:
                    if f in doc_: usableDocs[-1][f] = doc_[f]
        
        hat.multiPublish(baseQ, [json.dumps(d) for d in usableDocs])
        res = delEverywhere([d for d in usableDocs], collectionName)

    return(validDocs)
    
def cleanScoreDocs(docs, voc, weights, collectionName):
    # NOTE: docs need to contain 'URL' and 'sentences' (for scoreBatch)
    
    log.debug(Counter([ doc['errorCode'] for doc in docs ]))
    # score 'em all and flag the ones the model cannot understand
    docs2 = [ doc for doc in docs if doc.get('errorCode', 'allGood') == 'allGood' ]
    vecs = scoreBatch(docs2, voc, weights)
    log.info("Flagging incomprehensible documents...")
    naIx = np.isnan(vecs).any(1)
    for doc, isNaN in zip(docs2, naIx):
        if isNaN: doc['errorCode'] = 'cannotUnderstand'
    
    log.debug(Counter([ doc['errorCode'] for doc in docs ]))
    # find duplicates (in vector space)
    vecs = vecs[ ~naIx, : ]
    docs3 = [ doc for doc in docs2 if doc['errorCode'] == 'allGood' ]
    log.info("Flagging duplicates...")
    s = vecs.dot(vecs.T) # TODO: just compute the upper tri!
    dupes = (np.round(np.triu(s,1), decimals=12) == 1)
    dupeIndices = np.where(dupes)
    allDupes = pd.DataFrame({'origIx': dupeIndices[0], 'dupeIx': dupeIndices[1]})
    # dupeIx is repeated many times (it's transitive!)
    uniDupes = allDupes[ ~np.array(allDupes.duplicated(subset='dupeIx')) ]
    for ix in uniDupes.itertuples(index=False):
        docs3[ix.dupeIx]['errorCode'] = 'duplicated'
        docs3[ix.dupeIx]['dupliURL'] = docs3[ix.origIx]['URL']
    
    log.debug(Counter([ doc['errorCode'] for doc in docs ]))
    # deal with errors
    errorDocs = [ doc for doc in docs if doc['errorCode'] != 'allGood' ]
    if len(errorDocs) > 0:
        # remove errors from docs
        res = delDocs(errorDocs, collectionName)
        res = delDocs(errorDocs, collectionName+'Newbies')
        # add errors to <collectionName>Errors
        res = addErrors(errorDocs, collectionName)
        
    # this is just returning docs, vecs :)
    outZ = list(zip(*[ (doc, vec) for doc, vec in zip(docs2, vecs) if doc['errorCode'] == 'allGood' ]))
    return(list(outZ[0]), np.vstack(outZ[1]))

def fromNewbies(queryFilter, collectionName, voc, weights, coordModel, parser):
    
    log.info("Loading documents from {0}Newbies...".format(collectionName))
    docs, vecs = cleanScoreDocs(getValidDocs(queryFilter, collectionName+'Newbies', returnFields=[], parser=parser), voc, weights, collectionName)
    
    if not docs: return({})
    
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
    
    log.info("Adding documents to {0}...".format(collectionName))
    for doc, pid, x, y in zip(docs, newPids, xx, yy):
        doc['partition'] = int(pid)
        doc['tsne'] = [ x, y ]
        # remove hidden keys (like primary keys and keys used in parsing)
        for k in [ key for key in doc.keys() if key.startswith('_') ]:
            if k in doc: doc.pop(k)
    
    res = addDocs(docs, collectionName)
    res = delDocs(docs, collectionName + 'Newbies')
        
    pivotColl = database.col(collectionName + 'Pivots')
    log.info("Updating pivots in %s..." % pivotColl.name)
    # NOTE: without checking 'countExact' is faster BUT has concurrency issues!
    res = [ pivotColl.update_by_example({'partition': pid}, {'nDocs': count}) for pid, count in tqdm(newCounts.items()) ]
    #countExact = next(database.execute_query("FOR doc IN {0} FILTER doc.partition == {1} COLLECT WITH COUNT INTO c RETURN c".format(collection.name, pid)))
    
    return(newCounts)

def splitPartition(partitionId, voc, weights, collectionName, parser=None, partitionSize=250):

    log.info("Splitting partition %d in %s (aiming at partitionSize=%d)..." % (partitionId, collectionName, partitionSize))
    queryFilter = "FILTER doc.partition == {0}".format(partitionId)
    returnFields = ['_key', 'URL', 'sentences', 'errorCode']
    
    docs, vecs = cleanScoreDocs(getValidDocs(queryFilter, collectionName, returnFields=returnFields, parser=parser), voc, weights, collectionName)
    
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
