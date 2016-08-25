
import time, pickle, os, threading, sys, re
from flask import Flask, request
from flask_restful import Resource, Api
import numpy as np

from gixante.utils.arango import database, scoreBatch, cfg, getPivots, count
from gixante.utils.parsing import classifierSplitter, log

if len(sys.argv) < 2: sys.argv.append("news")
runDebug = sys.argv[-1].lower() == 'debug'

if runDebug: log.setLevel(0) # DEBUG is 10, but whatever

docCollName = sys.argv[1]
apiColl = database.col(docCollName + 'API')
weights, voc, coordModel = pickle.load(open(os.path.join(cfg['dataDir'], 'forManager.pkl'), 'rb'))
opposites = [ tuple(l.strip().split(',')) for l in open(os.path.join(cfg['dataDir'], 'opposites.csv'), 'r') ]
shifts = np.vstack([ weights[ voc[opp[1]] ] - weights[ voc[opp[0]] ] for opp in opposites ])
vocInv = dict([ (k, w) for w, k in voc.items() ])
pivotVecs, pivotIds, _ = getPivots(docCollName)
heartbeat = {'lastBeat': time.time(), 'period': cfg[docCollName + 'ApiHeartbeatPeriod'], 'runtime': 25}

Qend = " {{'URL': doc.URL, 'sentences': slice(doc.sentences, 0, 10), 'title': doc.title}}"
closestQ = "FOR doc IN %s FILTER doc.partition == {0} FILTER doc.contentLength > 500 FILTER doc.title != NULL RETURN" % docCollName + Qend
urlQ = "FOR doc in %s FILTER doc.URL IN {0} RETURN" % docCollName + Qend 

# helper functions
def lazyMatch(queryId, queryData, nDocs):
    apiColl.update_document(queryId, {'queryInProgress': True}) # bit overkill
    queryData['queryInProgress'] = True
    while len(queryData['docs']) < nDocs:
        q = closestQ.format(queryData['sortedPids'][ queryData['pidN'] ])
        newDocs, newVecs = scoreBatch(list(database.execute_query(q)), voc, weights, scoreType='mean')
        if len(newDocs):
            newDocSimil = newVecs.dot(np.array(queryData['vec']))
            currentSimil = [ doc['similarity'] for doc in queryData['docs'] ]
            nonDupes = np.where(~np.in1d(np.round(newDocSimil, 10), np.round(currentSimil, 10)))[0]
            for ix in nonDupes: queryData['docs'].append({'similarity': newDocSimil[ix], 'title': newDocs[ix]['title'], 'URL': newDocs[ix]['URL']})
            queryData['docs'] = sorted(queryData['docs'], key=lambda d: d['similarity'], reverse=True)
            queryData['nDocs'] = len(queryData['docs'])
        
        queryData['pidN'] += 1
        apiColl.update_document(queryId, queryData)
    
    apiColl.update_document(queryId, {'queryInProgress': False})

def backgroundChecks(docCollName):
    # updates pivots and heartbeat
    global pivotIds, pivotVecs, heartbeat
    while True:
        time.sleep(heartbeat['period'])
        pivotVecs, pivotIds, _ = getPivots(docCollName)
        heartbeat.update({'lastBeat': time.time()})

def unitVec(vec): return(vec / np.linalg.norm(vec))

def findOpposite(word, nCandidates=5000, nRelevShifts=5):
    vec = weights[ voc[word] ]
    wordL = word.lower()
    candidateIx = []
    for ix in np.argsort(-vec.dot(weights.T))[:nCandidates]:
        candidate = vocInv[ix].lower()
        if candidate != wordL and not re.match(wordL, candidate) and not re.match(candidate, wordL):
            candidateIx.append(ix)
    
    relevShifts = shifts[ np.argsort(-np.abs(vec.dot(shifts.T)))[:nRelevShifts] ]
    shiftedVecs = vec + relevShifts
    shiftRelevance = - relevShifts.dot(vec)
    
    weightedShift = unitVec(shiftRelevance.dot(shiftedVecs))
    
    simil = weights[ candidateIx, : ].dot(weightedShift)
    print([ (vocInv[candidateIx[ix]], simil[ix].mean()) for ix in np.argsort(-simil)[:10] ])
    return(vocInv[candidateIx[np.argmax(simil)]])

def contextRank(sentence, contextVec):
    words = [ w for w in classifierSplitter(sentence)[0].split() if w in voc ]
    
    if len(words) > 0:
        indices = {}
        seeds = {}
        for k in ['pos', 'neg']:
            if k == 'pos':
                seed = [voc[w] for w in words]
            elif k == 'neg':
                seed = [voc[findOpposite(w)] for w in words]
            vec = unitVec(weights[ seed ].mean(axis=0))
            sim = weights.dot(vec + contextVec/2) # TODO: why 2?
            indices[k] = np.argsort(-sim)[:100]
            seeds[k] = seed
        
        common = set(indices['pos']).intersection(set(indices['neg'])).difference(seeds)
        posIx = [ix for ix in indices['pos'] if ix not in common.difference(seeds['pos']) ][:10]
        posContext = [ vocInv[ix] for ix in posIx ]
        log.debug("Positive context: {0}".format(posContext))
        
        negIx = [ix for ix in indices['neg'] if ix not in common.difference(seeds['neg']) ][:10]
        negContext = [ vocInv[ix] for ix in negIx ]
        log.debug("Negative context: {0}".format(negContext))
        
        posVec = unitVec(weights[ posIx ].mean(axis=0))
        negVec = unitVec(weights[ negIx ].mean(axis=0))
        return((posVec, negVec, posContext, negContext))
    else:
        return(([], [], [], []))

# API classes
class Heartbeat(Resource):
    def put(self): pass
    def get(self): return(heartbeat)

class Statistics(Resource):
    def put(self, site):
        return(database.col(site + 'Stats').create_document(request.form))
    
    def get(self):
        return(database.col(docCollName).statistics)

class SimilToText(Resource):
    def put(self, csvWords=None):
        queryData = dict()
        queryData['createdTs'] = time.time()
        if 'info' in request.form: queryData['info'] = request.form['info']
        if csvWords:
            queryData['text'] = ' '.join(csvWords.split(','))
        else:
            queryData['text'] = request.form['text']
        
        queryData['docs'] = []
        queryData['nDocs'] = 0
        sentences = classifierSplitter(queryData['text'])
        rawVec = scoreBatch([ {'sentences': sentences} ], voc, weights, scoreType='mean')[1]
        if len(rawVec) == 0:
            return({'error': "Cannot understand text"})
        else:
            queryData['vec'] = [ float(x) for x in rawVec[0] ]
            simil = pivotVecs.dot(np.array(queryData['vec']))
            queryData['sortedPids'] = [ int(pid) for simil, pid in sorted(zip(simil, pivotIds), reverse=True) ]
            queryData['pidN'] = 0
            res = apiColl.create_document(queryData)
            return(res)
    
    def get(self, queryId, fields, minNumDocs=None, nMoreDocs=None, docFields=None):
        optFields = [ o is not None for o in [ minNumDocs, nMoreDocs, docFields ] ]
        assert all(optFields) or not any(optFields), "All or none of minNumDocs, nMoreDocs, docFields must be specified!"
        
        fields = fields.split(',')
        if all(optFields):
            docFields = docFields.split(',')
            queryData = apiColl.get_first_example({'_key': queryId})
            totNumDocs = max(len(queryData['docs']) + nMoreDocs, minNumDocs)
            if len(queryData['docs']) < totNumDocs and not ('queryInProgress' in queryData and queryData['queryInProgress']):
                queryData['queryInProgress'] = True
                t = threading.Thread(target=lazyMatch, args=(queryId, queryData, totNumDocs), name=queryId, daemon=True)
                t.start()
            
            # wait until at least minNumDocs docs are there...!
            while len(queryData['docs']) < minNumDocs:
                time.sleep(1)
                queryData = apiColl.get_first_example({'_key': queryId})
            
            out = dict([ (f, queryData[f]) for f in fields if f in queryData ])
            out['docs'] = [ dict([ (f, doc[f]) for f in docFields ]) for doc in queryData['docs'][:totNumDocs] ]
            return(out)
        else:
            # no docs required; just run a C++ API call
            ret = ", ".join([ "'{0}': doc.{0}".format(f) for f in fields ])
            return(next(database.execute_query("FOR doc IN {0} FILTER doc._key == '{1}' RETURN {{{2}}}".format(apiColl.name, queryId, ret))))

class Semantic(Resource):
    def put(self):
        pass
    
    def get(self, queryId, nEachSide, minNumDocs=None, rankPctDocs=0.5, semaSearch=None):
        if not minNumDocs: minNumDocs = nEachSide*100
        if not semaSearch: semaSearch = request.form['semaSearch']
        
        # fetch at least nEachSide*2 docs for now, but request minNumDocs in the background
        nDocsSoFar = next(database.execute_query("FOR doc IN {0} FILTER doc._key == '{1}' RETURN doc.nDocs".format(apiColl.name, queryId)))
        similDocs = SimilToText().get(queryId, 'nDocs,vec,semaSearch', nEachSide*2, minNumDocs - nDocsSoFar,  'URL')
        apiColl.update_document(queryId, {'semaSearch': similDocs.get('semaSearch', []) + [semaSearch]})
        
        docs = list(database.execute_query(urlQ.format([ doc['URL'] for doc in similDocs['docs'][:int(rankPctDocs*max(minNumDocs, similDocs['nDocs']))] ])))
        docs, vecs = scoreBatch(docs, voc, weights)
        posVec, negVec, posContext, negContext = contextRank(semaSearch, np.array(similDocs['vec']))
        if len(posContext) > 0:
            posSim = vecs.dot(posVec)
            negSim = vecs.dot(negVec)
            rank = np.argsort(np.abs(posSim/negSim) * (negSim - posSim))
            
            posDocs = [ {'title': docs[k]['title'], 'URL': docs[k]['URL']} for k in rank[:nEachSide] ]
            negDocs = [ {'title': docs[k]['title'], 'URL': docs[k]['URL']} for k in rank[-nEachSide:] ]
            error = None
        else:
            posDocs = negDocs = []
            error = "No context found"
        
        return({'nDocsAvail': similDocs['nDocs'], 'nDocsUsed': len(docs), 'posDocs': posDocs, 'negDocs': negDocs, 'posContext': posContext, 'negContext': negContext, 'error': error})

###

app = Flask(__name__)
api = Api(app)

checker = threading.Thread(target=backgroundChecks, args=[docCollName], name='backgroundChecks', daemon=True)
checker.start()

api.add_resource(Heartbeat, '/heartbeat')
api.add_resource(Statistics, '/getCollStats', '/addSiteStat/site=<string:site>')
api.add_resource(SimilToText, 
    '/get/id=<string:queryId>/fields=<string:fields>/minNumDocs=<int:minNumDocs>/nMoreDocs=<int:nMoreDocs>/docFields=<string:docFields>', 
    '/get/id=<string:queryId>/fields=<string:fields>',
    '/put/csvWords=<string:csvWords>', 
    '/post'
    )
api.add_resource(Semantic,
    '/semantic/id=<string:queryId>/nEachSide=<int:nEachSide>/minNumDocs=<int:minNumDocs>/rankPctDocs=<float:rankPctDocs>/semaSearch=<string:semaSearch>',
    '/semantic/id=<string:queryId>/nEachSide=<int:nEachSide>/semaSearch=<string:semaSearch>',
    '/semantic/id=<string:queryId>/nEachSide=<int:nEachSide>/minNumDocs=<int:minNumDocs>/rankPctDocs=<float:rankPctDocs>', # POST
    '/semantic/id=<string:queryId>/nEachSide=<int:nEachSide>', # POST
    )

log.info("Ready for business!")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=runDebug)
    
log.info("Goodbye!") 


