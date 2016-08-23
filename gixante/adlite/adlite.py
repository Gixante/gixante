
import os, sys, pickle, re, time, gzip
import numpy as np

sys.path.append('/home/bean/Code/Python')
from flask import Flask, request, url_for, render_template
from gixante.utils.parsing import classifierSplitter, log
from gixante.utils.arango import database, scoreBatch, getPivots, cfg

from pprint import pprint
from collections import Counter
from random import sample

# SETUP
app = Flask(__name__)
app.config.from_object(__name__)

# Load default config and override config from an environment variable
app.config.update(dict(
    SECRET_KEY='development key',
    USERNAME='admin',
    PASSWORD='default'
))
# app.config.from_envvar('FLASKR_SETTINGS', silent=True)

docCollName = 'news'
coll = database.col(docCollName)
queryDataColl = database.col('demoData')
#weights, voc, coordModel = pickle.load(open(os.path.join(cfg['dataDir'], 'forManager.pkl'), 'rb'))
weights, voc, coordModel = pickle.load(gzip.open(os.path.join(cfg['dataDir'], 'forAdlite.pkl.gz'), 'rb'))
opposites = [ tuple(l.strip().split(',')) for l in open(os.path.join(cfg['dataDir'], 'opposites.csv'), 'r') ]
shifts = np.vstack([ weights[ voc[opp[1]] ] - weights[ voc[opp[0]] ] for opp in opposites ])
vocInv = dict([ (k, w) for w, k in voc.items() ])
pivotVecs, pivotPartitionIds, pivotCounts = getPivots(docCollName)
pickle.dump((pivotVecs, pivotPartitionIds, pivotCounts), open('/tmp/pivots.pkl', 'wb'))
#pivotVecs, pivotPartitionIds, pivotCounts =  pickle.load(open('/tmp/pivots.pkl', 'rb'))
Qend = " {{'URL': doc.URL, 'sentences': slice(doc.sentences, 0, 10), 'title': doc.title}}"
closestQ = "FOR doc in %s FILTER doc.partition in {0} FILTER doc.contentLength > 500 FILTER doc.title != NULL RETURN" % docCollName + Qend
urlQ = "FOR doc in %s FILTER doc.URL IN {0} RETURN" % docCollName + Qend
nPerFetch = 5
nReturnDocs = 25
NUerror = "Sorry - your search was not understood! Please try again - note that this demo is for English only and is case sensitive."

### FUNCTIONS

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

# ROUTES
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/publishers')
def publishers():
    return render_template('publishers.html', showNavBar=None)

@app.route('/advertisers')
def advertisers():
    return render_template('advertisers.html')

#TODO: why and contact
#TODO: active list item in navbar

@app.route('/about')
def about():
    nDocs = coll.statistics['alive']['count']
    millDocs = "{:,}".format((nDocs - nDocs % 1e5) / 1e6)
    return render_template('about.html', millDocs=millDocs)
    
@app.route('/why')
def why():
    return render_template('why.html')

@app.route('/contact')
def contact():
    return(render_template('contact.html'))

@app.route('/add_contact', methods=['POST', 'GET'])
def add_contact():
    print(request.form)
    return(render_template('contact.html'))

# AD DEMO
@app.route('/ad_demo')
def ad_demo():
    nDocs = "{:,}".format(coll.statistics['alive']['count'])
    return render_template('ad_demo.html', nDocs=nDocs)

@app.route('/ad_initial', methods=['POST', 'GET'])
def ad_initial():
    print(request.form)
    queryData = dict()
    queryData['createdTs'] = time.time()
    queryData['paragraph'] = request.form['paragraph']
    queryData['docs'] = []
    sentences = classifierSplitter(queryData['paragraph'])
    rawVec = scoreBatch([ {'sentences': sentences} ], voc, weights, scoreType='mean')[1]
    if len(rawVec) == 0:
        return(form(error=NUerror))
    else:
        searchVec = rawVec[0]
        queryData['searchVec'] = [ float(x) for x in searchVec ]
        simil = pivotVecs.dot(np.array(queryData['searchVec']))
        queryData['sortedPids'] = [ int(pid) for simil, pid in sorted(zip(simil, pivotPartitionIds), reverse=True) ]
        queryData['callN'] = 0
        queryData['semaSearch'] = []
        res = queryDataColl.create_document(queryData)
        return(ad_fetch(res['_key']))

@app.route('/ad_fetch/<queryId>')
def ad_fetch(queryId):
    queryData = queryDataColl.get_first_example({'_key': queryId})
    q = closestQ.format(queryData['sortedPids'][ queryData['callN']*nPerFetch:(queryData['callN']+1)*nPerFetch ])
    newDocs, newVecs = scoreBatch(list(database.execute_query(q)), voc, weights, scoreType='mean')
    newDocSimil = newVecs.dot(np.array(queryData['searchVec']))
    currentSimil = [ doc['similarity'] for doc in queryData['docs'] ]
    nonDupes = np.where(~np.in1d(np.round(newDocSimil, 3), np.round(currentSimil, 3)))[0]
    
    for ix in nonDupes:
        # PATCH: some titles are in a list (old format I guess)
        if type(newDocs[ix]['title']) is list:
            log.warning("Document {0} has a listed title!".format(newDocs[ix]['URL']))
            newDocs[ix]['title'] = newDocs[ix]['title'][0]
            database.col(docCollName).update_by_example({'URL': newDocs[ix]['URL']}, {'title': newDocs[ix]['title']})
        # PATCH END
        queryData['docs'].append({'similarity': newDocSimil[ix], 'title': newDocs[ix]['title'], 'URL': newDocs[ix]['URL']})
        
    results = sorted(queryData['docs'], key=lambda d: d['similarity'], reverse=True)[:nReturnDocs]
    queryData['nGoodResults'] = len([ d for d in queryData['docs'] if d['similarity'] > results[-1]['similarity']/2 ])
    queryData['callN'] += 1
    queryDataColl.update_document(queryId, queryData)
    return(render_template('ad_results.html', **{'paragraph': queryData['paragraph'], 'results': results, 'nResults': queryData['nGoodResults'], '_queryId': queryId}))

@app.route('/ad_semantic/<queryId>', methods=['POST', 'GET'])
def ad_semantic(queryId):
    queryData = queryDataColl.get_first_example({'_key': queryId})
    posDocs = negDocs = posWords = negWords = error = None
    nSemaDocs = min(nReturnDocs*20, queryData['nGoodResults'])
    if 'semaSearch' in request.form:
        queryData['semaSearch'].append(request.form['semaSearch'])
        bestURLs = [ doc['URL'] for doc in sorted(queryData['docs'], key=lambda d: d['similarity'], reverse=True)[:nSemaDocs] ]
        docs = list(database.execute_query(urlQ.format(bestURLs)))
        docs, vecs = scoreBatch(docs, voc, weights)
        
        posVec, negVec, posContext, negContext = contextRank(queryData['semaSearch'][-1], np.array(queryData['searchVec']))
        print(len(posContext))
        if len(posContext) > 0:
            posSim = vecs.dot(posVec)
            negSim = vecs.dot(negVec)
            rank = np.argsort(np.abs(posSim/negSim) * (negSim - posSim))
            
            posDocs = [ {'title': docs[k]['title'], 'URL': docs[k]['URL']} for k in rank[:10] ]
            negDocs = [ {'title': docs[k]['title'], 'URL': docs[k]['URL']} for k in rank[-10:] ]
            posWords = ', '.join(posContext)
            negWords = ', '.join(negContext)
            queryDataColl.update_document(queryId, queryData)
        else:
            log.debug('ERROR!')
            error = NUerror
    
    semaSearch = queryData['semaSearch'][-1] if len(queryData['semaSearch']) > 0 else None
    return(render_template('ad_semantic.html', **{'paragraph': queryData['paragraph'], 'posDocs': posDocs, 'negDocs': negDocs, 'semaSearch': semaSearch, 'posWords': posWords, 'negWords': negWords, 'error': error, '_queryId': queryId, 'nSemaDocs': nSemaDocs, 'nGoodResults': queryData['nGoodResults']}))
