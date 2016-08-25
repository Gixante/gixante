"""
This module contains functions used in the content API
Keep it light! (don't load big files)
"""

import time, os, re, json, pickle
from flask import Flask, request
from flask_restful import Resource, Api
import numpy as np

from gixante.utils.arango import scoreBatch, getPivots
from gixante.utils.parsing import classifierSplitter, log

# CONFIG
# load config file
cfg = json.load(open(os.path.join(*(['/'] + __file__.split('/')[:-1] + ['config.json']))))

def keepPivotsUpdated(docCollName):
    global pivotIds, pivotVecs
    while True:
        time.sleep(300)
        pivotVecs, pivotIds, _ = getPivots(docCollName, pivotIds, pivotVecs)
        pickle.dump((pivotVecs, pivotIds), open('/tmp/pivots.pkl', 'wb'))

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
