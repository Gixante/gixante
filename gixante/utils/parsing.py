"""
This module contains all functions used to parse, split etc. (and some config for now)
Keep it light! (don't load big files)
"""

import logging, os, re, sys, json
from datetime import datetime as dt
from pandas import DataFrame, concat
from numpy import array, where
from enum import Enum
from collections import OrderedDict
from gensim import utils

# define logger
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
log = logging.getLogger("PID:%d" % os.getpid())
log.setLevel(logging.DEBUG)

# CONFIG
# load config file
configFile = os.path.join(*(['/'] + __file__.split('/')[:-1] + ['config.json']))
log.debug("Loading config from %s..." % configFile)
cfg = json.load(open(configFile))

removeAfter = ['^#', '\?', '\=']
staticFeatures = ['leanPct', 'length', 'nDash', 'nTokens']

# Will try to match domains in order
domain2coll = OrderedDict([
    ('web.archive.org/'       , None),
    ('www.theguardian.com/'   , 'news'),
    ('www.telegraph.co.uk/'   , 'news'),
    ('www.independent.co.uk/' , 'news'),
    ('www.standard.co.uk/'    , 'news'),
    ('/techcrunch.com/'       , 'news'),
    ('/www.techcrunch.com/'   , 'news'),
    ('bbc.co.uk/news'         , 'news'),
    ('bbc.co.uk/food/recipes' , 'recipes'),
    ('bbc.co.uk/food'         , 'otherBBCfood'),
    ('bbcgoodfood.com/recipes', 'recipes'),
    ('bbcgoodfood.com/'       , 'otherBBCfood'),
    ('unknown'                , None),
    ])

# Errors
knownErrors = {
    'allGood'         : 'no error',
    'otherError'      : 'of an unknown error',
    'empty'           : 'no valid text was found',
    'dodgy'           : 'the screening model thinks the URL will result in an error',
    'duplicated'      : 'it has identical embedding to another one',
    'cannotUnderstand': 'no word in its content is in the model dictionary',
    'parsingError'    : 'parser failed to add at least one of the required fields',
    'cannotDownload'  : 'it could not be downloaded',
    'unknownDomain'   : 'its domain was not recognised',
    }

# FUNCTIONS
# keep some diacriticals
PAT_SPACER = re.compile('([\.,;:!?\(\)])', re.UNICODE)
# remove everything that is not letter, number or in PAT_SPACER
PAT_REMOVE = re.compile(re.sub('^\(\[', '([^A-z 0-9', PAT_SPACER.pattern), re.UNICODE)
# ignore capitalisation at the beginning of string or after a full stop
PAT_TRIVIAL_CAPITAL = re.compile('(\. |^)([A-Z])([^A-Z])', re.UNICODE)

def classifierParser(text):
    # NOTE: this assumes that the text has already been cleaned (see newsparser.py)
    text = PAT_SPACER.sub(r' \1 ', text)
    text = re.sub(' {2,}', ' ', text)
    text = PAT_TRIVIAL_CAPITAL.sub(lambda x: x.group(0).lower(), text)
    return(text.strip())

def classifierSplitter(text, parser=classifierParser):
     return(re.sub('([A-z0-9]{2,} \.)[^A-z]', r'\1@', parser(text)).split('@'))

def cleanText(text):
    text = utils.deaccent(utils.to_unicode(' ' + text, errors='ignore'))
    text = PAT_REMOVE.sub(' ', text)
    return(re.sub('  +', ' ', text).strip())

def domain(URL):
    checkThis = re.sub('[^A-Za-z0-9_.~\-/:].*', '', URL)
    for dom in domain2coll.keys():
        if dom in checkThis: break
    return(dom)

def emptyField(doc, field, empties=[None, [], '', ['']]):
    # a field can be none, missing, empty, whatever
    return(field not in doc or doc[field] in empties)

# try to match a date in a string
months = array([ m for ml in zip(*[ dt.strftime(dt.strptime('%.2d' % (k+1), '%m'), '%b %B').lower().split() for k in range(12) ]) for m in ml ])
recognisedPatterns = {
    '[0-9]{2,2}-[0-9]{2,2}-[0-9]{4,4}': "%d-%m-%Y",
    '[0-9]{2,2}/[0-9]{2,2}/[0-9]{4,4}': "%d/%m/%Y",
    '[0-9]{4,4}-[0-9]{2,2}-[0-9]{2,2}': "%Y-%m-%d",
    '[0-9]{4,4}/[0-9]{2,2}/[0-9]{2,2}': "%Y/%m/%d",
    '^[0-9]{8,8}$': "%Y%m%d",
    }

def recogniseDate(datestr):
    
    datestr = re.sub('[^A-z0-9-/:]', ' ', datestr)
        
    try:
        thereAreWords = len(re.findall('[A-z]', datestr)) > 0
                
        d = [0]*3
        for sp in re.sub(',', ' ', datestr).split():
            if thereAreWords and sp.lower() in months:
                d[1] = (where(sp.lower() == months)[0][0] % 12)+1
            elif re.match('^[0-9]{1,2}$', sp):
                d[2] = int(sp)
            elif re.match('^[0-9]{4,4}$', sp):
                d[0] = int(sp)
            else:
                for pat in recognisedPatterns:
                    if re.match(pat, sp):
                        out = dt.strptime(re.match(pat, sp).group(), recognisedPatterns[pat])
                        if out.timestamp() > 0: return(out)
        
        if any([ x==0 for x in d ]): return None
        
        out = dt.strptime('%.4d%.2d%.2d' % tuple(d), '%Y%m%d')
        
        if out.timestamp() > 0: return(out)
    
    except:
        #log.error(datestr, sys.exc_info().__str__())
        return(None)

# URL parsing utils
commonWords = set(' '.join([ dt.strftime(dt.strptime('%.2d' % (k+1), '%m'), '%A %a %B %b').lower() for k in range(12) ]).split())
commonWords.update(set(['http', 'https', 'html', 'jpg', 'css', 'js']))
delRgx = re.compile('|'.join([ "%s.*" % d for d in removeAfter ]), re.UNICODE)

def stripURL(URL):
    # remove text after one of the characters in removeAfter, then strip each token of commonWords
    tokens = []
    for token in URL.split('/'):
        token = delRgx.sub('', token).lower()     
        splitted = re.split('([^A-z])', token)
        skinnySplits = []
        lastSeparator = ''
        for split in splitted:
            if split in commonWords:
                continue
            elif len(split) == 1:
                lastSeparator = split
            elif re.match('^[A-z]+$', split):
                if len(skinnySplits) > 0: skinnySplits.append(lastSeparator)
                skinnySplits.append(split)
        tokens.append(''.join(skinnySplits))
        
    return('/'.join([ t for t in tokens if len(t) > 0 ]))

def urlFeat(URL):
                   
    skinnyURL = stripURL(URL)
    skinnyTokens = skinnyURL.split('/')
        
    return({
        'length': len(URL),
        'tokens': skinnyTokens,
        'nTokens': len(skinnyTokens),
        'nDash': len(re.findall('-', URL)),
        'URL': URL,
        'skinnyURL': skinnyURL,
        'leanPct': len(skinnyURL) / len(URL),
        })

def dodgyFeatures(featURLs, relevTokens):
    tokens = [ set(f['tokens']) for f in featURLs ]
    binTokens = DataFrame(array([[ 1 if t in toks else 0 for t in relevTokens ] for toks in tokens ]), columns=relevTokens)
    features = DataFrame(featURLs).loc[ :, staticFeatures ]
    features = concat([ features, binTokens ], axis=1)
    return(features[ staticFeatures + relevTokens ])

