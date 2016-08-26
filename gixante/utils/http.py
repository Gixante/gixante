# this file contains all functions that need a http connection
# keep it light! (don't load big files)

import sys, urllib3, certifi, re, time
from collections import defaultdict, Counter
from lxml import etree

from gixante.utils.parsing import log, classifierSplitter, knownErrors, domain, recogniseDate, cleanText, months

# start a http pool manager and connect to ArangoDB
log.info("Starting a HTTP connection...")
urlPoolMan = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())

# FUNCTIONS
# special parser 'configuration' function up here
def configureAdd(doc, tree, field, useForSentences):
    # add functions and params here to make them visible to "add"
    fields2Adders = {
        'metas'              : [getMetas, {'tree': tree}],
        'title'              : [getTitle, {'tree': tree}],
        'createdTs'          : [getCreatedTs, {'tree': tree, 'URL': doc['URL']}],
        'body'               : [getBody, {'tree': tree}],
        'sentences'          : [getSentences, {'textObj': doc[useForSentences]}],
        'contentLength'      : [getContentlength, {'sentences': doc['sentences']}],
        'links'              : [getLinks, {'tree': tree, 'URL': doc['URL']}],
        'domain'             : [getDomain, {'URL': doc['URL']}],
        'source'             : [getSource, {'URL': doc['URL']}],
        'ampLink'            : [getAmpLink, {'tree': tree}],
        'canonicalLink'      : [getCanonicalLink, {'tree': tree}],
        'ingredients'        : [getIngredients, {'tree': tree}],
        'method'             : [getMethod, {'tree': tree}],
        'ingredientDetails'  : [getIngredientDetails, {'tree': tree}],
        }
    
    return(fields2Adders[field])

def parseHTML(URL, fields, useForSentences):
    
    # DEBUG
    # useForSentences = collArgs['news']['useForSentences']
    # fields = collArgs['news']['fields']
    
    # initialise to minimal doc (check that configureAdd(out, None, 'title', useForSentences) works)
    out = {
        'URL'          : URL,
        'domain'       : domain(URL),
        'parsedTs'     : [time.time()],
        'partition'    : 0,
        'errorCode'    : 'allGood',
        'parserLog'    : [],
        useForSentences: [],
        'sentences'    : [],
        }
    
    # download the doc
    try:
        tree = etree.ElementTree(etree.HTML(urlPoolMan.request('GET', URL).data))
    except:
        log.debug("Cannot download %s!" % URL)
        out['parserLog'].append('ERROR downloading %s: ' % URL + sys.exc_info().__str__())
        out['errorCode'] = 'cannotDownload'
        return(out)
    
    # parser: add all the fields
    out = addAll(out, tree, fields, useForSentences)
    
    # check for other errors
    if len(out['sentences']) == 0 or out['contentLength'] == 0:
        out['errorCode'] = 'empty'
    elif out['domain'] == 'unknown':
        out['errorCode'] = 'unknownDomain'
        
    return(out)

# HELPERS TO parseHTML
# these functions must return a defaultdict(list) or None
# returning None will create a "field not found" WARNING in the parserLog 
def retWrapper(obj, name):
    return(defaultdict(list, {name: obj}))

def getMetas(tree):
    out = defaultdict(list)
    metas = tree.xpath("//meta[@content]")
    for meta in metas:
        contix = [ k for k, key in enumerate(meta.keys()) if key == 'content' ][0]
        if contix <= 1: 
            out[meta.values()[(1-contix)**2]] = meta.values()[contix]
    
    return(retWrapper(out, 'metas'))

def getTitle(tree):
    return(retWrapper(fullText(tree.xpath('//title[1]')[0]), 'title'))

def getCreatedTs(URL, tree=None, minTs=631152000): # ignore dates before 1990-01-01
    # try extracting it from the URL first
    try:
        dateTokens = [t for t in URL.split('/') if t in dateWords or re.match('^[0-9]{2,4}$', t)]
        urlAttempts = [ recogniseDate('/'.join(dateTokens)), recogniseDate(' '.join(dateTokens)) ]
        urlAttempts = [ a for a in urlAttempts if a ]
    except:
        urlAttempts = []
    
    # jump out if it's in the URL
    if len(urlAttempts) > 0: return(retWrapper(min(urlAttempts).timestamp(), 'createdTs'))
    
    # otherwise go up the tree if available(jump out at the first match)
    if tree:
        for txt in [ txt.strip() for txt in tree.xpath('//*/text()') ]:
            if txt and len(txt) >= 6:
                d = recogniseDate(txt)
                if d: return(retWrapper(d.timestamp(), 'createdTs'))
    
    # extract it from the header
    try:
        headAttempts = [ recogniseDate(s) for s in urlPoolMan.request('HEAD', URL).getheaders().values() ]
        headAttempts = [ a for a in headAttempts if a ]
    except:
        headAttempts = []
    
    # now the full html
    try:
        bodyAttempts = [ recogniseDate(s) for s in re.findall('>([^<]*)<', urlPoolMan.request('GET', URL).data.decode()) ]
        bodyAttempts = [ a for a in bodyAttempts if a ]
    except:
        bodyAttempts = []
    
    ts = [ a.timestamp() for a in urlAttempts + headAttempts + bodyAttempts ]
    ts = [ t for t in ts if t >= minTs ]
    
    if len(ts):
        return(retWrapper(min(ts), 'createdTs'))
    else:
        return(retWrapper(0, 'createdTs'))

def getBody(tree):
    fatPeas = findFatP(tree)
    out = defaultdict(list)
    out['usedForBody'] = fatPeas['path']
    out['body'] = cleanText(' '.join([ p[1] for p in fatPeas['peas'] ]))
    return(out)

def getSentences(textObj, splitter=classifierSplitter):
    if type(textObj) is list:
        out = [ s for s in splitter(' '.join(textObj)) ]
    elif type(textObj) is dict:
        out = [ s for s in splitter(' '.join(textObj.values())) ]
    elif type(textObj) is str:
        out = [ s for s in splitter(textObj) ]
    return(retWrapper(out, 'sentences'))
        
def getContentlength(sentences):
    return(retWrapper(sum([ len(s) for s in sentences ]), 'contentLength'))

def getLinks(tree, URL):
    dom = domain(URL)
    absoluteLinks = set(tree.xpath("//*[starts-with(normalize-space(@href),'http')]/@href | //*[starts-with(normalize-space(@href),'www')]/@href"))
    relativeLinks = tree.xpath("//*[not(starts-with(normalize-space(@href),'http'))]/@href")
    absoluteLinks.update([ dom + l for l in relativeLinks if l.startswith('/') and not l.startswith('www') ])
    return(retWrapper(list(absoluteLinks), 'links'))

def getDomain(URL):
    return(retWrapper(domain(URL), 'domain'))
    
def getSource(URL):
    return(retWrapper(URL.split('/')[2], 'source'))

def getAmpLink(tree):
    ampLink = tree.xpath("//link[@rel='amphtml']/@href")
    if len(ampLink) > 0:# and urlPoolMan.request('HEAD', ampLink[0]).status == 200:
        return(retWrapper(ampLink[0], 'ampLink'))

def getCanonicalLink(tree):
    canoLink = tree.xpath("//link[@rel='canonical']/@href")
    if len(canoLink) > 0:# and urlPoolMan.request('HEAD', canoLink[0]).status == 200:
        return(retWrapper(canoLink[0], 'canonicalLink'))


def getIngredients(tree):
    # compatible with with bbcgoodfood and bbc.co.uk/food
    ingredients = []
    for p in tree.xpath('//li[normalize-space(@itemprop)="ingredients"]'):
        out = {'fullText': fullText(p)}
        if len(p.xpath('a')) > 0:
            out.update({'shortText': p.xpath('a/text()'), 'link': p.xpath('a/@href')})
        ingredients.append(out)
    
    return(retWrapper(ingredients, 'ingredients'))

def getMethod(tree):
    # compatible with with bbcgoodfood and bbc.co.uk/food
    return(retWrapper([ fullText(p) for p in tree.xpath('//li[normalize-space(@itemprop)="recipeInstructions"]/p') ], 'method'))

def getIngredientDetails(tree):
    # compatible with with bbcgoodfood and bbc.co.uk/food
    out = defaultdict(str)
    for pp in findFatP(tree)['peas']:
        title = 'Summary'
        # find the title of the section pp belongs to
        siblings = pp[0].getparent().iterchildren()
        for s in siblings:
            if s.tag == 'h2':
                title = s.text
            elif s.tag == 'p' and len(s.text) > 0:
                txt = fullText(s)
                if txt == pp[1]:
                    out[title] = ' '.join([out[title], txt]).strip()
                    break
    
    return(retWrapper(dict(out), 'ingredientDetails'))

def addAll(doc, tree, fields, useForSentences):
    # NOTE: doc is expected to be a dict
    
    for f in fields:
        try:
            addFun, addParams = configureAdd(doc, tree, f, useForSentences)
            addition = addFun(**addParams)
            if addition:
                doc.update(addition)
                doc['parserLog'].append("Added '%s'" % f)
            else:
                doc['parserLog'].append("WARNING adding '%s': no value found" % f)
        except:
            doc['parserLog'].append("ERROR adding '%s': " % f + sys.exc_info().__str__())
            doc['errorCode'] = 'parsingError'
    
    return(doc)
    
# lxml utils
def fullText(x):
    # mimicks bs4's getText()
    txt1 = x.text
    if txt1:
        return((txt1 + x.findtext('*', default='')).strip())
    else:
        return((x.findtext('*', default='')).strip())


def findFatP(tree):
    peas = [ p for p in tree.xpath('//p') if p.text ]
    paths = [ re.sub('\[[0-9]*\]|/p.*$', '', tree.getpath(p)) for p in peas ]
    texts = [ fullText(p) for p in peas ]
    nSpaces = [ len(t.split()) if t else 0 for t in texts ]
    # sum the #spaces by path
    sumSpaces = defaultdict(int)
    for path, nSp in zip(paths, nSpaces): sumSpaces[path] += nSp
    
    if len(sumSpaces) > 0:
        # found a spot!
        whereFat = Counter(sumSpaces).most_common(1)[0][0]
        return({'peas': [ (p, txt) for p, txt, path in zip(peas, texts, paths) if path == whereFat ], 'path': whereFat})
    else:
        return({'peas': [], 'path': None})

