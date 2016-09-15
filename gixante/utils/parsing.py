import sys, time, re, os

import pandas as pd
from datetime import datetime as dt
from lxml import etree
from collections import defaultdict, Counter
from Levenshtein import distance

from gixante.utils.common import log, classifierSplitter, recogniseDate, cleanText, stripURL, knownErrors

# load config file
configFile = os.path.join(*(['/'] + __file__.split('/')[:-1] + ['scraperConfig.tsv']))
log.debug("Loading scraping config from %s..." % configFile)

# configure module
global domains, useForSentences, splitter, requiredFields, domain2coll

def configForCollection(collectionName=None, _requiredFields='setFromCollection'):
    global domains, useForSentences, splitter, requiredFields, domain2coll
    
    splitter = classifierSplitter
    parserConfig = parserConfig = pd.read_csv(configFile, sep='\t', header=0)
    domain2coll = dict(zip(parserConfig.domain, parserConfig.collection))
    
    if collectionName:
        parserConfig = parserConfig.loc[ parserConfig.collection == collectionName, : ]
        defaultRequiredFields = set([f for fields in parserConfig.fields.unique() for f in fields.split(',')  ])
        ufs = parserConfig.useForSentences.unique()
        if len(ufs) != 1:
            raise RuntimeError("Multiple values for 'useForSententes for collection '{0}'; check config".format(collectionName))
        useForSentences = ufs[0]        
    else:
        defaultRequiredFields = []
        useForSentences = None
        
    domains = list(parserConfig.domain)
    requiredFields = _requiredFields if type(_requiredFields) is list else defaultRequiredFields

# FUNCTIONS
def fullText(x):
    # mimicks bs4's getText()
    return(''.join(x.itertext()))

def minDistance(str1, str2):
    if len(str1) < len(str2):
        shorter, longer = str1, str2
    else:
        shorter, longer = str2, str1

    nChar = len(shorter)
    
    minD = 1
    for k in range(len(longer)-nChar+1):
        d = distance(shorter, longer[k:k+nChar]) / nChar
        if d <= minD:
            minD, minK = d, k
        elif minD < 0.5 and d - minD > 0.1:
            break
    
    return(minD, minK)


# CLASSES
class Field:
    def __init__(self, name, default=None):
        self.name = name
        self._addFunctions = [ self.__getattribute__(a) for a in sorted(self.__dir__()) if re.match('^_add[0-9]', a) ]
        self._useAddFunction = 0
        self._additionalRequiredKwargs = dict()
        self.default = default
    
    def add(self, doc):
        fun, keys = self._addFunKeys()
        if not fun: raise NotImplementedError("No '_add0' method specified for class {0}".format(self.__class__.__name__))
        cleanKeys = [ re.sub('^_%s' % self.__class__.__name__, '', k) for k in keys ]
        args = [ doc[re.sub('^_%s' % self.__class__.__name__, '', k)] for k in cleanKeys if k in doc ]
        kwargs = dict([ (k, doc[v]) for k, v in self._additionalRequiredKwargs.items() if v in doc ])
        addendum = fun(*args, **kwargs)
                
        if self.containsValid(addendum):
            doc.update(addendum)
            status, msg = ("OK", "Added '{0}'".format(self.name))
        else:
            junk = str(addendum.pop(self.name) if self.name in doc else None)
            junk = junk[:50] + '...' if len(junk) > 50 else junk
            status, msg = ("WARNING", "Not adding invalid value ({0}) for '{1}'".format(junk, self.name))
        
        return(doc, status, msg)
        
    def _addFunKeys(self, which=None):
        if which == None: which = self._useAddFunction
        if self._addFunctions:
            addFun = self._addFunctions[ which ]
            addKeysRaw = addFun.__code__.co_varnames[:addFun.__code__.co_argcount]
            return(addFun, addKeysRaw)
        else:
            return(None, [])
    
    def _loopAddMethod(self):
        if self._useAddFunction + 1 < len(self._addFunctions):
            self._useAddFunction = self._useAddFunction + 1
        else:
            self._useAddFunction = 0
        
        return(self._useAddFunction)
    
    def requiredKeys(self, justCurrent=True):
        if justCurrent:
            kk = [self._useAddFunction]
        else:
            kk = range(len(self._addFunctions))
        
        reqForAdd = set()
        for k in kk:
            reqForAdd.update([ re.sub('^_%s' % self.__class__.__name__, '', k) for k in self._addFunKeys(k)[1] if k not in [ 'self', self.name, '_textObj'] ])
        
        return(list(reqForAdd) + list(self._additionalRequiredKwargs.values()))
    
    def _rw(self, x):
        return({self.name: x})
    
    def containsValid(self, whatDoc):
        return(whatDoc is not None and self.isValid(whatDoc.get(self.name, None)))
            
    def isValid(self, what):
        raise NotImplementedError("Subclass must implement abstract method")

class Parser:
    def __init__(self, urlPoolMan=None):
        
        self.urlPoolMan = urlPoolMan
        self.requiredKeys = requiredFields
                
        availFields = [ v() for k, v in globals().items() if type(v) is type and issubclass(v, Field) and k not in ['Field', 'ErrorCode'] ]
        self.fields = dict([ (f.name, f) for f in availFields if f._addFunctions ])
        
        missing = set(requiredFields).difference(self.fields.keys())
        if len(missing):
            raise NotImplementedError("Not configured to parse {0}".format(missing))
            
        self._fieldOrders = dict()
    
    def _log(self, level, msg):
        return("{0}: {1} on {2}".format(level.upper(), msg, dt.strftime(dt.now(), "%Y-%m-%d at %H:%M:%S")))
    
    def _maxOrder(self, initialKeys):
        nextIterKeys = initialKeys
        orderedKeys = []
        availKeys = []
        
        while nextIterKeys:
            availKeys.append(nextIterKeys)
            orderedKeys.extend(nextIterKeys)
            nextIterKeys = [ fn for fn, f in self.fields.items() if fn not in orderedKeys and all([ r in orderedKeys for r in f.requiredKeys() ]) ]
        
        missingReqKeys = [ r for r in self.requiredKeys if r not in orderedKeys ]
        if missingReqKeys:
            tryChangingTheseFields = [ f for f in self.fields.values() if f.name not in orderedKeys and len(f._addFunctions) > 1 ]
        else:
            tryChangingTheseFields = []
        
        return(orderedKeys, missingReqKeys, tryChangingTheseFields)
    
    def _configureFields(self, doc):
        
        initialKeys = list(doc.keys())
        orderKey = ','.join(sorted(initialKeys))
        
        if orderKey in self._fieldOrders:
            # set the add functions from the field stored
            fO = self._fieldOrders[orderKey]
            for key, funN in fO:
                if funN: self.fields[key]._useAddFunction = funN
            orderedKeys = list(zip(*fO))[0]
            msg = self._log("OK", "Parser configured")
        else:
            # reset the parser to defaults; then recalculate
            for f in self.fields.values(): f._useAddFunction = 0
            orderedKeys, missingReqKeys, tryChangingTheseFields = self._maxOrder(initialKeys)
            
            # try using different 'add' implementations if the default didn't work
            while tryChangingTheseFields:
                #print(tryChangingTheseFields[0].name, tryChangingTheseFields[0]._useAddFunction)
                nextUp = tryChangingTheseFields[0]._loopAddMethod()
                if nextUp == 0:
                    # we've come full circle with this one, move on
                    tryChangingTheseFields = tryChangingTheseFields[1:]
                else:
                    # recalculate
                    orderedKeys, missingReqKeys, tryChangingTheseFields = self._maxOrder(initialKeys)
            
            if missingReqKeys:
                msg = self._log("ERROR", "Cannot parse required fields {0} from initial keys {1}".format(missingReqKeys, initialKeys))
            else:
                msg = self._log("OK", "Parser configured")
            
            self._fieldOrders[orderKey] = [ (k, self.fields[k]._useAddFunction if k not in initialKeys else None) for k in orderedKeys ]
            
        return(orderedKeys, msg)
    
    def parseDoc(self, _doc):
        
        # initialise the parsed doc with a log
        doc = {'parserLog': _doc.get('parserLog', [])}
        
        # valudate existing fields
        for key, val in _doc.items():
            if key in self.fields and not self.fields[key].isValid(val):
                # parser knows how to validate and it's not looking good
                doc['parserLog'].append(self._log("WARNING", "Removing invalid field '{0}' for key '{1}'".format(val, key)))
            else:
                # pass on not validable fields or valid fields
                doc.update({key: val})
        
        # add pool manager
        if self.urlPoolMan: doc['__urlPoolMan'] = self.urlPoolMan
        
        orderedKeys, msg = self._configureFields(doc)
        doc['parserLog'].append(msg)
        
        shouldHaveFields = list(doc.keys())
        
        unexpectedError = None
        if not re.match("^ERROR", msg):
            for f in [ self.fields[k] for k in orderedKeys if k in self.fields and k not in doc.keys() ]:
                # print(f.name, doc.keys())
                # if f.name == 'metas': raise RuntimeError
                shouldHaveFields.append(f.name)
                missing = [ k for k in f.requiredKeys() if k not in doc.keys() ]
                if missing:
                    if f.default is not None: doc[f.name] = f.default
                    doc['parserLog'].append(self._log("WARNING", "Missing {0}, required by '{1}'".format(missing, f.name)))
                else:
                    try:
                        doc, logLevel, msg = f.add(doc)
                        #print(msg)
                        doc['parserLog'].append(self._log(logLevel, msg))
                        
                    
                    except Exception as e:
                        err = sys.exc_info().__str__()
                        msg = "Parser for URL '{0}' and field '{1}' failed with error: {2}".format(doc.get('URL', None), f.name, err)
                        log.debug(msg)
                        doc['parserLog'].append(self._log("ERROR", msg))
                        unexpectedError = e
                        break
    
        doc = ErrorCode(shouldHaveFields=self.requiredKeys, otherError=unexpectedError).update(doc)
        doc['parserLog'].append(self._log("OK", "Added errorCode"))
        return(doc)
    
    def strip(self, doc, stripRgx='^$', stripPrivate=True):
        strippenda = [ k for k in doc.keys() if re.match(stripRgx, k) ]
        if stripPrivate: strippenda = strippenda + [ k for k in doc.keys() if k.startswith('__') ]
        return(dict([ (k, v) for k, v in doc.items() if k not in strippenda ]))
    
    def isValid(self, doc, restrictValidationToFields=None):
        if not restrictValidationToFields: restrictValidationToFields = self.requiredKeys
        errorOK = ErrorCode(restrictValidationToFields).containsValid(doc)
        otherFieldsOK = all([ self.fields[k].containsValid(doc) for k in restrictValidationToFields if k in self.fields ])
        return(errorOK and otherFieldsOK)

# all the fields ###########################
class URL(Field):
    def __init__(self):
        Field.__init__(self, 'URL')
 
    def _add0(self, feedburner_origlink):
        return(self._rw(re.sub('\?ncid=rss$', '', feedburner_origlink)))

    def _add1(self, link):
        return(self._rw(link))
    
    def isValid(self, what):
        return(type(what) is str and re.match('^https?://|^www.', what) is not None)

class RefURL(Field):
    def __init__(self):
        Field.__init__(self, 'refURL')
    
    def _add0(self, title_detail):
        return(self._rw(title_detail.base))
        
    def isValid(self, what):
        return(URL().isValid(what))

class SkinnyURL(Field):
    def __init__(self):
        Field.__init__(self, 'skinnyURL')
    
    def _add0(self, URL):
        return(self._rw(stripURL(URL)))
        
    def isValid(self, what):
        return(type(what) is str and len(what) > 0)

class Domain(Field):
    def __init__(self):
        Field.__init__(self, 'domain')
        self.domains = domains
    
    def _add0(self, URL):
        checkThis = re.sub('[^A-Za-z0-9_.~\-/:].*', '', URL)
        for dom in self.domains:
            if dom in checkThis:
                return(self._rw(dom))

    def isValid(self, what):
        return(type(what) is str and len(what) > 0 and what != 'unknown')

class ContentLength(Field):
    def __init__(self):
        Field.__init__(self, 'contentLength', default=0)
    
    def _add0(self, sentences):
        return(self._rw(sum([ len(s) for s in sentences ])))
        
    def isValid(self, what):
        return(type(what) in [ int, float ] and what >= 0)

class ParsedTs(Field):
    def __init__(self):
        Field.__init__(self, 'parsedTs')
    
    def _add0(self, parsedTs=[]):
        return(self._rw(parsedTs + [time.time()]))
    
    def isValid(self, what):
        ts = CreatedTs()
        if type(what) is list:
            return(all([ ts.isValid(w) for w in what ]))
        else:
            return(False)

class CreatedTs(Field):
    def __init__(self):
        Field.__init__(self, 'createdTs')
        self._dateWords = set(' '.join([ dt.strftime(dt.strptime('%.2d' % (k+1), '%m'), '%b %B %a %A') for k in range(12) ]).lower().split())
    
    def _add0FromRSS(self, published_parsed):
        return(self._rw(time.mktime(published_parsed)))
        
    def _add1FromRSS(self, published):
        return(self._rw(recogniseDate(published).timestamp()))
    
    def _add2Online(self, __headers, __tree):
        
        # use the header
        attempts = [ recogniseDate(s) for s in __headers.values() ]
        attemptsTs = [ a.timestamp() for a in attempts if a ]
        # only consider valid if after 1990
        attemptsTs = [ ats for ats in attemptsTs if ats >= 631152000 ] 
        if len(attemptsTs):
            return((self._rw(min(attemptsTs))))
        
        # use the tree
        texts = [ fullText(x) for x in __tree.xpath('//*') ]
        texts = [ t for t in texts if len(texts) > 6 ]
        for txt in texts:
            d = recogniseDate(txt)
            if d: return(self._rw(d.timestamp()))
    
    def _add3FromUrl(self, URL):
        dateTokens = [t for t in URL.split('/') if t.lower() in self._dateWords or re.match('^[0-9]{2,4}$', t)]
        attempts = [ recogniseDate('/'.join(dateTokens)), recogniseDate(' '.join(dateTokens)) ]
        attempts = [ a for a in attempts if a ]
        
        if len(attempts) > 0:
            return(self._rw(min(attempts).timestamp()))
        
    def isValid(self, what):
        # after 1990 and before tomorrow :)
        return(type(what) in [ int, float ] and what >= 631152000 and what <= time.time() + 86400)

class Sentences(Field):
    def __init__(self):
        Field.__init__(self, 'sentences')
        self._additionalRequiredKwargs.update({'_textObj': useForSentences})
        self.splitter = splitter
    
    def _add0(self, _textObj):
        if type(_textObj) is list:
            sent = [ s for s in self.splitter(' '.join(_textObj)) ]
        elif type(_textObj) is dict:
            sent = [ s for s in self.splitter(' '.join(_textObj.values())) ]
        elif type(_textObj) is str:
            sent = [ s for s in self.splitter(_textObj) ]
        else:
            raise NotImplementedError("Don't know how to add sentences from '{1}' of type {2}".format(self._textField, type(_textObj).__name__))
        
        return(self._rw(sent))

    def isValid(self, what):
        if type(what) is list and all([ type(w) is str for w in what ]):
            return(sum([ len(w) for w in what ]) > 0)
        else:
            return(False)

class Tags(Field):
    def __init__(self):
        Field.__init__(self, 'rssTags')
    
    def _add0FromRSS(self, tags):
        return(self._rw([ t['term'] for t in tags if 'term' in t ]))
        
    def isValid(self, what):
        if type(what) is list and all([ type(w) is str for w in what ]):
            return(sum([ len(w) for w in what ]) > 0)
        else:
            return(False)

class Metas(Field):
    def __init__(self):
        Field.__init__(self, 'metas')
    
    def _add0(self, __tree):
        out = defaultdict(list)
        metas = __tree.xpath("//meta[@content]")
        for meta in metas:
            contix = [ k for k, key in enumerate(meta.keys()) if key == 'content' ][0]
            if contix <= 1: 
                out[meta.values()[1-contix]].append(meta.values()[contix])
        
        return(self._rw(dict([ (k, v[0] if len(v)==1 else v) for k, v in out.items() ])))
    
    def isValid(self, what):
        return(type(what) is dict and len(what) > 0)

class Title(Field):
    def __init__(self):
        Field.__init__(self, 'title')
    
    def _add0(self, __tree):
        return(self._rw(fullText(__tree.xpath('//title[1]')[0])))
    
    def isValid(self, what):
        return(type(what) is str and len(what) > 0)

class SampleContent(Field):
    def __init__(self):
        Field.__init__(self, '__sampleContent')
        
    def _add0(self, summary_detail):
        # all stripped sentences in the body
        stripped = [ txt.strip() for txt in etree.ElementTree(etree.HTML(summary_detail['value'])).xpath('//body')[0].itertext() ]
        # remove titles (not ending with '.') or 'read more' types (ending in '...')
        return(self._rw(' '.join([ txt for txt in stripped if re.match('.*[^\.]\.$', txt) ])))
    
    def isValid(self, what):
        return(type(what) is str and len(what) > 0)

class FatPeas(Field):
    def __init__(self):
        Field.__init__(self, '__fatPeas')
    
    def _ppt(self, __tree):
        peas = [ p for p in __tree.xpath('//p') ]
        # remove occurrence index (like /div[0]) and trailing /span
        paths = [ re.sub('(/span)+$', '', re.sub('\[[0-9]*\]|/p.*$', '', __tree.getpath(p))) for p in peas ]
        texts = [ fullText(p) for p in peas ]
        return(peas, paths, texts)
            
    def _add0WithSampleContent(self, __tree, __sampleContent):
        peas, paths, texts = self._ppt(__tree)
        allText = ' '.join(texts)
        
        minD, minK = minDistance(__sampleContent, allText)
        
        if minD < 0.5:
            totLen = 0
            for txt, path in zip(texts, paths):
                totLen += len(txt) + 1
                if totLen >= minK: break
            
            whereContent = path
            
            return(self._rw({'peas': [ (p, txt) for p, txt, path in zip(peas, texts, paths) if path == whereContent ], 'path': whereContent}))
        else:
            return(self._add1TreeOnly(__tree))
    
    def _add1TreeOnly(self, __tree):
        peas, paths, texts = self._ppt(__tree)
        
        nSpaces = [ len(t.split()) if t else 0 for t in texts ]
        # sum the nSpaces by path
        sumSpaces = defaultdict(int)
        for path, nSp in zip(paths, nSpaces): sumSpaces[path] += nSp
        
        if len(sumSpaces) > 0:
            # found a spot!
            whereContent = Counter(sumSpaces).most_common(1)[0][0]
            out = {'peas': [ (p, txt) for p, txt, path in zip(peas, texts, paths) if path == whereContent ], 'path': whereContent}        
            return(self._rw(out))
        
    def isValid(self, what):
        return(type(what) is dict and what.get('peas', None) and what.get('path', None))

class Body(Field):
    def __init__(self):
        Field.__init__(self, 'body')
    
    def _add0FromFatPeas(self, __fatPeas):
        return({'usedForBody': __fatPeas['path'], 'body': cleanText(' '.join([ p[1] for p in __fatPeas['peas'] ]))})
        
    # def _add1FromSample(self, __sampleContent):
    #     return({'usedForBody': 'sampleOnly', 'body': cleanText(__sampleContent)})
    
    def isValid(self, what):
        return(type(what) is str and len(what) > 0)

class Links(Field):
    def __init__(self):
        Field.__init__(self, 'links')
    
    def _add0(self, domain, __tree):
        U = URL()
        absoluteLinks = set(__tree.xpath("//*[starts-with(normalize-space(@href),'http')]/@href | //*[starts-with(normalize-space(@href),'www')]/@href"))
        relativeLinks = __tree.xpath("//*[not(starts-with(normalize-space(@href),'http'))]/@href")
        absoluteLinks.update([ domain + l for l in relativeLinks if l.startswith('/') and not l.startswith('www') ])
        return(self._rw([ l for l in absoluteLinks if U.isValid(l) ]))
    
    def isValid(self, what):
        U = URL()
        return(type(what) is list and any([ U.isValid(l) for l in what ]) > 0)

class AmpLink(Field):
    def __init__(self):
        Field.__init__(self, 'ampLink')
    
    def _add0(self, __tree):
        ampLink = __tree.xpath("//link[@rel='amphtml']/@href")
        if len(ampLink) > 0:
            return(self._rw(str(ampLink[0])))
    
    def isValid(self, what):
        return(URL().isValid(what))

class CanonicalLink(Field):
    def __init__(self):
        Field.__init__(self, 'canonicalLink')
    
    def _add0(self, __tree):
        canoLink = __tree.xpath("//link[@rel='canonical']/@href")
        if len(canoLink) > 0:
            return(self._rw(str(canoLink[0])))
    
    def isValid(self, what):
        return(URL().isValid(what))

class Tree(Field):
    def __init__(self):
        Field.__init__(self, '__tree')
    
    def _add0(self, URL, __urlPoolMan):
        return(self._rw(etree.ElementTree(etree.HTML(__urlPoolMan.request('GET', URL, timeout=5).data))))
    
    def isValid(self, what):
        return(type(what) is etree._ElementTree and len(what.xpath('//*/text()')) > 0)

class Headers(Field):
    def __init__(self):
        Field.__init__(self, '__headers')
    
    def _add0(self, URL, __urlPoolMan):
        return(self._rw(dict(__urlPoolMan.request('HEAD', URL, timeout=5).getheaders())))
    
    def isValid(self, what):
        return(type(what) is dict and len(what) > 0)

# special classes
# The 'add' logic would not work for ErrorCode
class ErrorCode(Field):
    def __init__(self, shouldHaveFields, otherError=None):
        Field.__init__(self, 'errorCode')
        self.shouldHaveFields = shouldHaveFields
        self.otherError = otherError
        self._knownErrorsRules = [ # req field / errorCode / isError logic, listed from more to less descriptive
            ['domain',        'unknownDomain'                                                                      ],
            ['parserLog',     'parsingError', lambda doc: any([ re.match('^ERROR', l) for l in doc['parserLog'] ]) ],
            ['sentences',     'empty'                                                                              ],
            ['contentLength', 'empty',        lambda doc: doc['contentLength'] == 0                                ],
            ]
    
    def _containsError(self, doc, field, code, isError=lambda x: False):
        if field in self.shouldHaveFields and (field not in doc or isError(doc)):
            return(code)
    
    def update(self, doc):
        
        if self.otherError:
            # contains all unexpected errors, including connection errors
            if 'urllib3' in str(self.otherError):
                doc['errorCode'] = 'cannotDownload'
            else:
                doc['errorCode'] = 'parsingError'
            return(doc)
        
        for rule in self._knownErrorsRules:
            e = self._containsError(doc, *rule)
            if e:
                doc['errorCode'] = e
                break
        
        if not self.containsValid(doc): doc['errorCode'] = 'allGood'
        
        return(doc)
    
    def isValid(self, what):
        return(what in knownErrors)
