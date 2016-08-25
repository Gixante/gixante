import sys

sys.path.append('/home/bean/Code/gixante/gixante/utils')
from api import get, put, cfg, log

#from gixante.utils.api import get, put, cfg, log
from gixante.utils.arango import cleanupTests, database

if len(sys.argv) < 2: sys.argv.append("news")
if len(sys.argv) < 3: sys.argv.append("adlite")
if len(sys.argv) < 4: sys.argv.append("20")

#apiRoot = 'http://{0}:{1}'.format(cfg[ sys.argv[1] + 'ApiIP' ], cfg[ sys.argv[1] + 'ApiPort' ])
apiRoot = 'http://localhost:5000'
site = sys.argv[2]
log.setLevel(int(sys.argv[3]))

log.info("Testing resource 'Statistics'...")
testStatsGet = get(apiRoot + '/getCollStats')
testStatsPut = put(apiRoot + '/addSiteStat/site={0}'.format(site), data={'dummy': 'this is a test', '_flag': 'test'})

log.info("Testing resource 'SimilToText'...")
testSimilPost = put(apiRoot + '/post', data={'text': 'this is a simple test', '_flag': 'test'})
testSimilPut = put(apiRoot + '/put/csvWords=this,is,a,simple,test')
database.col(testSimilPut['_id'].split('/')[0]).update_document(testSimilPut['_id'].split('/')[1], {'_flag': 'test'})

testSimilGet = get(apiRoot + '/get/id={0}/fields=queryInProgress,nDocs/minNumDocs=25/nMoreDocs=1000/docFields=URL'.format(testSimilPut['_key']))
testSimilGetStatus = get(apiRoot + '/get/id={0}/fields=queryInProgress,nDocs'.format(testSimilPut['_key']))
while testSimilGetStatus['queryInProgress']:
    testSimilGetStatus = get(apiRoot + '/get/id={0}/fields=queryInProgress,nDocs'.format(testSimilPut['_key']))
    print("Fetched so far: {0}".format(testSimilGetStatus['nDocs']), end="\r")

log.info("Testing resource 'Semantic'...")
testSema1 = get(apiRoot + '/semantic/id={0}/nEachSide=10/minNumDocs=100/rankPctDocs=0.5/semaSearch=positive'.format(testSimilPut['_key']))
testSema2 = get(apiRoot + '/semantic/id={0}/nEachSide=10/semaSearch=positive'.format(testSimilPost['_key']))
testSema3 = get(apiRoot + '/semantic/id={0}/nEachSide=10/minNumDocs=100/rankPctDocs=0.5'.format(testSimilPut['_key']), data={'semaSearch': 'cool'})
testSema4 = get(apiRoot + '/semantic/id={0}/nEachSide=10'.format(testSimilPost['_key']), data={'semaSearch': 'cool'})

log.info("Cleaning up the database...")
cleanupTests([ testStatsPut['_id'].split('/')[0], testSimilPut['_id'].split('/')[0] ])
