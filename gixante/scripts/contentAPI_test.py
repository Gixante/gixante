import sys

from gixante.utils.api import get, put, cfg, log
from gixante.utils.arango import cleanupTests, database

if len(sys.argv) < 2: sys.argv.append("news")
if len(sys.argv) < 3: sys.argv.append("adlite")
if len(sys.argv) < 4: sys.argv.append("20") # INFO logging level

#apiRoot = 'http://{0}:{1}'.format(cfg[ sys.argv[1] + 'ApiIP' ], cfg[ sys.argv[1] + 'ApiPort' ])
apiRoot = 'http://localhost:9999'
site = sys.argv[2]
log.setLevel(int(sys.argv[3]))

log.info("Testing resource 'Statistics'...")
testStatsGet = get(apiRoot + '/getCollStats')
testStatsPut = put(apiRoot + '/addSiteStat/site={0}'.format(site), data={'dummy': 'this is a test', '_flag': 'test'})

log.info("Testing resource 'SimilToText'...")
testSimilPost = put(apiRoot + '/post', data={'text': 'this is a simple test', '_flag': 'test'})
database.col(testSimilPost['_id'].split('/')[0]).update_document(testSimilPost['_id'].split('/')[1], {'_flag': 'test'})

testSimilGet = get(apiRoot + '/get/id={0}/fields=queryInProgress,nDocs/minNumDocs=25/nMoreDocs=1000/docFields=URL'.format(testSimilPost['_key']))
testSimilGetStatus = get(apiRoot + '/get/id={0}/fields=queryInProgress,nDocs'.format(testSimilPost['_key']))
while testSimilGetStatus['queryInProgress']:
    testSimilGetStatus = get(apiRoot + '/get/id={0}/fields=queryInProgress,nDocs'.format(testSimilPost['_key']))
    print("Fetched so far: {0}".format(testSimilGetStatus['nDocs']), end="\r")

log.info("Testing resource 'Semantic'...")
testSema1 = get(apiRoot + '/semantic/id={0}/nEachSide=10/minNumDocs=100/rankPctDocs=0.5/semaSearch=positive'.format(testSimilPost['_key']))
testSema2 = get(apiRoot + '/semantic/id={0}/nEachSide=10/semaSearch=positive'.format(testSimilPost['_key']))
testSema3 = get(apiRoot + '/semantic/id={0}/nEachSide=10/minNumDocs=100/rankPctDocs=0.5'.format(testSimilPost['_key']), data={'semaSearch': 'cool'})
testSema4 = get(apiRoot + '/semantic/id={0}/nEachSide=10'.format(testSimilPost['_key']), data={'semaSearch': 'cool'})

log.info("Cleaning up the database...")
cleanupTests([ d['_id'].split('/')[0] for d in [testStatsPut, testSimilPost] ])
