import sys
from requests import put, get
from gixante.utils.arango import database, log

if len(sys.argv) < 2: sys.argv.append("localhost")
if len(sys.argv) < 3: sys.argv.append("5000")
if len(sys.argv) < 4: sys.argv.append("adlite")

apiRoot = 'http://{0}:{1}/'.format(sys.argv[1], sys.argv[2])
site = sys.argv[3]
cleanupQ = "FOR doc IN {0} FILTER doc.flag == 'test' REMOVE doc IN {0}"

log.info("Testing resource 'Statistics'...")
testStatsGet = get(apiRoot + 'getCollStats').json()
testStatsPut = put(apiRoot + 'addSiteStat/site={0}'.format(site), data={'dummy': 'this is a test', 'flag': 'test'}).json()

log.info("Testing resource 'SimilToText'...")
testSimilPost = put(apiRoot + 'post', data={'text': 'this is a simple test', 'flag': 'test'}).json()
testSimilPut = put(apiRoot + 'put/csvWords=this,is,a,simple,test').json()
database.col(testSimilPut['_id'].split('/')[0]).update_document(testSimilPut['_id'].split('/')[1], {'flag': 'test'})

testSimilGet = get(apiRoot + 'get/id={0}/fields=queryInProgress,nDocs/minNumDocs=25/nMoreDocs=1000/docFields=URL'.format(testSimilPut['_key'])).json()
testSimilGetStatus = get(apiRoot + 'get/id={0}/fields=queryInProgress,nDocs'.format(testSimilPut['_key'])).json()
while testSimilGetStatus['queryInProgress']:
    testSimilGetStatus = get(apiRoot + 'get/id={0}/fields=queryInProgress,nDocs'.format(testSimilPut['_key'])).json()
    print("Fetched so far: {0}".format(testSimilGetStatus['nDocs']), end="\r")

log.info("Testing resource 'Semantic'...")
testSema1 = get(apiRoot + 'semantic/id={0}/nEachSide=10/minNumDocs=100/rankPctDocs=0.5/semaSearch=positive'.format(testSimilPut['_key'])).json()
testSema2 = get(apiRoot + 'semantic/id={0}/nEachSide=10/semaSearch=positive'.format(testSimilPost['_key'])).json()
testSema3 = get(apiRoot + 'semantic/id={0}/nEachSide=10/minNumDocs=100/rankPctDocs=0.5'.format(testSimilPut['_key']), data={'semaSearch': 'cool'}).json()
testSema4 = get(apiRoot + 'semantic/id={0}/nEachSide=10'.format(testSimilPost['_key']), data={'semaSearch': 'cool'}).json()

log.info("Cleaning up the database...")
cleanup = database.col(testStatsPut['_id'].split('/')[0]).remove_by_keys([testStatsPut['_id'].split('/')[1]])
cleanup = database.col(testSimilPut['_id'].split('/')[0]).remove_by_keys([testSimilPost['_id'].split('/')[1], testSimilPut['_id'].split('/')[1]])

database.execute_query(cleanupQ.format(testStatsPut['_id'].split('/')[0]))
database.execute_query(cleanupQ.format(testSimilPut['_id'].split('/')[0]))
