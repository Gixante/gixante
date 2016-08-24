from flask import Flask, request, url_for, render_template
from requests import put, get

apiRoot = 'http://localhost:5000'

### test the API
testStat = get(apiRoot + '/getCollStats').json()
testStatAdd = put(apiRoot + '/addSiteStat/site=adlite', data={'dummyField': 'dummyValue'}).json()
testPut = put(apiRoot + '/put/csvWords=this,is,a,dummy,test').json()
testGet = get(apiRoot + '/get/id={0}/fields=text,queryInProgress,nDocs,_key/minNumDocs={1}/nMoreDocs={2}/docFields=URL,title'.format(testPut['_key'], 10, 25)).json()
testGetFast = get(apiRoot + '/get/id={0}/fields=queryInProgress,text'.format(testPut['_key'])).json()
testSema = get(apiRoot + '/semantic/id={0}/nEachSide={1}/minNumDocs={2}/rankPctDocs={3}'.format(testPut['_key'], 10, 25, 0.5), data={'semaSearch': 'cool'}).json()
queryId = testPut['_key']
###

app = Flask(__name__)
app.config.from_object(__name__)

nGet = 1000
nReturnDocs = 25
nSemantic = 10

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

@app.route('/about')
def about():
    nDocs = get(apiRoot + '/getCollStats').json()['alive']['count']
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
    res = put(apiRoot + '/addSiteStat/site=adlite', data=request.form).json()
    return(render_template('thankyou.html'))

# AD DEMO
@app.route('/ad_demo')
def ad_demo(error=None):
    nDocs = "{:,}".format(get(apiRoot + '/getCollStats').json()['alive']['count'])
    return render_template('ad_demo.html', nDocs=nDocs, error=error)

@app.route('/ad_initial', methods=['POST', 'GET'])
def ad_initial():
    res = put(apiRoot + '/post', data={'text': request.form['paragraph']}).json()
    if 'error' in res:
        return(ad_demo(error=res['error']))
    else:
        return(ad_fetch(res['_key']))

@app.route('/ad_fetch/<queryId>')
@app.route('/ad_fetch/<queryId>/<int:nMoreDocs>')
def ad_fetch(queryId, nMoreDocs=nGet):
    out = get(apiRoot + '/get/id={0}/fields=text,queryInProgress,nDocs,_key/minNumDocs={1}/nMoreDocs={2}/docFields=URL,title'.format(queryId, nReturnDocs, nMoreDocs)).json()
    out['docs'] = out['docs'][:nReturnDocs]
    print(out['nDocs'])
    return(render_template('ad_results.html', **out))
    
@app.route('/ad_semantic_intro/<queryId>', methods=['POST', 'GET'])
def ad_semantic_intro(queryId):
    out = get(apiRoot + '/get/id={0}/fields=text,_key,nDocs'.format(queryId)).json()
    return(render_template('ad_semantic_intro.html', **out))
    
@app.route('/ad_semantic/<queryId>', methods=['POST', 'GET'])
def ad_semantic(queryId):
    # rank some docs semantically
    out = get(apiRoot + '/semantic/id={0}/nEachSide={1}/minNumDocs={2}/rankPctDocs={3}'.format(queryId, nSemantic, nGet, 0.5), data=request.form).json()
    # add some info
    out.update(get(apiRoot + '/get/id={0}/fields=queryInProgress,text,_key,nDocs'.format(queryId)).json())
    out.update(request.form)
    return(render_template('ad_semantic.html', **out))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
