import sys, time
from flask import Flask, request, url_for, render_template
from gixante.utils.api import get, put, cfg, log, checkHeartbeat, HeartbeatError
from threading import Thread

runDebug = sys.argv[-1].lower() == 'debug'

#if runDebug:
#apiRoot = 'http://localhost:5000'
#log.setLevel(0)
#else:
apiRoot = 'http://{0}:{1}'.format(cfg[ 'newsApiIP' ], cfg[ 'newsApiPort' ])
log.setLevel(0)

nGet = 1000
nReturnDocs = 25
nSemantic = 10

app = Flask(__name__)
app.config.from_object(__name__)

lastHB = 'Heartbeat monitor not initialised'

def hbCheck():
    return(lastHB)

def hbBackground():
    global lastHB
    while True:
        try:
            checkHeartbeat(apiRoot + '/heartbeat')
            lastHB = 'OK'
        except HeartbeatError as hbe:
            lastHB = hbe.__str__()
        except Exception as e:
            log.debug(sys.exc_info().__str__())
            lastHB = "An internal error has occurred"
        
        time.sleep(10)

app.jinja_env.globals.update(APIstatus=hbCheck)
hbThread = Thread(target=hbBackground, name='hbBackground', daemon=True)
hbThread.start()

# ROUTING - BASIC PAGES
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/publishers')
def publishers():
    return render_template('publishers.html')

@app.route('/advertisers')
def advertisers():
    return render_template('advertisers.html')

@app.route('/about')
def about():
    out = get(apiRoot + '/getCollStats')
    if 'alive' in out:
        millDocs = "more than {:,} million".format((out['alive']['count'] - out['alive']['count'] % 1e5) / 1e6)
    else:
        millDocs = 'millions of'
    return render_template('about.html', millDocs=millDocs)
    
@app.route('/why')
def why():
    return render_template('why.html')

@app.route('/contact')
@app.route('/contact/<message>')
def contact(message=''):
    return(render_template('contact.html', message=message))

@app.route('/add_contact', methods=['POST', 'GET'])
def add_contact():
    data = dict([ (k, v) for k, v in request.form.items() ])
    if runDebug: data.update({'_flag': 'test'})
    res = put(apiRoot + '/addSiteStat/site=adlite', data=data)
    return(render_template('thankyou.html'))

# ROUTING - AD DEMO
@app.route('/ad_demo')
def ad_demo(NUerror=None):
    out = get(apiRoot + '/getCollStats')
    out['NUerror'] = NUerror
    if 'alive' in out: out['nDocs'] = "{:,}".format(out['alive']['count'])
    return render_template('ad_demo.html', **out)

@app.route('/ad_initial', methods=['POST', 'GET'])
def ad_initial():
    data = dict([ (k, v) for k, v in request.form.items() ])
    if runDebug: data.update({'_flag': 'test'})
    res = put(apiRoot + '/post', data=data)
    if 'NUerror' in res:
        return(ad_demo(NUerror=res['NUerror']))
    else:
        return(ad_fetch(res['_key']))

@app.route('/ad_fetch/<queryId>')
@app.route('/ad_fetch/<queryId>/<int:nMoreDocs>')
def ad_fetch(queryId, nMoreDocs=nGet):
    out = get(apiRoot + '/get/id={0}/fields=text,queryInProgress,nDocs,_key/minNumDocs={1}/nMoreDocs={2}/docFields=URL,title'.format(queryId, nReturnDocs, nMoreDocs))
    if 'docs' in out: out['docs'] = out['docs'][:nReturnDocs]
    return(render_template('ad_results.html', **out))
    
@app.route('/ad_semantic_intro/<queryId>', methods=['POST', 'GET'])
def ad_semantic_intro(queryId):
    out = get(apiRoot + '/get/id={0}/fields=text,_key,nDocs'.format(queryId))
    return(render_template('ad_semantic_intro.html', **out))
    
@app.route('/ad_semantic/<queryId>', methods=['POST', 'GET'])
def ad_semantic(queryId):
    # rank some docs semantically
    out = get(apiRoot + '/semantic/id={0}/nEachSide={1}/minNumDocs={2}/rankPctDocs={3}'.format(queryId, nSemantic, nGet, 0.5), data=request.form)
    # add some info
    out.update(get(apiRoot + '/get/id={0}/fields=queryInProgress,text,_key,nDocs'.format(queryId)))
    out.update(request.form)
    return(render_template('ad_semantic.html', **out))

@app.route('/api_error/<int:code>', methods=['POST', 'GET'])
@app.route('/api_error/<int:code>/<message>', methods=['POST', 'GET'])
def api_error(code, message=None):
    log.error("API error {0} encountered on a live page: {1}".format(code, message))
    
    data = {'message': message, 'code': code}
    if runDebug:
        data.update({'_flag': 'test'})
    else:
        data.update({'_flag': 'liveError'})
    
    res = put(apiRoot + '/addSiteStat/site=adlite', data=data)
    return(render_template('api_error.html'))

log.info("Ready for business!")

#if runDebug:
#    if __name__ == '__main__':
#        app.run(host='0.0.0.0', port=(5000 + runDebug), debug=runDebug)
#    
#    log.info("Goodbye!") 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
