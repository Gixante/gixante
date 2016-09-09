import sys, time
from flask import Flask, request, url_for, render_template

import gixante.utils.api
api.configForCollection('news')

from gixante.utils.parsing import log

runDebug = sys.argv[-1].lower() == 'debug'

nGet = 1000
nReturnDocs = 25
nSemantic = 10

app = Flask(__name__)
app.jinja_env.globals.update(APIok=api.heartbeat.isOK)

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
    out = api.get('getCollStats')
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
    res = api.put('addSiteStat/site=adlite', data=data)
    return(render_template('thankyou.html'))

# ROUTING - AD DEMO
@app.route('/ad_demo')
def ad_demo(NUerror=None):
    out = api.get('getCollStats')
    out['NUerror'] = NUerror
    if out['APImessage'] == 'OK': out['nDocs'] = "{:,}".format(out['alive']['count'])
    return render_template('ad_demo.html', **out)

@app.route('/ad_initial', methods=['POST', 'GET'])
def ad_initial():
    data = dict([ (k, v) for k, v in request.form.items() ])
    if runDebug: data.update({'_flag': 'test'})
    res = api.put('post', data=data)
    if 'NUerror' in res:
        return(ad_demo(NUerror=res['NUerror']))
    else:
        return(ad_fetch(res['_key']))

@app.route('/ad_fetch/<queryId>')
@app.route('/ad_fetch/<queryId>/<int:nMoreDocs>')
def ad_fetch(queryId, nMoreDocs=nGet):
    out = api.get('get/id={0}/fields=text,queryInProgress,nDocs,_key/minNumDocs={1}/nMoreDocs={2}/docFields=URL,title'.format(queryId, nReturnDocs, nMoreDocs))
    if 'docs' in out: out['docs'] = out['docs'][:nReturnDocs]
    return(render_template('ad_results.html', **out))
    
@app.route('/ad_semantic_intro/<queryId>', methods=['POST', 'GET'])
def ad_semantic_intro(queryId):
    out = api.get('get/id={0}/fields=text,_key,nDocs'.format(queryId))
    return(render_template('ad_semantic_intro.html', **out))
    
@app.route('/ad_semantic/<queryId>', methods=['POST', 'GET'])
def ad_semantic(queryId):
    # rank some docs semantically
    out = api.get('semantic/id={0}/nEachSide={1}/minNumDocs={2}/rankPctDocs={3}'.format(queryId, nSemantic, nGet, 0.5), data=request.form)
    # add some info
    out.update(api.get('get/id={0}/fields=queryInProgress,text,_key,nDocs'.format(queryId)))
    out.update(request.form)
    return(render_template('ad_semantic.html', **out))

@app.route('/api_error', methods=['POST', 'GET'])
@app.route('/api_error/<int:code>', methods=['POST', 'GET'])
@app.route('/api_error/<int:code>/<message>', methods=['POST', 'GET'])
def api_error(code=None, message=None):
    log.error("API error {0} encountered on a live page: {1}".format(code, message))
    
    data = {'message': message, 'code': code}
    if runDebug:
        data.update({'_flag': 'test'})
    else:
        data.update({'_flag': 'liveError'})
    
    res = api.put('addSiteStat/site=adlite', data=data)
    return(render_template('api_error.html'))

log.info("Ready for business!")

if runDebug:
   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=(5000 + runDebug), debug=runDebug)
   
   log.info("Goodbye!")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
