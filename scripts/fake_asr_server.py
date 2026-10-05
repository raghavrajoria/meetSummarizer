"""Offline HTTP ASR v2 fixture server; never included in production services."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from threading import Thread
import json
from indicmeet.contract import canonicalize

@contextmanager
def fake_server(*,mode="slow",row=None,token="test-token"):
    row=row or {"idx":0,"start":0.,"end":1.,"speaker":"SPEAKER_01","text":"source", "lang":"en","quality":"accepted"}
    result=canonicalize([row]);result[0]['asr']['model_version']='fake-asr-v2';result[0]['segment_id']='host-id-is-ignored'
    class Handler(BaseHTTPRequestHandler):
        posts=0;polls=0;auth=None;request_body=None
        def respond(self,code,body,headers=None):
            data=json.dumps(body).encode();self.send_response(code)
            for key,value in (headers or {}).items():self.send_header(key,value)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def authorized(self):
            type(self).auth=self.headers.get('Authorization')
            if type(self).auth!='Bearer '+token:self.respond(401,{'detail':'Unauthorized'});return False
            return True
        def do_POST(self):
            if not self.authorized():return
            type(self).posts+=1;type(self).request_body=self.rfile.read(int(self.headers.get('Content-Length','0')))
            if mode=='reject':return self.respond(413,{'detail':'use async'})
            if 'sync=true' in self.path:return self.respond(200,result)
            self.respond(202,{'job_id':'fake-job-1'})
        def do_GET(self):
            if self.path=='/healthz':return self.respond(200,{'status':'ok','model_version':'fake-asr-v2'})
            if not self.authorized():return
            type(self).polls+=1
            if self.path!='/jobs/fake-job-1' or mode=='unknown':return self.respond(404,{'detail':'Unknown job_id'})
            if mode=='failed':return self.respond(200,{'status':'failed','progress':25,'stage':'asr','error':{'code':'MODEL_FAILED','detail':'ASR failed'}})
            if mode=='transient' and type(self).polls<=2:return self.respond(429 if type(self).polls==1 else 503,{'detail':'busy'},{'Retry-After':'0'})
            status='running' if mode=='stall' or type(self).polls<3 else 'done'
            body={'status':status,'progress':30 if status=='running' else 100,'stage':'asr' if status=='running' else 'done'}
            if status=='done':body['result']=result
            self.respond(200,body)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield 'http://127.0.0.1:'+str(server.server_port),Handler
    finally:server.shutdown();thread.join(2);server.server_close()
