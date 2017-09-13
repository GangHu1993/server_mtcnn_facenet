# coding:utf-8
import numpy as np
import cv2
import json
import base64
from face_recon import Face_recon
#from darknet import detect_interface

detection_model_path = './haarcascade_frontalface_default.xml'
_debug = True

face_recon = Face_recon()

def sayHello():
    return 'hello, I am a server'

def detect_faces(image_name):
    img = image_name
    face_cascade = cv2.CascadeClassifier(detection_model_path)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    return result

from http.server import HTTPServer,BaseHTTPRequestHandler
class RequestHandler(BaseHTTPRequestHandler):
  def _writeheaders(self):
    self.send_response(200);
    self.send_header('Content-type','text/html');
    self.end_headers()
  def do_Head(self):
    self._writeheaders()
  def do_GET(self):
    self._writeheaders()
    self.wfile.write("""<!DOCTYPE HTML>
                    <html lang="en-US">
                    <head>
                        <title></title>
                    </head>
                    <body>
                        this is HuDou server!<br/>
                        face recongnition test!<br/>
                        send data:<br/>
                        byte[]<br/>
                        return format json<br/>
                        return example:<br/>
                        {u'person': [0.9140915870666504, 225.07601928710938, 229.7716827392578, 87.51119995117188, 264.13665771484375],<br/> 
                         u'horse': [0.9088721871376038, 506.5354919433594, 226.71127319335938, 206.71493530273438, 214.5184326171875], <br/>
                         u'dog': [0.9217720627784729, 132.75021362304688, 307.6336669921875, 127.34422302246094, 90.42282104492188]}<br/>
                    </body>
                    </html>""")
  def do_POST(self):
    self._writeheaders()
    #print self.headers
    if self.path == '/face_recon.json':
        '''body length'''
        length = self.headers['content-length']
        '''read the body  '''      
        nbytes = int(length)
        data = self.rfile.read(nbytes)

        # decodejson = json.loads(data)
        # imb = base64.b64decode(decodejson['img'])
        nparr = np.fromstring(data, np.uint8)
        img_np = cv2.imdecode(nparr,1)
        #print (img_np)

        '''show the deal time'''
        if _debug:
            t1 = cv2.getTickCount()

        #r = detect_interface(data,nbytes)
        #r = sayHello()
        #r = detect_faces(img_np)
        r = face_recon.face_rec(img_np)
        print ('##########################')
        print (r)

        if _debug:
            t2 = cv2.getTickCount()
            t = (t2-t1)/cv2.getTickFrequency()
            #print 'the server deal total time'
            #print t
        self.wfile.write(json.dumps(r).encode())

addr = ('',4321)
server = HTTPServer(addr,RequestHandler)
server.serve_forever()
