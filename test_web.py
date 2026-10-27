import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import numpy as np
import cv2
from fastapi.testclient import TestClient
from app import Engine, create_app

class Boxes:
    data=np.array([[100,100,200,400,.9,0]],dtype=float)
    cls=data[:,5]
    def cpu(self):return self
    def numpy(self):return self
    def __getitem__(self,k):return self
class Model:
    names={0:'person',9:'gloves',10:'helmet',16:'safety-vest'}
    def __init__(self,*a):pass
    def predict(self,*a,**k):return [types.SimpleNamespace(boxes=Boxes())]
class Tracker:
    def __init__(self,*a,**k):pass
    def update(self,*a):return np.array([[100,100,200,400,1,.9,0,0]])
class WebTests(unittest.TestCase):
    def test_api(self):
        with tempfile.TemporaryDirectory() as d:
            engine=Engine('missing.pt','missing.mp4',output=d)
            with TestClient(create_app(engine)) as c:
                self.assertEqual(c.get('/').status_code,200)
                self.assertEqual(c.get('/api/status').json()['status']['state'],'idle')
                self.assertEqual(c.post('/api/control',json={'action':'resume'}).status_code,409)
                self.assertEqual(c.post('/api/control',json={'action':'bad'}).status_code,409)
                self.assertEqual(c.post('/api/control',json={'action':'stop'},headers={'origin':'http://other'}).status_code,403)
                self.assertEqual(c.get('/captures/..%2Fapp.py').status_code,404)
                with c.websocket_connect('/ws') as ws:
                    self.assertEqual(ws.receive_json()['status']['state'],'idle')
    def test_video(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); video=root/'clip.avi'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),25,(640,480))
            self.assertTrue(writer.isOpened())
            for _ in range(65):writer.write(np.zeros((480,640,3),dtype=np.uint8))
            writer.release()
            zone=root/'zone.json';zone.write_text(json.dumps({'version':1,'source':str(video.resolve()),'frame_size':[640,480],'points':[[.01,.01],[.99,.01],[.99,.99],[.01,.99]]}))
            engine=Engine('fake.pt',str(video),zone=zone,output=root/'captures')
            fake={'ultralytics':types.SimpleNamespace(YOLO=Model),'ultralytics.trackers':types.ModuleType('ultralytics.trackers'),'ultralytics.trackers.byte_tracker':types.SimpleNamespace(BYTETracker=Tracker)}
            with patch.dict(sys.modules,fake):
                engine.start();deadline=time.monotonic()+5
                while engine.frame_id<3 and time.monotonic()<deadline:time.sleep(.02)
                engine.control('pause');time.sleep(.15)
                count=engine.frame_id;time.sleep(.15);self.assertEqual(engine.frame_id,count)
                with self.assertRaises(ValueError):engine.start()
                engine.control('resume');engine.thread.join(8)
                snap=engine.snapshot();self.assertEqual(snap['status']['state'],'ended',snap)
                kinds={e['type'] for e in snap['events']}
                self.assertIn('danger_zone',kinds);self.assertIn('no_helmet',kinds)
                self.assertTrue(engine.jpeg.startswith(b'\xff\xd8'))
                self.assertTrue(list((root/'captures').rglob('*.jpg')))
                self.assertTrue(list((root/'captures').rglob('events.jsonl')))
                first=engine.status['session'];engine.start();engine.control('stop');engine.thread.join(5)
                self.assertNotEqual(engine.status['session'],first)
                self.assertEqual(engine.status['state'],'stopped')
if __name__=='__main__':unittest.main()
