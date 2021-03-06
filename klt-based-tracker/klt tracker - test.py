#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
Usage
-----
lk_track.py [<video_source>]
Keys
----
ESC - exit
'''

import numpy as np
import cv2
from common import anorm2, draw_str
from time import clock
import math

# some constants and default parameters

subpix_params = dict(zeroZone=(-1,-1),winSize=(10,10),
                     criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,20,0.03))

lk_params = dict( winSize  = (15, 15),  
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

				  
				  
feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

#feature_params = dict(maxCorners=1000,qualityLevel=0.02,minDistance=3)  # SLOWER
					   
class App:
    def __init__(self, video_src):
		self.track_len = 10
		self.detect_interval = 5
		self.tracks = []
		self.xx = 320
		self.yy = 240
		self.roi = [111, 49, 55, 155]
		# ----------------------------------
		self.cam = cv2.VideoCapture(video_src)
		#self.cam.set(3,1280)
		#self.cam.set(4,720)
		self.frame_idx = 0

    def run(self):
        while True:
            ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
				img0, img1 = self.prev_gray, frame_gray
				p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
				p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
				p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
				d = abs(p0-p0r).reshape(-1, 2).max(-1)
				if (len(p1)>2):
					transform = cv2.estimateRigidTransform(p1, p0r, False)
					if transform is not None:
						dx = transform[0][2]
						dy = transform[1][2]
						scaleX = math.sqrt((transform[0][0]*transform[0][0]) + (transform[0][1]*transform[0][1]));
						scaleY = math.sqrt((transform[1][0]*transform[1][0]) + (transform[1][1]*transform[1][1]));
						self.roi = [self.roi[0]-dx, self.roi[1]-dy, self.roi[2], self.roi[3]]
						
				good = d < 1
				new_tracks = []
				for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
					if not good_flag:
						continue
					tr.append((x, y))
					if len(tr) > self.track_len:
						del tr[0]
					new_tracks.append(tr)
					cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
				self.tracks = new_tracks
				cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
				draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
			
				mask = np.zeros_like(frame_gray)
				mask[round(max(0,self.roi[1])):round(min(self.yy,self.roi[1] + self.roi[3])),round(max(0,self.roi[0])):round(min(self.xx,self.roi[0] + self.roi[2]))] = 255  # create mask based on roi

				#mask[:] = 255
				for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
					cv2.circle(mask, (x, y), 5, 0, -1)
				p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
				p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
				p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
				#cv2.cornerSubPix(frame_gray,p, **subpix_params)  ##  OPTIONAL
				if p is not None:
					for x, y in np.float32(p).reshape(-1, 2):
						self.tracks.append([(x, y)])
            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.rectangle(vis, (int(self.roi[0]), int(self.roi[1])),  (int(self.roi[0] + self.roi[2]), int(self.roi[1] + self.roi[3])), color=(255, 0, 0))
            cv2.imshow('lk_track', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print __doc__
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()