# RAW YUV Sliding direct ROI comparison
# -- bruce force matching: ie: relatively slow, but highly paralizable; simple, so maybe it could be relative fast considering DSP
# -- makes use of grayscale AND color information without RGB conversion (YUYV compatible) 
# -- Currently using hanning window to minimize background
# -- rotation or scale could be added in different ways, but with high enough frame rate this becomes unneeded
# -- custom technique vs other options?
# -- unique contribution: limit on how much a system can learn, to prevent over learning the background when static
# ... maybe take the original ROI histogram and apply it as a weighted transform to the learned image roi? long-term form of drift/background rejection?
#  .. =>  original histogram can slowly learn?

#  Simliar to Camshift in tracking quality

import numpy as np
import cv2

### GSTREAMER STUFF; not required if videocapture used instead
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst


eps = 1e-5

class STEVES:
    def __init__(self, frame, rect):
		x1, y1, x2, y2 = rect
		self.pos = x,y = int(x1), int(y1)
		self.size = w,h = int(x2-x1), int(y2-y1)
		self.multi = w*h/1.5
		self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)
		self.H = frame[y:y+h, x:x+w]*self.win

    def update(self, frame, rate = 0.1):
		#global out
		(x, y), (w, h) = self.pos, self.size
		
		bestx=dx=0
		besty=dy=0
		delta = cv2.absdiff(self.H,frame[y+dy:y+dy+h, x+dx:x+dx+w]*self.win)
		best = delta.sum()
		
		###############
		#	if best<self.multi*3: # there is nothing much to learn, so maybe we could just skip to the next frame to speed things up when in this case?
		#		return best	
		###############
		
			#############  UBER X ONLY //4 pixel region search
		dx=(bestx-4)
		delta = cv2.absdiff(self.H,frame[y+dy:y+dy+h, x+dx:x+dx+w]*self.win)
		delta = delta.sum()
		if (delta<best):
			best=delta
			bestx=dx
		
		dx=(bestx+4)
		delta = cv2.absdiff(self.H,frame[y+dy:y+dy+h, x+dx:x+dx+w]*self.win)
		delta = delta.sum()
		if (delta<best):
			best=delta
			bestx=dx
		dx=bestx
		############# UBER Y ONLY
		dy=(besty-4)
		delta = cv2.absdiff(self.H,frame[y+dy:y+dy+h, x+dx:x+dx+w]*self.win)
		delta = delta.sum()
		if (delta<best):
			best=delta
			besty=dy	
		
		dy=(besty+4)
		delta = cv2.absdiff(self.H,frame[y+dy:y+dy+h, x+dx:x+dx+w]*self.win)
		delta = delta.sum()
		if (delta<best):
			best=delta
			besty=dy
		dy=besty
		#############  UBER X ONLY //4 pixel region search
		dx=(bestx-2)
		delta = cv2.absdiff(self.H,frame[y+dy:y+dy+h, x+dx:x+dx+w]*self.win)
		delta = delta.sum()
		if (delta<best):
			best=delta
			bestx=dx
		
		dx=(bestx+2)
		delta = cv2.absdiff(self.H,frame[y+dy:y+dy+h, x+dx:x+dx+w]*self.win)
		delta = delta.sum()
		if (delta<best):
			best=delta
			bestx=dx
		dx=bestx
		############# UBER Y ONLY
		dy=(besty-2)
		delta = cv2.absdiff(self.H,frame[y+dy:y+dy+h, x+dx:x+dx+w]*self.win)
		delta = delta.sum()
		if (delta<best):
			best=delta
			besty=dy	
		
		dy=(besty+2)
		delta = cv2.absdiff(self.H,frame[y+dy:y+dy+h, x+dx:x+dx+w]*self.win)
		delta = delta.sum()
		if (delta<best):
			best=delta
			besty=dy
		dy=besty
		#############  XUBER X ONLY // 2-pixel shift search
		dx=(bestx-1)
		delta = cv2.absdiff(self.H,frame[y+dy:y+dy+h, x+dx:x+dx+w]*self.win)
		delta = delta.sum()
		if (delta<best):
			best=delta
			bestx=dx
		
		dx=(bestx+1)
		delta = cv2.absdiff(self.H,frame[y+dy:y+dy+h, x+dx:x+dx+w]*self.win)
		delta = delta.sum()
		if (delta<best):
			best=delta
			bestx=dx
		dx=bestx
		############# XUBER Y ONLY
		dy=(besty-1)
		delta = cv2.absdiff(self.H,frame[y+dy:y+dy+h, x+dx:x+dx+w]*self.win)
		delta = delta.sum()
		if (delta<best):
			best=delta
			besty=dy	
		
		dy=(besty+1)
		delta = cv2.absdiff(self.H,frame[y+dy:y+dy+h, x+dx:x+dx+w]*self.win)
		delta = delta.sum()
		if (delta<best):
			best=delta
			besty=dy	
		dy=besty
		#############

		## TODO:  IF NOT FOUND, continue scanning all of image for it, outwards from center?
		
		###  KEEP learned images saved every few seconds, stack them, and find a way to use them later.
		## create a gstreamer pipeline to upload videos that way. will auto create thumbnails this way, just repackage? 
		
		## Add guasian blur to absdiff to see if that helps with noise reduction and mild deformation changes. better chance of matching without introducing much error.  slower?
		
		if best>self.multi*2: ##  Higher PSR than this and it probably isn't a good match
			x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
			cv2.rectangle(frame, (x1+2, y1+2), (x2-2, y2-2), (255))
			cv2.rectangle(frame, (x1, y1), (x2, y2), (255))
			cv2.putText(frame, "PSR: "+str(int(best/100)), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255))
			cv2.imshow("vis",frame*2)
			cv2.imshow("H",self.H/20)
			#out.write(frame)
			return best  ## failed to track
		
	
		if best>self.multi*1:  ##  Don't over learn, else we might start to learn the background and that will get stick. 
			self.H = self.H * (1.0-rate) + frame[y+besty:y+besty+h, x+bestx:x+bestx+w]*self.win * rate  ## learn if needed only
			
		self.pos = x,y = (int(x+bestx), int(y+besty))
		cv2.imshow("H",self.H/20)
		x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
		cv2.rectangle(frame, (x1, y1), (x2, y2), (255))
		
		cv2.putText(frame,  "PSR: "+str(int(best/100)), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, .3, (255))
		cv2.imshow("vis",frame*2)
		#out.write(frame)
		return best




### Stuff below isn't very important;  just gstreamer stuff really.

class SES():	
	trackers = []

global ses
ses = SES()
ses.counter=10
#global out
#out = cv2.VideoWriter('output.avi',0, 24.0,(640*2,480))


def on_new_buffer(appsink):
	global ses
	#global out
	w=640
	h=480
	size=w*h*2
	sample = appsink2.emit('pull-sample')
	buf=sample.get_buffer()
	data=buf.extract_dup(0,buf.get_size())
	stream=np.fromstring(data,np.uint8) #convert data form string to numpy array // THIS IS YUV , not BGR.  improvement here if changed on gstreamer side
	image=np.array(stream[0:size:2].reshape(h,w)) # create the y channel same size as the image
	cv2.imshow("main",image)
	#blur = cv2.GaussianBlur(image[:,:],(11,11),0)
	dct = cv2.dct(np.float32(image)/255)
	blur = cv2.GaussianBlur(image,(15,15),0)
	dct2 = cv2.dct(np.float32(blur)/255)
	idct= cv2.idct(dct[0:int(h/4),0:int(w/4)]-dct2[0:int(h/4),0:int(w/4)])
	
	ses.frame = np.uint8(np.clip(idct/idct.max(), 0, 1)*255)
	ses.frame = cv2.GaussianBlur(ses.frame,(15,15),0)*2
	## 
	if ses.counter>0:
		rect = 30, 25, 60, 65  ####### Manually selecting ROI.  Width needs to be 2x due to the raw YUYV structure of the image
		if ses.counter==1:
			tracker = STEVES(ses.frame, rect)
			ses.trackers.append(tracker)
		ses.counter-=1
	
	#if ses.counter<-600:
	#	out.release()
    
	for tracker in ses.trackers:
		tracker.update(ses.frame)

	cv2.waitKey(1)
	return False
				
if __name__ == '__main__':  ## This code is to pipe in video from either the Tracer or from the Intel Realsense Camera
	Gst.init(None)
	CLI2="ksvideosrc device-index=2 ! video/x-raw,format=YUY2,width=640,height=480,framerate=60/1 ! appsink name=sink" ## 60 FPS
	pipline2=Gst.parse_launch(CLI2)
	appsink2=pipline2.get_by_name("sink")
	appsink2.set_property("max-buffers",20) # prevent the app to consume huge part of memory
	appsink2.set_property('emit-signals',True) #tell sink to emit signals
	appsink2.set_property('sync',False) #no sync to make decoding as fast as possible
	appsink2.set_property('drop', True) ##  qos also
	appsink2.set_property('async', True)
	appsink2.connect('new-sample', on_new_buffer) #connect signal to callable func
	pipline2.set_state(Gst.State.PLAYING)
	bus2 = pipline2.get_bus();
	msg2 = bus2.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)

	