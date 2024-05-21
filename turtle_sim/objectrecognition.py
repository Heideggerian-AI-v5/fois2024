import base64
import cv2 as cv
import math
import numpy
import os
from PIL import Image
import queue
import threading
import time
import torch
from ultralytics import YOLO
from websockets.sync.client import connect

from constants import imgWidth, imgHeight, endImg, endDepth, modelPath, turtleImgSocketURL
from dbgvis import startVisualizationThread, stopVisualizationThread, debugDataLock, debugData
from utils import makeStartStopFns

frameQueue = queue.LifoQueue()
recognitionResultsLock = threading.Lock()
imageReady = threading.Condition()
objectRecognitionResults = {"image": None, "results": None, "resultDicts": None, "depthImg": None, "depthImgPrev": None}

def type2MaskColor(s):
    h = hash(s)
    r = ((h&0xFF0000)>>16)/255.0
    g = ((h&0xFF00)>>8)/255.0
    b = (h&0xFF)/255.0
    return numpy.array((r,g,b), dtype=numpy.float32)

rxGVar = {}
recGVar = {}

def imageReception():
    with connect(turtleImgSocketURL) as websocket:
        while rxGVar.get("keepOn"):
            msg = websocket.recv()
            frameQueue.put(msg)

def getDetections(results):
    def _clip(x):
        x = numpy.array(x, dtype=numpy.int32)
        x[:,0] = numpy.clip(x[:,0],0,imgWidth-1)
        x[:,1] = numpy.clip(x[:,1],0,imgHeight-1)
        return x
    results = results[0] # For some reason YOLO returns a list but it seems only the first element matters.
    if results.masks is None:
        return []
    names = [results.names[round(x)] for x in results.boxes.cls.tolist()]
    confs = results.boxes.conf.tolist()
    boxes = results.boxes.xyxyn.tolist()
    polys = [_clip(x) for x in results.masks.xy]
    aux = {}
    for t,c,b,p in zip(names, confs, boxes, polys):
        if (2 < len(p)):
            if t not in aux:
                aux[t] = {"type": t, "confidence": [], "box": [], "polygon": [], "mask": numpy.zeros((imgHeight,imgWidth), dtype=numpy.uint8)}
            aux[t]["confidence"].append(c)
            aux[t]["box"].append(b)
            aux[t]["polygon"].append(p)
            aux[t]["mask"] += cv.fillPoly(numpy.zeros((imgHeight,imgWidth), dtype=numpy.uint8), pts = [p], color = 255)
    return [v for v in aux.values()]

def objectRecognition(dbg=False,modelFileName=None):
    if modelFileName is None:
        modelFileName=modelPath
    model = YOLO(modelFileName)
    while recGVar.get("keepOn"):
        msg = frameQueue.get()
        # clear queue
        while not frameQueue.empty():
            frameQueue.get()
            frameQueue.task_done()
        image = base64.b64decode(msg)
        # Ignore true semantic segmentation for now. We should use segmentation obtained from trained perception.
        image, depthImage, _ = image[:endImg], numpy.frombuffer(image[endImg:endDepth]).reshape((imgHeight,imgWidth)), numpy.frombuffer(image[endDepth:], dtype=numpy.uint16).reshape((imgHeight,imgWidth))
        pilImage = Image.frombytes("RGB", (imgHeight,imgWidth), image)
        st = time.perf_counter()
        results = model(pilImage, conf=0.128, verbose=False)
        print("YOLO", time.perf_counter()-st)
        retq = getDetections(results)
        if dbg:
            dbgImg = numpy.zeros((imgHeight,imgWidth,3),dtype=numpy.float32)
            print("ObjTypes seen: ", [x["type"] for x in retq])
            for e in retq:
                if ("mask" in e) and ("type" in e):
                    rgbMask = cv.cvtColor(e["mask"],cv.COLOR_GRAY2RGB).astype(numpy.float32)
                    dbgImg += rgbMask*type2MaskColor(e["type"])*(1/255.)
            debugDataLock.acquire()
            #dbgImg = cv.Laplacian(depthImage, cv.CV_32FC1, ksize=5)
            debugData["segmentation"] = dbgImg
            debugDataLock.release()
        recognitionResultsLock.acquire()
        objectRecognitionResults["resultDicts"] = retq
        objectRecognitionResults["image"] = pilImage
        objectRecognitionResults["results"] = results
        objectRecognitionResults["depthImgPrev"] = objectRecognitionResults["depthImg"]
        objectRecognitionResults["depthImg"] = depthImage
        recognitionResultsLock.release()
        with imageReady:
            imageReady.notify_all()
        frameQueue.task_done()
    with imageReady:
        imageReady.notify_all()

startReceptionThread, stopReceptionThread = makeStartStopFns(rxGVar, imageReception)
startRecognitionThread, stopRecognitionThread = makeStartStopFns(recGVar, objectRecognition)

if "__main__" == __name__:
    startVisualizationThread()
    startReceptionThread()
    startRecognitionThread(dbg=True,modelFileName=modelPath)
    input("Reception/Recognition threads running. Switch to the turtlebot GUI and move it around to see the changing segmentation masks. When you want to exit this program, press ENTER in this terminal.")
    stopRecognitionThread()
    stopReceptionThread()
    print("Threads stopped, exiting.")
    stopVisualizationThread()

