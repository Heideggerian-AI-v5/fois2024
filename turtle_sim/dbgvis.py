import cv2 as cv
import threading

from utils import makeStartStopFns

dbgGVar = {}

debugDataLock = threading.Lock()
debugData = {"segmentation": None, "opticalFlow": None}

segWndName = "Segmentation masks"
flwWndName = "Optical flow"

def dbgVis():
    while dbgGVar.get("keepOn"):
        debugDataLock.acquire()
        segImg = debugData["segmentation"]
        flwImg = debugData["opticalFlow"]
        debugDataLock.release()
        if segImg is not None:
            cv.imshow(segWndName, segImg)
        if flwImg is not None:
            cv.imshow(flwWndName, flwImg)
        cv.waitKey(10)

startVisualizationThread, stopVisualizationThread = makeStartStopFns(dbgGVar, dbgVis)
