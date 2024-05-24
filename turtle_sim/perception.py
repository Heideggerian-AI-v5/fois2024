import contact
import cv2 as cv
import math
import numpy
import threading
import time

import inflection

from constants import imgWidth, imgHeight, segmentationDilationRadius
from constants import feature_params, lk_params, opticalFlowColor, contactColor, FOVdeg, FOVrad, halfH, f, APPROACHVELOCITY, DEPARTUREVELOCITY
from objectrecognition import imageReady, recognitionResultsLock, objectRecognitionResults
from objectrecognition import startReceptionThread, stopReceptionThread, startRecognitionThread, stopRecognitionThread
from dbgvis import startVisualizationThread, stopVisualizationThread, debugDataLock, debugData
from utils import makeStartStopFns

perceptionQuestionsLock = threading.Lock()
perceptionResultsLock = threading.Lock()
perceptionReady = threading.Condition()
perceptionQueries = {"relativeMovements": [], "contacts": []}
perceptionResults = {"robotRelativeVelocities": [], "relativeMovements": [], "contacts": [], "image": None, "contactMasks": {}}

perceptionGVar= {}

def remapResultNames(resultDicts):
    def _remap(x):
        x["type"] = inflection.camelize(x["type"])
        return x
    return [_remap(e) for e in resultDicts]

def updatePNDPair(prev, now, keys, newvals):
    _ = [prev.pop(ok) for ok in list(prev.keys()) if ok not in keys]
    _ = [now.pop(ok) for ok in list(now.keys()) if ok not in keys]
    for k in keys:
        prev[k] = now.get(k)
    if newvals is None:
        for k in keys:
            now[k] = None
    else:
        for k in keys:
            now[k] = newvals.get(k)
    return prev, now

def getQueriedObjects(queries, recognitionResultDicts):
    # IF only having queries about particular object pairs the below code works
    #retq = set()
    #for k in queries.keys():
    #    retq = retq.union(*queries[k])
    #if None in retq:
    #    retq.remove(None)
    # HOWEVER we will in general be interested in queries of the form "what approaches X?"
    retq = set([e["type"] for e in recognitionResultDicts])
    return retq

uMatrix = numpy.zeros((imgHeight,imgWidth))
vMatrix = numpy.zeros((imgHeight,imgWidth))
for v in range(imgHeight):
    for u in range(imgWidth):
        uMatrix[v][u] = u-halfH
        vMatrix[v][u] = v-halfH

def inverseProjection(point, depthImg):
    if isinstance(point, numpy.ndarray):
        u,v = point.ravel().astype(int)
    else:
        u,v = point
    u = min(imgWidth-1,max(0,u))
    v = min(imgHeight-1,max(0,v))
    depth = depthImg[v][u]
    xi = (u - halfH)*depth/(f)
    yi = (v - halfH)*depth/(f)
    return numpy.array((xi, yi, depth))

def getFeatures(previousFeatures, previousGray, previousMaskImg):
    if previousMaskImg is None:
        return None
    if previousFeatures is None:
        previousFeatures = numpy.array([])
    adj_feature_params = feature_params.copy()
    alreadyPresent = 0
    alreadyPresent = len(previousFeatures)
    adj_feature_params['maxCorners'] = feature_params['maxCorners'] - alreadyPresent
    if (0 < adj_feature_params['maxCorners']):
        newFeatures = cv.goodFeaturesToTrack(previousGray, mask=previousMaskImg, **adj_feature_params)
        if newFeatures is not None:
            newFeatures = newFeatures.astype(numpy.float32)
            if (0 < len(newFeatures)):
                if (0 < len(previousFeatures)):
                    previousFeatures = numpy.concatenate((previousFeatures, newFeatures))
                else:
                    previousFeatures = newFeatures
    return previousFeatures

def computeOpticalFlow(previousFeatures, previousGray, gray, maskImg, depthImgPrev, depthImg):
    def _insideMask(x, maskImg):
        cx, cy = x.ravel().astype(int)
        cx = min(imgWidth-1,max(0,cx))
        cy = min(imgHeight-1,max(0,cy))
        return 0 < maskImg[cy][cx]
    if (previousFeatures is None) or (0 == len(previousFeatures)):
        return None, None, None, None
    if maskImg is None:
        previous3D = [inverseProjection(x, depthImgPrev) for x in previousFeatures]
        return previousFeatures, None, previous3D, None
    next, status, error = cv.calcOpticalFlowPyrLK(previousGray, gray, previousFeatures, None, **lk_params)
    good_old = numpy.array([],dtype=numpy.float32)
    good_new = numpy.array([],dtype=numpy.float32)
    if next is not None:
        good_old = previousFeatures[status == 1]
        good_old = good_old.reshape((len(good_old), 1, 2))
        good_new = next[status==1].astype(numpy.float32)
        good_new = good_new.reshape((len(good_new), 1, 2))
    onSameObj = [_insideMask(x, maskImg) for x in good_new]
    aux = [(x,y) for x,y,z in zip(good_new, good_old, onSameObj) if z]
    good_new = numpy.array([x[0] for x in aux]).reshape((len(aux),1,2)).astype(numpy.float32)
    good_old = numpy.array([x[1] for x in aux]).reshape((len(aux),1,2)).astype(numpy.float32)
    previous3D = [inverseProjection(x, depthImgPrev) for x in good_old]
    now3D = [inverseProjection(x, depthImg) for x in good_new]
    return good_old, good_new, previous3D, now3D

def getRobotRelativeKinematics(previous3D, now3D):
    if 0 == min(len(now3D),len(previous3D)):
        return None, None
    retq = [(nw - pr) for pr, nw in zip(previous3D, now3D)]
    return sum(now3D)/len(now3D), sum(retq)/len(retq)

def getRelativeMovements(previous3D, now3D, queries):
    def _relativeSpeed(vs, vo, ps, po):
        if (vs is None) or (vo is None) or (ps is None) or (po is None):
            return None
        dVVec = [a-b for a,b in zip(vo, vs)]
        dPVec = [a-b for a,b in zip(po, ps)]
        dPNorm = math.sqrt(sum(x*x for x in dPVec))
        dPDir = [x/dPNorm for x in dPVec]
        return sum(a*b for a,b in zip(dPDir, dVVec))
    kinematicData = {k: getRobotRelativeKinematics(previous3D[k], now3D[k]) for k in now3D.keys() if (previous3D.get(k) is not None) and (now3D[k] is not None)}
    kinematicData["Agent"] = (numpy.array([0,0,0]), numpy.array([0,0,0]))
    retq = []
    for s,oq in queries:
        if oq is None:
            oq = [x for x in kinematicData.keys() if s!=o]
        else:
            oq = [oq]
        for o in oq:
            if (s in kinematicData) and (o in kinematicData):
                dV = _relativeSpeed(kinematicData[s][1], kinematicData[o][1], kinematicData[s][0], kinematicData[o][0])
                if dV is None:
                    continue
                if APPROACHVELOCITY >= dV:
                    retq.append(("approaches", s, o))
                elif DEPARTUREVELOCITY <= dV:
                    retq.append(("departs", s, o))
                else:
                    retq.append(("stillness", s, o))
    return retq

def getSelfMask(dilate=False):
    mask = numpy.zeros((imgHeight,imgWidth), dtype=numpy.uint8)
    thicc = 5
    if dilate:
        thicc = 10
    return cv.line(mask, (thicc-1,imgHeight-thicc), (imgWidth-thicc,imgHeight-thicc), 255, thicc)

def getContacts(maskImgs, depthImg, queries):
    dilationKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*5 + 1, 2*5 + 1), (5, 5))
    queries = [e for e in queries if ((e[0] in maskImgs) or ("Agent" == e[0])) and ((e[1] in maskImgs) or ("Agent" == e[1]) or (e[1] is None))]
    retq = []
    contactPixels = {}
    contactMasks = {}
    sT = time.perf_counter()
    querySet = {}
    dilImgs = {}
    for s,oq in queries:
        if oq is None:
            oq = [(k, False) for k in maskImgs.keys() if s != k]
            #if "Agent" != s:
            #    oq.append(("Agent", False))
        else:
            oq = [(oq, True)]
        for o,d in oq:
            p = (s,o)
            rp = (o, s)
            if rp in querySet:
                if d:
                    querySet[rp] = True
                continue
            if p not in querySet:
                querySet[p] = False
            if d:
                querySet[p] = True
            for e in p:
                if e not in dilImgs:
                    dilImg = cv.dilate(maskImgs[e], dilationKernel)
                    dilImgs[e] = dilImg
    for so, detail in querySet.items():
        s, o = so
        cmaskS, cmaskO = contact.contact(5, 0.05, imgHeight, imgWidth, s, o, dilImgs, maskImgs, depthImg)
        pixelsS = numpy.argwhere((cmaskS > 0))
        pixelsO = numpy.argwhere((cmaskO > 0))
        contacting = (0 < len(pixelsS)) and (0 < len(pixelsO))
        if contacting:
            retq.append(("contacting", s, o))
        else:
            retq.append(("-contacting", s, o))
        if detail:
            pixelsO.T[[0,1]] = pixelsO.T[[1,0]]
            pixelsS.T[[0,1]] = pixelsS.T[[1,0]]
            contactPixels[(s,o,o)] = pixelsO
            contactPixels[(s,o,s)] = pixelsS    
            contactMasks[(s,o,o)] = cmaskO
            contactMasks[(s,o,s)] = cmaskS
    print("XYC", time.perf_counter()-sT)
    return retq, contactPixels, contactMasks

def perception(dbg = False):
    def _objPolygons(label,img,outfile):
        contours, hierarchy = cv.findContours(image=img, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
        contours = [x.reshape((len(x), 2)) for x in contours]
        for polygon, h in zip(contours, hierarchy[0]):
            if 0 > h[3]:
                pstr = ""
                for p in polygon:
                    pstr += ("%f %f " % (p[0]/240., p[1]/240.))
                if 0 < len(polygon):
                    pstr += ("%f %f " % (polygon[0][0]/240., polygon[0][1]/240.))
                _ = outfile.write("%d %s\n" % (label, pstr))    
    perceptionQueriesLocal = {"relativeMovements": [], "contacts": []}
    previousGray, gray = None, None
    previousMaskImgs, maskImgs, previousFeatures, nowFeatures, previous3D, now3D = {}, {}, {}, {}, {}, {}
    maskImgs = {}
    while perceptionGVar.get("keepOn"):
        with imageReady:
            imageReady.wait()
        recognitionResultsLock.acquire()
        image = objectRecognitionResults["image"]
        resultDicts = objectRecognitionResults["resultDicts"]
        depthImg = objectRecognitionResults["depthImg"]
        depthImgPrev = objectRecognitionResults["depthImgPrev"]
        recognitionResultsLock.release()
        resultDicts = remapResultNames(resultDicts)
        if image is None:
            continue
        npImg = numpy.ascontiguousarray(numpy.float32(numpy.array(image)[:,:,(2,1,0)]))/255.0
        previousGray = gray
        gray = numpy.uint8(cv.cvtColor(npImg, cv.COLOR_BGR2GRAY)*255)
        previous3D, now3D = {}, {}
        if (previousGray is None) or (gray is None) or (depthImgPrev is None) or (depthImg is None):
            previousMaskImgs, maskImgs, previousFeatures, nowFeatures = {}, {}, {}, {}
            continue
        perceptionQuestionsLock.acquire()
        perceptionQueriesLocal = {k:v for k,v in perceptionQueries.items()}
        perceptionQuestionsLock.release()
        queriedObjects = getQueriedObjects(perceptionQueriesLocal, resultDicts)
        updatePNDPair(previousMaskImgs, maskImgs, queriedObjects, {e["type"]: e["mask"] for e in resultDicts})
        maskImgs["Agent"] = getSelfMask()
        updatePNDPair(previousFeatures, nowFeatures, queriedObjects, None)
        for o in queriedObjects:
            previousFeatures[o] = getFeatures(previousFeatures.get(o), previousGray, previousMaskImgs.get(o))
            previousFeatures[o], nowFeatures[o], previous3D[o], now3D[o] = computeOpticalFlow(previousFeatures.get(o), previousGray, gray, maskImgs.get(o), depthImgPrev, depthImg)
        relativeMovements = getRelativeMovements(previous3D, now3D, perceptionQueriesLocal["relativeMovements"])
        contacts, contactPixels, contactMasks = getContacts(maskImgs, depthImg, perceptionQueriesLocal["contacts"])
        perceptionResultsLock.acquire()
        perceptionResults["relativeMovements"] = relativeMovements
        perceptionResults["contacts"] = contacts
        perceptionResults["image"] = image
        perceptionResults["contactMasks"] = contactMasks
        perceptionResultsLock.release()
        if dbg:
            #print("QC\n", perceptionQueriesLocal["contacts"])
            print("Relative Movements", sorted(perceptionResults["relativeMovements"]))
            print("Contacts", sorted(perceptionResults["contacts"]))
        with perceptionReady:
            perceptionReady.notify_all()
        if dbg:
            for cd, pixels in contactPixels.items():
                if 0 < len(pixels):
                    print(cd)
                for p in pixels:
                    npImg = cv.line(npImg, p, p, contactColor, 1)
            for k in nowFeatures.keys():
                if (previousFeatures.get(k) is None) or (nowFeatures.get(k) is None):
                    continue
                for i, (new, old) in enumerate(zip(nowFeatures[k], previousFeatures[k])):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    npImg = cv.line(npImg, (a,b), (c,d), opticalFlowColor, 2)
            debugDataLock.acquire()
            debugData["opticalFlow"] = npImg
            debugDataLock.release()

startPerceptionThread, stopPerceptionThread = makeStartStopFns(perceptionGVar, perception)

if "__main__" == __name__:
    perceptionQueries["relativeMovements"] = [("Agent", "Trashbin"), ("SmallPottedPlant", "TallChair")]
    perceptionQueries["contacts"] = [("SmallPottedPlant", "TallChair")]
    startVisualizationThread()
    startReceptionThread()
    startRecognitionThread(dbg=True)
    startPerceptionThread(dbg=True)
    input("Reception/Recognition and Perception threads running. Switch to the turtlebot GUI and move it around to see the changing segmentation masks and optical flow/contact computations. When you want to exit this program, press ENTER in this terminal.")
    stopPerceptionThread()
    stopRecognitionThread()
    stopReceptionThread()
    print("Threads stopped, exiting.")
    stopVisualizationThread()

