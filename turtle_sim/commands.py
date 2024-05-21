import copy
import math
import numpy
import pybullet
import requests
import time

from PIL import Image
import base64

from turtle_sim.world import stubbornTry

def _checkArgs(args):
    retq = []
    for a in args:
        var, msg = a
        if var is None:
            retq.append(msg)
    return retq
    
def toGetTime(requestData, w, agentName, todos):
    return requests.status_codes.codes.ALL_OK, {"response": {"time": float(time.perf_counter())}}
    
def toGetWorldState(requestData, w, agentName, todos):
    return requests.status_codes.codes.ALL_OK, {"response": {"worldState": w.worldDump()}}
    
def toSetWorldState(requestData, w, agentName, todos):
    worldStateIn = requestData.get('worldStateIn', None)
    lacks = _checkArgs([[worldStateIn, "Request lacks worldStateIn parameter."]])
    if 0 < len(lacks):
        return requests.status_codes.codes.BAD_REQUEST, {'response': ' '.join(lacks)}
    w.greatReset(worldStateIn)
    return requests.status_codes.codes.ALL_OK, {"response": "Ok."}
    
def toGetStateUpdates(requestData, w, agentName, todos):
    updates = {}
    for name, data in w._kinematicTrees.items():
        mesh = stubbornTry(lambda : pybullet.getVisualShapeData(data['idx'], -1, w._pybulletConnection))[0][4].decode("utf-8")
        updates[name] = {'filename': str(data['filename']), 'position': list(w.getObjectProperty((name,), 'position')), 'orientation': list(w.getObjectProperty((name,), 'orientation')), 'at': str(w.getObjectProperty((name,), 'at')), 'mesh': mesh, 'customStateVariables': copy.deepcopy(data.get('customStateVariables', {})), 'joints': w.getObjectProperty((name,), 'jointPositions')}
    return requests.status_codes.codes.ALL_OK, {"response": {"updates": updates, "currentCommand": "", "abeActions": []}}

def toPreloadObject(requestData, w, agentName, todos):
    otype = requestData.get('type', None)
    if otype not in w._objectKnowledge:
        return requests.status_codes.codes.NOT_FOUND, {"response": ('World does not have object type %s in its knowledge base' % otype)}
    ## TODO: add a return value to check that this is ok ...
    w.preload(otype, None, [0,0,-10])
    return requests.status_codes.codes.ALL_OK, {'response': 'ok'}

def serverGetImage(w,agentName):
    position, orientation, _, _ = w.getKinematicData((agentName,))
    position = [position[0], position[1], position[2] + 0.6]
    rv = pybullet.rotateVector(orientation, [1,0,0])
    cameraTarget = [position[0] + rv[0], position[1] + rv[1], position[2] - 0.3]
    near, far = 0.1, 15
    halfH, FOVdeg = 120, 60
    FOVrad = FOVdeg*math.pi/180.0
    projectionMatrix = pybullet.computeProjectionMatrixFOV(FOVdeg, 1, near, far)
    viewMatrix = pybullet.computeViewMatrix(position, cameraTarget, [0,0,1])
    ans = pybullet.getCameraImage(240,240,viewMatrix=viewMatrix,projectionMatrix=projectionMatrix,shadow = False, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    depthImg = (far*near/(far - (far-near)*numpy.reshape(ans[3], (240,240)))).astype(float)
    segImg = numpy.reshape(pybullet.getCameraImage(240,240,viewMatrix=viewMatrix,projectionMatrix=projectionMatrix,shadow = False, renderer=pybullet.ER_TINY_RENDERER)[4], (240,240)).astype(numpy.uint16)
    rgbBuffer = numpy.reshape(ans[2], (240, 240, 4)) * 1. / 255.
    rgbim = Image.fromarray((rgbBuffer * 255).astype(numpy.uint8))
    rgbim_noalpha = rgbim.convert('RGB')
    rgbBytes = rgbim_noalpha.tobytes()
    base64Image = base64.b64encode(rgbBytes+depthImg.tobytes()+segImg.tobytes())
    return base64Image, rgbim_noalpha, depthImg, segImg

def toGetImage(requestData, w, agentName, todos):
    #timeReq = float(time.perf_counter())
    position, orientation, _, _ = w.getKinematicData((agentName,))
    position = [position[0], position[1], position[2] + 0.6]
    rv = pybullet.rotateVector(orientation, [1,0,0])
    cameraTarget = [position[0] + rv[0], position[1] + rv[1], position[2] - 0.3]
    near, far = 0.1, 15
    halfH, FOVdeg = 120, 60
    FOVrad = FOVdeg*math.pi/180.0
    projectionMatrix = pybullet.computeProjectionMatrixFOV(FOVdeg, 1, near, far)
    viewMatrix = pybullet.computeViewMatrix(position, cameraTarget, [0,0,1])
    ans = pybullet.getCameraImage(240,240,viewMatrix=viewMatrix,projectionMatrix=projectionMatrix,shadow = False, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    #timeImageGen = float(time.perf_counter())
    rgbBuffer = numpy.reshape(ans[2], (240, 240, 4)) * 1. / 255.
    rgbim = Image.fromarray((rgbBuffer * 255).astype(numpy.uint8))
    rgbim_noalpha = rgbim.convert('RGB')
    #rgbim_noalpha.save("turtleSim.jpg")
    rgbBytes = rgbim_noalpha.tobytes()
    base64Image = base64.b64encode(rgbBytes)
    #timeSend = float(time.perf_counter())
    #print("Time to imageGen", timeImageGen - timeReq)
    #print("Time to convert", timeSend - timeImageGen)
    return requests.status_codes.codes.ALL_OK, {"response": {"image": base64Image.decode()}}

def toSetWheelSpeeds(requestData, w, agentName, todos):
    w._kinematicTrees[agentName]["customStateVariables"]["left"] = requestData["left"]
    w._kinematicTrees[agentName]["customStateVariables"]["right"] = requestData["right"]
    return requests.status_codes.codes.ALL_OK, {"response": "ok"}

def processInstantRequest(fn, requestData, w, agentName, todos):
    status, response = fn(requestData, w, agentName, todos)
    return False, status, response
    
commandFns = {
    "to-get-time": [processInstantRequest, toGetTime, None], 
    "to-get-world-state": [processInstantRequest, toGetWorldState, None],
    "to-set-world-state": [processInstantRequest, toSetWorldState, None],
    "to-preload-object": [processInstantRequest, toPreloadObject, None],
    "to-get-state-updates": [processInstantRequest, toGetStateUpdates, None],
    "to-get-image": [processInstantRequest, toGetImage, None],
    "to-set-wheel-speeds": [processInstantRequest, toSetWheelSpeeds, None]}

