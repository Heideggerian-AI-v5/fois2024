import argparse
import copy
import json
import math
import numpy
import os
import platform
import pybullet
import requests
import signal
import sys
import threading
import time
import websockets
import asyncio
from websockets.sync.server import serve
import functools

import cv2 as cv

from flask import Flask, request

import turtle_sim.world as world

from turtle_sim.commands import commandFns

from turtle_sim.commands import serverGetImage

def startProcessCommand(command, requestData, w, agentName, todos):
    return commandFns[command][0](commandFns[command][1], requestData, w, agentName, todos)
    #return doingAction, status, response
    
def stopProcessCommand(command, requestData, w, agentName):
    status, response = commandFns[command][2](requestData, w, agentName)
    return False, status, response

def placeCamera(w, item):
    pos, rot, _, _ = w.getKinematicData((item,))
    px, py, _ = pos
    _, _, yaw = world.stubbornTry(lambda : pybullet.getEulerFromQuaternion(rot))
    world.stubbornTry(lambda : pybullet.resetDebugVisualizerCamera(1,yaw*180/math.pi-90,-20, [px,py,1]))
    
def serviceRequest(command, request, requestDictionary, responseDictionary, updating, executingAction):
    if command not in commandFns:
        return {'response': 'Unrecognized command.'}, requests.status_codes.codes.NOT_IMPLEMENTED
    with updating:
        commandId = str(time.time())
        answeringRequest = threading.Condition()
        try:
            requestDictionary[commandId] = [answeringRequest, command, request.get_json(force=True)]
        except SyntaxError:
            return {'response': 'Malformed json.'}, requests.status_codes.codes.BAD_REQUEST
    with answeringRequest:
        answeringRequest.wait()
    with updating:
        _, status, response = responseDictionary.pop(commandId)
    return json.dumps(response), status

def thread_function_flask(requestDictionary, responseDictionary, updating, executingAction):
    flask = Flask(__name__)
    @flask.route("/abe-sim-command/<string:command>", methods=['POST'])
    def flaskRequest(command):
        return serviceRequest(command, request, requestDictionary, responseDictionary, updating, executingAction)
    flask.run(port=54321, debug=True, use_reloader=False)

def handleINT(signum, frame):
    sys.exit(0)

labelMap = {"CardboardBox": 0, "Coatrack": 1, "Hextable": 2, "Pillow": 3, "PottedPlant": 4, "SmallPottedPlant": 5, "Table": 6, "TableLamp": 7, "TallChair": 8, "Trashbin": 9, "Hook": 10, "Tongs": 11, "BeerMug": 12}

def segmentationSnapshot(w, agentName):
    def _bboxArea(c):
        mx, my, Mx, My = 240,240,0,0
        for p in c:
            if mx > p[0]:
                mx = p[0]
            if Mx < p[0]:
                Mx = p[0]
            if my > p[1]:
                my = p[1]
            if My < p[1]:
                My = p[1]
        return (Mx-mx)*(My-my)
    def _id2Label(w, i):
        if i in w._idx2KinematicTree:
            return labelMap.get(w._kinematicTrees[w._idx2KinematicTree[i]]["type"], -1)
        return -1
    fnamePrefix = "seg_%s" % time.asctime().replace(" ", "_").replace(":","_")
    _, rgb, _, sem = serverGetImage(w, agentName)
    rgb.save(fnamePrefix+".jpg")
    segImgs = {}
    for k,row in enumerate(sem):
        for j,c in enumerate(row):
            if c not in segImgs:
                segImgs[c] = numpy.zeros((240,240),dtype=numpy.uint8)
            segImgs[c][k][j] = 255
    with open(fnamePrefix + ".txt", "w") as outfile:
        for k, v in segImgs.items():
            label = _id2Label(w, k)
            if -1 != label:
                contours, hierarchy = cv.findContours(image=v, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_TC89_KCOS)
                #contours, hierarchy = cv.findContours(image=v, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
                #contours, hierarchy = cv.findContours(image=v, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
                contours = [x.reshape((len(x), 2)) for x in contours]
                for polygon, h in zip(contours, hierarchy[0]):
                    if 0 > h[3]:
                        pstr = ""
                        for p in polygon:
                            pstr += ("%f %f " % (p[0]/240., p[1]/240.))
                        if 0 < len(polygon):
                            pstr += ("%f %f " % (polygon[0][0]/240., polygon[0][1]/240.))
                        _ = outfile.write("%d %s\n" % (label, pstr))

def processWheelRequests(w, agentName):
    #keys = world.stubbornTry(lambda : pybullet.getKeyboardEvents())
    keys = (pybullet.getKeyboardEvents())
    csv = w._kinematicTrees[agentName]["customStateVariables"]
    leftWheelVelocity = csv["left"]
    rightWheelVelocity = csv["right"]
    forward,turn = 0,0
    speed=50
    turtleIdx = w._kinematicTrees[agentName]["idx"]
    for k,v in keys.items():
        if (k == pybullet.B3G_RIGHT_ARROW and (v&(pybullet.KEY_WAS_TRIGGERED+pybullet.KEY_IS_DOWN))):
            turn = -0.5
        if (k == pybullet.B3G_RIGHT_ARROW and (v&pybullet.KEY_WAS_RELEASED)):
            turn = 0
        if (k == pybullet.B3G_LEFT_ARROW and (v&(pybullet.KEY_WAS_TRIGGERED+pybullet.KEY_IS_DOWN))):
            turn = 0.5
        if (k == pybullet.B3G_LEFT_ARROW and (v&pybullet.KEY_WAS_RELEASED)):
            turn = 0
        if (k == pybullet.B3G_UP_ARROW and (v&(pybullet.KEY_WAS_TRIGGERED+pybullet.KEY_IS_DOWN))):
            forward=1
        if (k == pybullet.B3G_UP_ARROW and (v&pybullet.KEY_WAS_RELEASED)):
            forward=0
        if (k == pybullet.B3G_DOWN_ARROW and (v&(pybullet.KEY_WAS_TRIGGERED+pybullet.KEY_IS_DOWN))):
            forward=-1
        if (k == pybullet.B3G_DOWN_ARROW and (v&pybullet.KEY_WAS_RELEASED)):
            forward=0
        if (k == pybullet.B3G_SPACE) and (v&pybullet.KEY_WAS_RELEASED):
            segmentationSnapshot(w, agentName)
    rightWheelVelocity+= (forward+turn)*speed
    leftWheelVelocity += (forward-turn)*speed
    #print("LRV", leftWheelVelocity, rightWheelVelocity)
    #w._kinematicTrees[agentName]["customStateVariables"]["left"] = leftWheelVelocity
    #w._kinematicTrees[agentName]["customStateVariables"]["right"] = rightWheelVelocity
    world.stubbornTry(lambda : pybullet.setJointMotorControl2(turtleIdx,0,pybullet.VELOCITY_CONTROL,targetVelocity=leftWheelVelocity,force=1000))
    world.stubbornTry(lambda : pybullet.setJointMotorControl2(turtleIdx,1,pybullet.VELOCITY_CONTROL,targetVelocity=rightWheelVelocity,force=1000))

def hello(websocket, w, agentName):
    while True:
        imageToSend = serverGetImage(w, agentName)[0]
        websocket.send(imageToSend)

def minidump(websocket, w):
    while True:
        wd = w.worldDump()
        websocket.send(json.dumps(wd).encode())
        time.sleep(0.1)

def thread_function_websockets(w, agentName):
    with websockets.sync.server.serve(functools.partial(hello, w = w, agentName = agentName), "localhost", 8765) as server:
        server.serve_forever()

def thread_function_worldstate(w):
    with websockets.sync.server.serve(functools.partial(minidump, w=w), "localhost", 4321) as server:
        server.serve_forever()

def runBrain():
    parser = argparse.ArgumentParser(prog='runBrain', description='Run the Turtle Sim', epilog='kwargs for the loadObjectList is a dictionary. Possible keys are linearVelocity (of base, value is a float), angularVelocity (of base, value is a float) and jointPositions (value is a dictionary where keys are link names and values are floats representing the position of the parent joint for the link).')
    parser.add_argument('-fdf', '--frameDurationFactor', default="1.0", help='Attempts to adjust the ratio of real time of frame to simulated time of frame. A frame will always last, in real time, at least as long as it is computed. WARNING: runBrain will become unresponsive to HTTP API calls if this is set too low. Recommended values are above 0.2.')
    parser.add_argument('-sfr', '--simFrameRate', default="160", help='Number of frames in one second of simulated time. Should be above 60.')
    parser.add_argument('-a', '--agent', help='Name of the agent to control in the loaded scene')
    parser.add_argument('-g', '--useGUI', action='store_true', help='Flag to enable the GUI')
    parser.add_argument('-o', '--useOpenGL', action='store_true', help='Flag to enable hardware acceleration. Warning: often unstable on Linux; ignored on MacOS')
    parser.add_argument('-p', '--preloads', default=None, help='Path to a file containing a json list of objects to preload. Each element of this list must be of form [type, name, position]')
    parser.add_argument('-w', '--loadWorldDump', default=None, help='Path to a file containing a json world dump from a previous run of Turtle Sim')
    parser.add_argument('-l', '--loadObjectList', default='./turtle_sim/defaultScene.json', help='Path containing a json list of objects to load in the scene. Each element in the list must be of form [type, name, position, orientation, kwargs] (kwargs optional)')
    arguments = parser.parse_args()

    objectTypeKnowledge = json.loads(open('./turtle_sim/objectKnowledge.json').read())
    objectTypeKnowledge = {x['type']: x for x in objectTypeKnowledge}

    useGUI = arguments.useGUI
    useOpenGL = arguments.useOpenGL
    preloads = arguments.preloads
    gravity = (0,0,-10) # TODO argparse
    loadWorldDump = arguments.loadWorldDump
    loadObjectList = arguments.loadObjectList
    frameDurationFactor = float(arguments.frameDurationFactor)
    sfr = int(arguments.simFrameRate)
    agentName = arguments.agent

    if useOpenGL:
        pybulletOptions = "--opengl3"
    else:
        pybulletOptions = "--opengl2" # Software-only "tiny" renderer. Should work on Linux and when support for graphical hardware acceleration is inconsistent.

    isAMac = ('Darwin' == platform.system())
    ## WORLD CREATION line: adjust this as needed on your system.
    # TODO if you want to run headless: useGUI=False in World()
    if not isAMac:
        w = world.World(pybulletOptions = pybulletOptions, useGUI=useGUI, objectKnowledge=objectTypeKnowledge, simFrameRate=sfr)
    else:
        w = world.World(pybulletOptions = "", useGUI=useGUI, objectKnowledge=objectTypeKnowledge, simFrameRate=sfr) # Hardware-accelerated rendering. Seems necessary on newer Macs.
        
    w.setGravity(gravity)

    objectInstances = []

    if preloads is not None:
        preloads = json.loads(open(preloads).read())
        for oType, oname, position in toPreload:
            w.preload(oType, oname, position)
    if loadWorldDump is not None:
        w.greatReset(json.loads(open(loadWorldDump).read()))
    elif loadObjectList is not None:
        for x in json.loads(open(loadObjectList).read()):
            spec = None
            if 4 == len(x):
                otype, oname, position, orientation = x
            elif 5 == len(x):
                otype, oname, position, orientation, spec = x
            else:
                continue
            objI = w.addObjectInstance(otype, oname, position, orientation)
            objectInstances.append(objI)
            if spec is not None:
                if "linearVelocity" in spec:
                    w.setObjectProperty((oname), "linearVelocity", spec["linearVelocity"])
                if "angularVelocity" in spec:
                    w.setObjectProperty((oname), "angularVelocity", spec["angularVelocity"])
                if "jointPositions" in spec:
                    for lnk, pos in spec["jointPositions"].items():
                        w.setObjectProperty((oname, lnk), "jointPosition", pos)
                        
    agentTypes = {"TurtleBot"} # TODO: <<- that is a good place to insert a query to an OWL reasoner
    # TODO: these are also a good opportunity to use OWL reasoning queries
    sceneAgents = [k for k in w._kinematicTrees.keys() if w._kinematicTrees[k]["type"] in agentTypes]
    
    waitingFor = 0
    
    if (agentName is None) or (agentName not in w._kinematicTrees.keys()):
        agentName = [x['name'] for x in w._kinematicTrees.values() if 'TurtleBot' == x['type']][0]
    
    executingAction = threading.Condition()
    updating = threading.Lock()
    requestDictionary = {}
    responseDictionary = {}
    currentAction = None

    flaskThread = threading.Thread(target=thread_function_flask, args=(requestDictionary, responseDictionary, updating, executingAction))
    flaskThread.start()
    signal.signal(signal.SIGINT, handleINT)
    signal.signal(signal.SIGTERM, handleINT)

    websocketThread = threading.Thread(target=thread_function_websockets, args=(w,agentName))
    websocketWDThread = threading.Thread(target=thread_function_worldstate, args=(w,))
    websocketThread.start()
    websocketWDThread.start()
    
    todos = {"currentAction": None, "goals": [], "requestData": {}, "command": None, "cancelled": False}
    
    while True:
        stepStart = time.perf_counter()
        with updating:
            for commandId, commandSpec in list(requestDictionary.items()):
                if commandId == todos["currentAction"]:
                    continue
                answeringRequest, command, requestData = commandSpec
                doingAction, status, response = startProcessCommand(command, requestData, w, agentName, todos)
                responseDictionary[commandId] = doingAction, status, response
                requestDictionary.pop(commandId)
                with answeringRequest:
                    answeringRequest.notify_all()
            processWheelRequests(w, agentName)
            w.update()
            placeCamera(w, agentName)
        stepEnd = time.perf_counter()
        if not isAMac:
            time.sleep(max((frameDurationFactor/(sfr*1.0))-(stepEnd-stepStart), 0.001))

if "__main__" == __name__:
    runBrain()
