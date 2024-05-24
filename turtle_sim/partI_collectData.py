import argparse
import itertools
import networkx
import os

from pathlib import Path

import silkie
from dbgvis import startVisualizationThread, stopVisualizationThread, debugDataLock, debugData
from objectrecognition import startReceptionThread, stopReceptionThread, startRecognitionThread, stopRecognitionThread
from perception import perceptionQueries, perceptionQuestionsLock, perceptionResults, perceptionResultsLock, perceptionReady, startPerceptionThread, stopPerceptionThread
from utils import makeStartStopFns
from constants import perceptionInterpretationTheoryFile, updateSchemasTheoryFile, connQueryTheoryFile, closureTheoryFile,  schemaInterpretationTheoryFile, updateQuestionsTheoryFile, backgroundFactsFile, gatherSupportDataFile, modelNObjsPath, rawImgDataFolder
from reasoning import startReasoningThread, stopReasoningThread

if "__main__" == __name__:
    parser = argparse.ArgumentParser(prog='partI_collectData.py', description='Part I of the experiment: look at frames collected by the robot as it is moving, and look for instances of support', epilog='')
    parser.add_argument('-dest', '--destination', default="", help='Path to folder where frames are to be stored.')
    arguments = parser.parse_args()
    dest=arguments.destination
    if ""==dest:
        dest=rawImgDataFolder
    Path(dest).mkdir(parents=True, exist_ok=True)

    gatherSupportData = silkie.loadDFLFacts(gatherSupportDataFile)
    gatherSupportData["storeAt"]=silkie.PFact("storeAt")
    gatherSupportData["storeAt"].addFact(dest,dest,silkie.STRICT)
    
    startVisualizationThread()
    startReceptionThread()
    startRecognitionThread(dbg=True,modelFileName=modelNObjsPath)
    startPerceptionThread(dbg=True)
    startReasoningThread(dbg=True, persistentSchemas=gatherSupportData)
    input("Reception/Recognition, Perception, and Reasoning threads running. Switch to the turtlebot GUI (e.g., click on its window) and move it around to see the changing segmentation masks and optical flow/contact computations. When you want to exit this program, press ENTER in this terminal.")
    stopReasoningThread()
    stopPerceptionThread()
    stopRecognitionThread()
    stopReceptionThread()
    print("Threads stopped, exiting.")
    stopVisualizationThread()

