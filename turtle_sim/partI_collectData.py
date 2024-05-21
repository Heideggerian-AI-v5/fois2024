import itertools
import networkx
import os

import silkie
from dbgvis import startVisualizationThread, stopVisualizationThread, debugDataLock, debugData
from objectrecognition import startReceptionThread, stopReceptionThread, startRecognitionThread, stopRecognitionThread
from perception import perceptionQueries, perceptionQuestionsLock, perceptionResults, perceptionResultsLock, perceptionReady, startPerceptionThread, stopPerceptionThread
from utils import makeStartStopFns
from constants import perceptionInterpretationTheoryFile, updateSchemasTheoryFile, connQueryTheoryFile, closureTheoryFile,  schemaInterpretationTheoryFile, updateQuestionsTheoryFile, backgroundFactsFile, gatherSupportDataFile, modelNObjsPath
from reasoning import startReasoningThread, stopReasoningThread

if "__main__" == __name__:
    gatherSupportData = silkie.loadDFLFacts(gatherSupportDataFile)
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
