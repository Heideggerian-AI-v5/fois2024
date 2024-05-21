import cv2 as cv
import math
import os

imgWidth, imgHeight = (240, 240)
endImg = imgHeight*imgWidth*3
endDepth = endImg + imgHeight*imgWidth*8

rawImgDataFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/raw")
yoloImgDataFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/yolodata")

modelPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../object_recognition_segmentation/best.pt")
modelNObjsPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model/obj/model.pt")
modelSObjsPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model/supp/model.pt")

turtleImgSocketURL = "ws://localhost:8765"

segmentationDilationRadius = 5
feature_params = dict(maxCorners = 5, qualityLevel = 0.2, minDistance = 2, blockSize = 5)
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
opticalFlowColor = (255/255., 128/255., 255/255.)
contactColor = (128/255., 255/255., 255/255.)
FOVdeg = 60
FOVrad = FOVdeg*math.pi/180.0
halfH = imgHeight/2
f = halfH/(math.tan(FOVrad/2))

APPROACHVELOCITY = -0.02
DEPARTUREVELOCITY = 0.02

perceptionInterpretationTheoryFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), './perception_interpretation.dfl')
updateSchemasTheoryFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), './update_schemas.dfl')
connQueryTheoryFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), './conn_queries.dfl')
closureTheoryFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), './closure.dfl')
schemaInterpretationTheoryFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), './schema_interpretation.dfl')
updateQuestionsTheoryFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), './update_questions.dfl')
backgroundFactsFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), './background_facts.dfl')
transportSmallPottedPlantFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), './transport_small_potted_plant.dfl')
checkSupportFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), './check_support.dfl')
gatherSupportDataFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), './gather_support.dfl')
