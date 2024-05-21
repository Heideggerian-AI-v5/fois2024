import argparse
import os
import random
import shutil

from constants import rawImgDataFolder, yoloImgFolder, modelSObjsPath

baseFolder = os.path.dirname(os.path.abspath(__file__))
cwdFolder = os.getcwd()

def main():
    parser = argparse.ArgumentParser(prog='partII_trainYOLO', description='Prepare data for and then train a YOLO object detector', epilog='')
    parser.add_argument('-i', '--inputDirectory', default="", help='Folder to look in for annotated images. The annotated images are *.jpg/*.txt file pairs where the filename begins with \"seg_\".')
    parser.add_argument('-o', '--outputDirectory', default="", help='Folder to put training data in.')
    parser.add_argument('-m', '--model', default="", help="Path to store the resulting model file at.")
    parser.add_argument('-b', '--batch', default="8", help='Batch size.')
    parser.add_argument('-e', '--epochs', default="300", help='Number of epochs to train for.')
    parser.add_argument('-is', '--imageSize', default="640", help='One integer describing the size to rescale images to before feeding to yolo.')
    parser.add_argument('-p', '--patience', default="50", help="Number of epochs to wait for improvement before stopping training early.")
    arguments = parser.parse_args()
    inputDirectory = rawImgDataFolder
    if "" != arguments.inputDirectory:
        inputDirectory = arguments.inputDirectory
    outputDirectory = yoloImgFolder
    if "" != arguments.outputDirectory:
        outputDirectory = arguments.outputDirectory
    trainFolder = os.path.join(outputDirectory,"train")
    validFolder = os.path.join(outputDirectory,"valid")
    testFolder = os.path.join(outputDirectory,"test")
    str2IntMap = {}
    try:
        os.mkdir(outputDirectory)
    except FileExistsError:
        try:
            shutil.rmtree(trainFolder)
            shutil.rmtree(validFolder)
            shutil.rmtree(testFolder)
        except Exception as e:
            pass
    for e in [trainFolder, validFolder, testFolder]:
        os.mkdir(e)
        os.mkdir(os.path.join(e,"images"))
        os.mkdir(os.path.join(e,"labels"))
    allFiles = os.listdir(inputDirectory)
    filePairs = {}
    for e in allFiles:
        prefix = e[:-4]
        ext = e[-3:]
        if prefix not in filePairs:
            filePairs[prefix] = {"jpg": None, "txt": None}
        filePairs[prefix][ext] = os.path.join(inputDirectory,e)
    targetFiles = [(k, v["jpg"], v["txt"]) for k,v in filePairs.items() if (v["jpg"] is not None) and (v["txt"] is not None)]
    ## TODO: ensure fair splitting between train/valid/test folders for all labels
    random.shuffle(targetFiles)
    N = len(targetFiles)
    lastTrain = int(N*0.7)
    lastValid = int(N*0.9)
    for k,f in enumerate(targetFiles):
        pref, jpgPath, txtPath = f
        if k <= lastTrain:
            imgFolder = "train/images"
            labelsFile = "train/labels/" + pref + ".txt"
        elif k <= lastValid:
            imgFolder = "valid/images"
            labelsFile = "valid/labels/" + pref + ".txt"
        else:
            imgFolder = "test/images"
            labelsFile = "test/labels/" + pref + ".txt"
        shutil.copy2(jpgPath,os.path.join(outputDirectory, imgFolder))
        lines = open(txtPath).read().splitlines()
        linesN = []
        for l in lines:
            sp = l.find(' ')
            otype = l[:sp]
            poly = l[sp+1:]
            if otype not in str2IntMap:
                str2IntMap[otype] = len(str2IntMap)
            nl = str2IntMap[otype] + " " + poly
            linesN.append(nl)
        txt = "\n".join(linesN)
        with open(os.path.join(outputDirectory,labelsFile), "w") as outfile:
            _ = outfile.write("%s" % txt)
    yamlFile = os.path.join(outputDirectory,"data.yaml")
    with open(yamlFile, "w") as outfile:
        _ = outfile.write("train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n")
        _ = outfile.write("nc: %d\n" % len(str2IntMap))
        _ = outfile.write("names: %s\n\n" % str([x[1] for x in sorted([(v,k) for k,v in str2IntMap.items()])]))
        _ = outfile.write("roboflow:\n  workspace: none\n  project: yolo\n  version: 1\n  license: CC BY 4.0\n  url: none\n")
    model = ultralytics.YOLO('yolov8s-seg.pt')
    results = model.train(data=yamlFile, batch=int(arguments.batch), epochs=int(arguments.epochs), imgsz=int(arguments.imageSize), patience=int(arguments.patience))
    modelSrcFile = os.path.join(outputDirectory, "runs/segment/train/weights/best.pt")
    modelDestFile = modelSObjsPath
    if "" != arguments.model:
        modelDestFile = arguments.model
    shutil.copy(modelSrcFile, modelDestFile)
        
        
if "__main__" == __name__:
    main()
