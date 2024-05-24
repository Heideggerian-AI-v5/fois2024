import itertools
import networkx
import cv2 as cv
import os
import time

import silkie
from dbgvis import startVisualizationThread, stopVisualizationThread, debugDataLock, debugData
from objectrecognition import startReceptionThread, stopReceptionThread, startRecognitionThread, stopRecognitionThread
from perception import perceptionQueries, perceptionQuestionsLock, perceptionResults, perceptionResultsLock, perceptionReady, startPerceptionThread, stopPerceptionThread
from utils import makeStartStopFns
from constants import perceptionInterpretationTheoryFile, updateSchemasTheoryFile, connQueryTheoryFile, closureTheoryFile,  schemaInterpretationTheoryFile, updateQuestionsTheoryFile, backgroundFactsFile, transportSmallPottedPlantFile

reasoningGVar= {}

def closenessQuery(graph, source, target):
    """
    If source and target are connected: returns a list of nodes that are closer to the target than source.
    If no path, returns [].
    """
    try:
        lengths = networkx.shortest_path_length(graph, target=target)
    except networkx.NodeNotFound:
        #print("NF")
        return []
    if source not in lengths:
        #print("NL")
        return []
    #print(lengths)
    return [k for k,v in lengths.items() if v < lengths[source]] + [source]

def necessaryVertexQuery(graph, source, target):
    """
    If source and target are connected: returns a list of nodes that all paths from source to target must pass through.
    If source and target are not connected, returns [].
    """
    def _weight(bigWeight, forbiddenV, startV, targetV, attributes):
        if forbiddenV == startV:
            return bigWeight
        return 1
    try:
        networkx.shortest_path_length(graph, source=source, target=target)
    except networkx.exception.NetworkXNoPath:
        return []
    N = graph.number_of_nodes()
    retq = []
    for e in graph.nodes:
        if (e != source) and (e != target):
            if N <= networkx.shortest_path_length(graph, source=source, target=target, weight=(lambda s,t,a: _weight(N,e,s,t,a))):
                retq.append(e)
    return retq

def copyFacts(a):
    retq = {k: silkie.PFact(k) for k in a}
    for k in retq:
        _ = [retq[k].addFact(t[1], t[2], silkie.DEFEASIBLE) for t in a[k].getTriples()]
    return retq

def mergeFacts(a,b):
    for k in b:
        if k not in a:
            a[k] = silkie.PFact(k)
        _ = [a[k].addFact(t[1], t[2], silkie.DEFEASIBLE) for t in b[k].getTriples()]
    return a

def triples2Facts_internal(ts, prefix=None):
    if prefix is None:
        prefix = ""
    retq = {}
    for t in ts:
        p,s,o = t
        if p.startswith("-"):
            p = "-" + prefix + p[1:]
        else:
            p = prefix + p
        if p not in retq:
            retq[p] = silkie.PFact(p)
        retq[p].addFact(s, o, silkie.DEFEASIBLE)
    return retq

def triples2Facts(ts, prefix=None):
    if isinstance(ts,dict):
        retq = {}
        for k,v in ts.items():
            nretq = triples2Facts_internal(v,prefix)
            for nk,nv in nretq.items():
                if nk not in retq:
                    retq[nk] = silkie.PFact(nk)
                _ = [retq[nk].addFact(t[1], t[2], silkie.DEFEASIBLE) for t in nv.getTriples()]
        return retq
    else:
        return triples2Facts_internal(ts,prefix)

def conclusions2Facts(cs):
    return triples2Facts(cs.defeasiblyProvable)

def conclusions2Graph(cs):
    retq = networkx.DiGraph()
    _ = [retq.add_edge(t[1],t[2]) for t in cs.defeasiblyProvable if "connQueryEdge" == t[0]]
    #_ = [print((t[1],t[2])) for t in cs.defeasiblyProvable if "connQueryEdge" == t[0]]
    return retq

def reifyConclusions(cs):
    stets = [(t[0][len("stet_"):], t[1], t[2]) for t in cs.defeasiblyProvable if t[0].startswith("stet_")]
    reifiables = sorted([t for t in cs.defeasiblyProvable if t[0].startswith("reifiable_")])
    reifieds = []
    for k,r in enumerate(reifiables):
        p, s, o = r
        p = p[len("reifiable_"):]
        reification = "reification%d"%k
        reifieds += [("isA", reification, p), ("hasS", reification, s), ("hasO", reification, o)]
    return triples2Facts(stets + reifieds)

def makePerceptionQueries(cs):
    def peekElement(s):
        for e in s:
            return e
    retq = {"relativeMovements": [], "contacts": [], "closeness": [], "unavoidables": []}
    msgs = {"wrongDirections": [], "dangerousApproaching": [], "dangerousDistancing": [], "dangerousStillness": []}
    cpts = {"participants": []}
    queries = [t[1] for t in cs.defeasiblyProvable if ("isA" == t[0]) and ("Query" == t[2])]
    messages = [t[1] for t in cs.defeasiblyProvable if ("isA" == t[0]) and ("Message" == t[2])]
    captures = [t[1] for t in cs.defeasiblyProvable if ("isA" == t[0]) and ("DataCapture" == t[2])]
    queries = {k: {} for k in queries}
    messages = {k: {} for k in messages}
    captures = {k: {} for k in captures}
    print("!!!", queries, messages, captures)
    for t in cs.defeasiblyProvable:
        p,s,o=t
        for d in [queries, messages, captures]:
            if s in d:
                if p not in d[s]:
                    d[s][p] = set()
                d[s][p].add(o)
    for qd in queries.values():
        qlist = peekElement(qd["forQueryList"])
        oq = qd.get("hasO")
        if {""} == oq:
            oq = [None]
        descriptions = [qd["hasS"], oq]
        if qlist not in ["relativeMovements", "contacts"]:
            descriptions += [qd["hasP"],qd["hasQS"],qd["hasQueryType"]]
        for description in itertools.product(*descriptions):
            retq[qlist].append(description)
    for md in messages.values():
        qlist = peekElement(md["forMessageList"])
        descriptions = [md["hasS"], md.get("hasO")]
        for description in itertools.product(*descriptions):
            msgs[qlist].append(description)
    for cd in captures.values():
        for fn, o, c in itertools.product(cd["snapshotFn"], cd["snapshotO"], cd["snapshotC"]):
            cpts["participants"].append((fn,o,c))
    return retq, msgs, cpts

def displayMessages(messages):
    printed = False
    for m in messages.get("wrongDirections", []):
        s,o = m
        print("WDIR: %s is not moving towards %s" % (s,o))
        printed = True
    for m in messages.get("dangerousApproaching", []):
        s,o = m
        if s!=o:
            print("DAPP: %s approaches %s but should not" % (s,o))
            printed = True
    for m in messages.get("dangerousDistancing", []):
        s,o = m
        if s!=o:
            print("DDEP: %s departs from %s but should not" % (s,o))
            printed = True
    for m in messages.get("dangerousStillness", []):
        s,o = m
        if s!=o:
            print("DSTL: %s stays still relative to %s but should not" % (s,o))
            printed = True
    if printed:
        print("=======")
    return

def storeTrainingFrames(captures, image, contactMasks, storeAt):
    def _haveSomething(captures, contactMasks):
        return any([((c,o,o) in contactMasks) or ((o,c,o) in contactMasks) for _,o,c in captures["participants"]])
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
                _ = outfile.write("%s %s\n" % (label, pstr))
    if not _haveSomething(captures, contactMasks):
        return
    fnamePrefix = os.path.join(storeAt, "seg_%s" % time.asctime().replace(" ", "_").replace(":","_"))
    image.save(fnamePrefix + ".jpg")
    with open(fnamePrefix + ".txt", "w") as outfile:
        for fn, o, c in captures["participants"]:
            label = o+fn
            mask = contactMasks.get((o,c,o))
            if mask is None:
                mask = contactMasks.get((c,o,o))
            if mask is not None:
                _objPolygons(label, mask, outfile)

def queryDesc2Facts(qd):
    source, target, _, _, qtype = qd
    triples = [("isA", "runningQuery", qtype), ("hasSource", "runningQuery", source), ("hasTarget", "runningQuery", target)]
    return triples2Facts(triples)

def reasoning(dbg=False, persistentSchemas=None):
    '''
    Stages:
      observations (tuples) + previous persistent schemas (dfl facts) + theory -> reifiable relations, stet relations (triples)
      reifiable/stet relations (triples) + theory -> new persistent schemas (dfl facts)
      new persistent schemas (dfl facts) + theory -> connquery graph edges (triples)
      connquery -> connquery results (triples)
      add connquery results to persistent schemas
      new persistent schemas (dfl facts) + theory -> reifiable questions, stet relations (triples)
      reifiable questions/stet relations + theory -> new questions (tuples)
    '''
    storeAt=persistentSchemas["storeAt"].getTriples()[0][1]
    perceptionQueriesLocal = {"relativeMovements": [], "contacts": [], "closeness": [], "unavoidables": []}
    perceptionResultsLocal = {"relativeMovements": [], "contacts": []}
    perceptionInterpretationTheory = silkie.loadDFLRules(perceptionInterpretationTheoryFile)
    updateSchemasTheory = silkie.loadDFLRules(updateSchemasTheoryFile)
    connQueryTheory = silkie.loadDFLRules(connQueryTheoryFile)
    closureTheory = silkie.loadDFLRules(closureTheoryFile)
    schemaInterpretationTheory = silkie.loadDFLRules(schemaInterpretationTheoryFile)
    updateQuestionsTheory = silkie.loadDFLRules(updateQuestionsTheoryFile)
    backgroundFacts = silkie.loadDFLFacts(backgroundFactsFile)
    while reasoningGVar.get("keepOn"):
        with perceptionReady:
            perceptionReady.wait()
        perceptionResultsLock.acquire()
        perceptionResultsLocal = {k: v for k,v in perceptionResults.items()}
        image = perceptionResultsLocal.pop("image")
        contactMasks = perceptionResultsLocal.pop("contactMasks")
        perceptionResultsLock.release()
        ## observations (tuples) + prev persistent schemas (dfl facts) + theory -> reifiable/stet relations (triples)
        #print(perceptionResultsLocal)
        #perceptionResultsLocal["relativeMovements"].append(("departs", "SmallPottedPlant", "TallChair"))
        facts = mergeFacts(persistentSchemas, triples2Facts(perceptionResultsLocal, "observed_"))
        theory, _, i2s, _ = silkie.buildTheory(perceptionInterpretationTheory,facts,backgroundFacts)
        conclusions = silkie.idx2strConclusions(silkie.dflInference(theory), i2s)
        #print("===PI\n",sorted(conclusions.defeasiblyProvable))
        ## reifiable/stet relations (triples) + theory -> new persistent schemas (dfl facts)
        facts = reifyConclusions(conclusions)
        #print("-----\n", [sorted(x.getTriples()) for x in facts.values()])
        theory, _, i2s, _ = silkie.buildTheory(updateSchemasTheory,facts,backgroundFacts)
        conclusions = silkie.idx2strConclusions(silkie.dflInference(theory), i2s)
        #print("===US\n",sorted(conclusions.defeasiblyProvable))
        persistentSchemas = conclusions2Facts(conclusions)
        gatheredResults = {}
        for qk,fn in [("closeness",closenessQuery), ("unavoidables",necessaryVertexQuery)]:
            if qk in perceptionQueriesLocal:
                for q in perceptionQueriesLocal[qk]:
                    source, target, p, s, qtype = q
                    ## new persistent schemas (dfl facts) + theory -> connquery graph edges (triples)
                    facts = mergeFacts(copyFacts(persistentSchemas), queryDesc2Facts(q))
                    theory, _, i2s, _ = silkie.buildTheory(connQueryTheory,facts,backgroundFacts)
                    conclusions = silkie.idx2strConclusions(silkie.dflInference(theory), i2s)
                    #print("===CQ\n",sorted(conclusions.defeasiblyProvable))
                    ## connquery -> connquery results (triples)
                    graph = conclusions2Graph(conclusions)
                    triples = [(p, s, e) for e in fn(graph, source, target)]
                    #print(qk,q,triples)
                    gatheredResults = mergeFacts(gatheredResults, triples2Facts(triples))
        ## add connquery results to persistent schemas
        persistentSchemas = mergeFacts(persistentSchemas, gatheredResults)
        theory, _, i2s, _ = silkie.buildTheory(closureTheory,persistentSchemas,backgroundFacts)
        conclusions = silkie.idx2strConclusions(silkie.dflInference(theory), i2s)
        #print("===CT\n",sorted(conclusions.defeasiblyProvable))
        persistentSchemas = conclusions2Facts(conclusions)
        ## new persistent schemas (dfl facts) + theory -> reifiable questions, stet relations (triples)
        theory, _, i2s, _ = silkie.buildTheory(schemaInterpretationTheory,persistentSchemas,backgroundFacts)
        conclusions = silkie.idx2strConclusions(silkie.dflInference(theory), i2s)
        #print("===SI\n",sorted(conclusions.defeasiblyProvable))
        ## reifiable questions/stet relations + theory -> new questions (tuples)
        facts = reifyConclusions(conclusions)
        theory, _, i2s, _ = silkie.buildTheory(updateQuestionsTheory,facts,backgroundFacts)
        conclusions = silkie.idx2strConclusions(silkie.dflInference(theory), i2s)
        #print("===UQ\n",sorted(conclusions.defeasiblyProvable))
        perceptionQueriesLocal, messages, captures = makePerceptionQueries(conclusions)
        #print(perceptionQueriesLocal, messages)
        displayMessages(messages)
        storeTrainingFrames(captures, image, contactMasks, storeAt)
        perceptionQuestionsLock.acquire()
        for k,v in perceptionQueriesLocal.items():
            perceptionQueries[k] = v
        perceptionQuestionsLock.release()
    return

startReasoningThread, stopReasoningThread = makeStartStopFns(reasoningGVar, reasoning)

if "__main__" == __name__:
    transportSmallPottedPlant = silkie.loadDFLFacts(transportSmallPottedPlantFile)
    startVisualizationThread()
    startReceptionThread()
    startRecognitionThread(dbg=True)
    startPerceptionThread(dbg=True)
    startReasoningThread(dbg=True, persistentSchemas=transportSmallPottedPlant)
    input("Reception/Recognition, Perception, and Reasoning threads running. Switch to the turtlebot GUI and move it around to see the changing segmentation masks and optical flow/contact computations. When you want to exit this program, press ENTER in this terminal.")
    stopReasoningThread()
    stopPerceptionThread()
    stopRecognitionThread()
    stopReceptionThread()
    print("Threads stopped, exiting.")
    stopVisualizationThread()

