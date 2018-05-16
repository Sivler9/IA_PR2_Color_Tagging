# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 10:31:35 2017

@author: ramon
"""
import sys

reload(sys)
sys.setdefaultencoding('utf8')

import json
from skimage import io
from skimage.transform import rescale
import numpy as np
import ColorNaming as cn

import os.path
if os.path.isfile('TeachersLabels.py') and True:
    student = False
    import TeachersLabels as lb
    import TeachersKMeans as km
else:
    student = True
    import Labels as lb
    import KMeans as km

TestFolder = 'Test/'
ImageFolder = 'Images/'
if not os.path.isdir(TestFolder):
    os.makedirs(TestFolder)

def TestInfo(Test,Options,GTFile, NImage):
    global student
    File = TestFolder + '%02d'%Test + 'Test' + '.txt'
    if student:
        with open(File) as infile:
            data = json.load(infile)
        Options = data['o']
        GTFile = data['f']
        NImage = data['i']
    else:
        with open(File, 'w') as outfile:
            json.dump({'o':Options,'f':GTFile,'i':NImage}, outfile, ensure_ascii=False)
    return Options,GTFile, NImage

def TestFinalCheck(lab1=0,ind1=0,lab2=0,ind2=0):
    global student
    File = TestFolder + 'final.txt'
    if student:
        with open(File) as infile:
            data = json.load(infile)
        l1 = data['l1']
        i1 = data['i1']
        l2 = data['l2']
        i2 = data['i2']
        print "\n\nlast own check =============================================================="
        print " YOUR RESULT:"
        print lab1
        print ind1
        print lab2
        print ind2
        print "\n DESIRED RESULT:"
        print l1
        print i1
        print l2
        print i2
    else:
        with open(File, 'w') as outfile:
            json.dump({'l1':lab1,'i1':ind1,'l2':lab2, 'i2':ind2}, outfile, ensure_ascii=False)
    return lab1,ind1,lab2,ind2

def PrintTestResult(Mess, s, t, ok):
    print     '========================== '+ Mess + ' =========================='
    if ok:
        print '                              SUCCESFUL'
        print '==============================================================================\n'
    else:
        print '                                FAIL!!!'
        print '==============================================================================\n'
        print 'your result >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
        print s
        print 'desired result <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'
        print t
        print '\n\n'

def CheckTest(Message, D, File, student):
    same = 0
    if student:
        with open(File) as infile:
            DT = json.load(infile)
        try:
            if type(D) is list:
                same = sum([d in DT for d in D])==len(DT)
            else:
                if type(D) is float:
                    D = np.array(D)
                DT = np.array(DT)
                same = np.allclose(D,DT,rtol=0.0001,atol=0)
        except:
            pass

        PrintTestResult(Message, D, DT, same)
    else:
        if type(D) is not list and type(D) is not float:
            D = D.tolist()
        with open(File, 'w') as outfile:
            json.dump(D, outfile, ensure_ascii=False)
    return same

######################################################################################################
def TestSolution(Test, Options, GTFile, NImage, tests):
    global student
    Options, GTFile, NImage = TestInfo(Test, Options, GTFile, NImage)
    ######################################################################################################
    GT = lb.loadGT(ImageFolder + GTFile)

    im = io.imread(ImageFolder + GT[NImage][0])
#    im = rescale(im, 0.7, preserve_range=True)

    Messages = []
    Results = []
    print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
    print '!!!!!!!!!!!!!!!!!!!   TEST '+str(Test)+'   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
    print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    if 1 in tests:
        Message = '-1 testing data init'

        File = TestFolder + '%02d'%Test + Message + '.txt'

        k_m = km.KMeans(im, Options['K'], Options)
        D=k_m.X[:10]
        Results.append(CheckTest(Message, D, File, student))
        Messages.append(Message)

    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    if 2 in tests:
        Message = '-2 testing distance function'

        File = TestFolder + '%02d'%Test + Message + '.txt'

        X=np.arange(50,0,-0.5).reshape(-1,4)
        C=np.arange(6,0,-0.5).reshape(-1,4)
        D = km.distance(X,C)
        Results.append(CheckTest(Message, D, File, student))
        Messages.append(Message)

    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    if 3 in tests:
        Message = '-3 testing cluster grouping'

        File = TestFolder + '%02d'%Test + Message + '.txt'

        k_m = km.KMeans(im, Options['K'], Options)
        k_m._cluster_points()
        D=k_m.clusters[:100]
        Results.append(CheckTest(Message, D, File, student))
        Messages.append(Message)

    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    if 4 in tests:
        Message = '-4 testing centroid update'

        File = TestFolder + '%02d'%Test + Message + '.txt'

        k_m = km.KMeans(im, Options['K'], Options)
        k_m._iterate()
        D=k_m.centroids
        Results.append(CheckTest(Message, D, File, student))
        Messages.append(Message)

    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    if 5 in tests:
        Message = '-5 testing centroid convergence'

        File = TestFolder + '%02d'%Test + Message + '.txt'

        k_m = km.KMeans(im, Options['K'], Options)
        k_m.run()
        D=k_m.centroids
        Results.append(CheckTest(Message, D, File, student))
        Messages.append(Message)

    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    if 6 in tests:
        Message = '-6 testing color labels extraction (simple labels)'

        File = TestFolder + '%02d'%Test + Message + '.txt'

        k_m = km.KMeans(im, Options['K'], Options)
        k_m.run()
        Options['single_thr']=0

        if k_m.centroids.shape[1]==3:
            k_m.centroids = cn.ImColorNamingTSELabDescriptor(k_m.centroids)
        lab,_ = lb.getLabels(k_m, Options)
        Results.append(CheckTest(Message, lab, File, student))
        Messages.append(Message)

    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    if 7 in tests:
        Message = '-7 testing color labels extraction (compound labels)'
        File = TestFolder + '%02d'%Test + Message + '.txt'

        k_m = km.KMeans(im, Options['K'], Options)
        k_m.run()
        Options['single_thr']=1

        if k_m.centroids.shape[1]==3:
            k_m.centroids = cn.ImColorNamingTSELabDescriptor(k_m.centroids)
        lab,_ = lb.getLabels(k_m, Options)
        Results.append(CheckTest(Message, lab, File, student))
        Messages.append(Message)

    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    if 8 in tests:
        Message = '-8 testing color labels extraction (0.6)'
        File = TestFolder + '%02d'%Test + Message + '.txt'

        k_m = km.KMeans(im, Options['K'], Options)
        k_m.run()
        Options['single_thr']=0.6

        if k_m.centroids.shape[1]==3:
            k_m.centroids = cn.ImColorNamingTSELabDescriptor(k_m.centroids)
        lab,ind = lb.getLabels(k_m, Options)
        Results.append(CheckTest(Message, lab, File, student))
        Messages.append(Message)

    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    if 9 in tests:
        Message = '-9 testing process image function'
        File = TestFolder + '%02d'%Test + Message + '.txt'
        Options['single_thr'] *= 1.10

        lab,ind,k_m = lb.processImage(im, Options)
        Results.append(CheckTest(Message, lab, File, student))
        Messages.append(Message)

    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    if 10 in tests:
        Message = '-10 testing similarity metric 100'
        File = TestFolder + '%02d'%Test + Message + '.txt'

        import random
        A = GT[NImage][1]
        B = GT[NImage][1][:]
        random.shuffle(B)
        D = lb.similarityMetric(A,B, Options)
        Results.append(CheckTest(Message, D, File, student))
        Messages.append(Message)

    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    if 11 in tests:
        Message = '-11 testing similarity metric 2'
        File = TestFolder + '%02d'%Test + Message + '.txt'

        import random
        A = GT[NImage][1]
        B = GT[NImage+2][1][:]
        random.shuffle(B)
        D = lb.similarityMetric(A,B, Options)
        Results.append(CheckTest(Message, D, File, student))
        Messages.append(Message)

    if student:
        print "\n\n\n                SUMMARY  TEST " + str(Test)
        for i in range(len(Messages)):
            print Messages[i] + "    " + ("OK" if Results[i] else "FAIL")
        print "\n\n"
        return sum(Results),len(Results)
    return 1,1

######################################################################################################
import time
t=time.time()
GTFile = 'LABELSlarge.txt'
Options = {'colorspace':'RGB', 'K':6, 'km_init':'first', 'fitting':'Fisher', 'single_thr':0.6, 'metric':'basic', 'verbose':False}
score=[]
score.append(TestSolution(1, Options, GTFile, 11, [1,2,3,4,5,6,7,8,9,10,11]))
score.append(TestSolution(2, Options, GTFile, 123, [1,3,4,5,6,7,8,9,10,11]))
Options = {'colorspace':'ColorNaming', 'K':3, 'km_init':'first', 'fitting':'Fisher', 'single_thr':0.6, 'metric':'basic', 'verbose':False}
score.append(TestSolution(3, Options, GTFile, 143, [1,3,4,5,6,7,8,9,10,11]))

Options = {'colorspace':'Lab', 'K':3, 'km_init':'first', 'fitting':'Fisher', 'single_thr':0.6, 'metric':'basic', 'verbose':False}
score.append(TestSolution(4, Options, GTFile, 43, [1,3,4,5,6,9,10,11]))

Options = {'colorspace':'HSV', 'K':3, 'km_init':'first', 'fitting':'Fisher', 'single_thr':0.6, 'metric':'basic', 'verbose':False}
score.append(TestSolution(5, Options, GTFile, 0, [1,3,4,5,6,9,10,11]))

GT = lb.loadGT(ImageFolder + GTFile)
im = io.imread(ImageFolder + GT[0][0])
im = rescale(im, 0.5, preserve_range=True)

Final = sum([x[0] for x in score])
Over = sum([x[1] for x in score])

print "NIUs: ",lb.NIUs()
print "Final Score: %d / %d   Final mark: %f"%(Final,Over, 5.0*Final/Over)

Options = {'colorspace':'RGB', 'K':6, 'km_init':'random', 'fitting':'Fisher', 'single_thr':0.6, 'metric':'basic', 'verbose':False}
lab1,ind1,k_m = lb.processImage(im, Options)
Options = {'colorspace':'HSV', 'K':0, 'km_init':'first', 'fitting':'Fisher', 'single_thr':0.6, 'metric':'basic', 'verbose':False}
lab2,ind2,k_m = lb.processImage(im, Options)
TestFinalCheck(lab1,ind1,lab2,ind2)
print "\n\nyour code lasted for %f seconds. Teachers version lasted for 10 seconds"%(time.time()-t)
