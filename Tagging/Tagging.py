# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 12:55:55 2016

@author: Ramon
"""
import os
from skimage import io
import numpy as np
from Tkinter import Tk, Frame, Checkbutton, Label, Button, Radiobutton, Entry
from Tkinter import StringVar, BooleanVar, BOTH, W, E, N, S, LEFT, RIGHT, Y, X, TOP, BOTTOM, GROOVE, DISABLED, CENTER
import tkFileDialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
#import time

import os.path
if os.path.isfile('TeachersLabels.py'): 
    import TeachersLabels as lb
else:
    import Labels as lb

"""
NAME = {'key':'options_name','options_gui':[('op1','srt1')..('opn','strn')]}
NAME is one of the options in the user interface
'options_name': is the key in the options dictionary that the code will look at
'options_gui': are pairs of strings ('opt_x','str_x') where 'opt_x' is the string 
that will appear in the user interface, and 'str_x' is the string that the code 
will check out on the 'options_name' key
"""
COLORSSPACE = {'key':'colorspace','options_gui':[('RBG','RGB'),('CIE Lab', 'Lab'),('11 basic colors','ColorNaming'),('Other','Other')]}
FITTING = {'key':'fitting','options_gui':[('None','None'),('Fisher', 'Fisher'),('Other', 'Other')]}
CENTROIDSINIT = {'key':'km_init','options_gui':[('First points','first'),('Randomize', 'random'),('Other','Other')]}
SIMILARITY = {'key':'metric','options_gui':[('Basic','basic'),('Other 1','other1'),('Other 2','other2')]}
 
class Tagging(Frame):
  
    def __init__(self, parent, options):
        if options == None:
            options = {}            
        DefaultOptions = {'colorspace':'RGB', 'K':6, 'km_init':'first', 'verbose':False, 'single_thr':0.6, 'metric':'basic'}
        for op in DefaultOptions:
            if not op in options:
                options[op] = DefaultOptions[op]
        self.options = options
        self.K_ant = 0        
        self.num_im=-1
        Frame.__init__(self, parent)   
        self.parent = parent
        self.parent.title("TAGGING")        

        self.F1 = Frame(parent, width=150,borderwidth=2,relief=GROOVE)
        self.F2 = Frame(parent, width=150,borderwidth=2,relief=GROOVE)
        self.F3 = Frame(parent)
        self.F4 = Frame(parent,borderwidth=2,relief=GROOVE)
        self.F5 = Frame(parent,borderwidth=2,relief=GROOVE)
        self.F1.pack(side=LEFT, fill=Y, pady = 4, padx=4)
        self.F2.pack(side=LEFT, fill=Y, pady = 4)
        self.F3.pack(side=TOP, expand=True, fill=BOTH, pady = 4, padx = 4)
        self.F4.pack(side=TOP, fill=BOTH,  expand=0, padx = 4)
        self.F5.pack(side=BOTTOM, fill=BOTH,  expand=0, pady = 4, padx = 4)
        self.F1.pack_propagate(0)
        self.F2.pack_propagate(0)
        self.F3.pack_propagate(0)

        lbl = Label(self.F1, text="Groundtruth")
        lbl.grid(row=0,sticky=W)
        Frame(self.F1,borderwidth=1,relief=GROOVE,height=2).grid(row=1,sticky=W)

        lbl = Label(self.F2, text="Detected")
        lbl.grid(row=0,sticky=W, columnspan=2)
        Frame(self.F2,borderwidth=1,relief=GROOVE,height=2).grid(row=1,sticky=W, columnspan=2)
        
        self.filename = Button(self.F3,text="filename",command=self.ChooseFile)
        self.filename.pack(side=TOP,expand=0)

#        #matplotlib setup
#        plt.ion()
        self.figure=plt.figure(figsize=(2,2), frameon=False)
        self.axes=self.figure.add_subplot(111)
        self.Canvas=FigureCanvasTkAgg(self.figure, master=self.F3)
#        self.Canvas.show()        
        self.Canvas.get_tk_widget().pack(side=TOP,fill=BOTH,expand=1 )
        plt.axis('off')


        b = Button(self.F4,text=" <<<10 ",command=self.PreviousFast)
        b.pack(side=LEFT, padx=(20,2), pady=3)
        b = Button(self.F4,text=" <<1 ",command=self.Previous)
        b.pack(side=LEFT, pady=3)
        self.im_name = Button(self.F4,text="image",bd=0, command=self.Results)
        self.im_name.pack(side=LEFT, expand=True)        
        b = Button(self.F4,text=" 1>> ",command=self.Next)
        b.pack(side=LEFT)
        b = Button(self.F4,text=" 10>>> ",command=self.NextFast)
        b.pack(side=LEFT, padx=(2,20))

        F51 = Frame(self.F5)
        F52 = Frame(self.F5)
        F53 = Frame(self.F5)
        F51.pack(fill=BOTH, expand=1, side=LEFT)
        Frame(self.F5,borderwidth=1,relief=GROOVE,width=2).pack(side=LEFT, fill=Y)
        F52.pack(fill=BOTH, expand=1, side=LEFT)
        Frame(self.F5,borderwidth=1,relief=GROOVE,width=2).pack(side=LEFT, fill=Y)
        F53.pack(fill=BOTH, expand=1, side=LEFT)

        Label(F51, text="Color space").pack()
        Frame(F51,borderwidth=1,relief=GROOVE,height=2).pack(fill=X)
        Label(F52, text="K-Means").pack()
        Frame(F52,borderwidth=1,relief=GROOVE,height=2).pack(fill=X)
        Label(F53, text="Labelling").pack()
        Frame(F53,borderwidth=1,relief=GROOVE,height=2).pack(fill=X)
        
        self.ColorMode = StringVar()
        self.ColorMode.set(self.options[COLORSSPACE['key']]) 
        for text, mode in COLORSSPACE['options_gui']:
            b = Radiobutton(F51, text=text, variable=self.ColorMode, value=mode, command=self.Results)
            b.pack(anchor=W)        

        Frame(F51,borderwidth=1,relief=GROOVE,height=2).pack(fill=X)
        Label(F51, text="Grouping", padx=20).pack()
        Frame(F51,borderwidth=1,relief=GROOVE,height=2).pack(fill=X)
        self.Group = BooleanVar()
        self.Group.set(True)
        b = Checkbutton(F51, text='Group same label colors', variable=self.Group, command=self.Results)
        b.pack(anchor=W)
        

        F521 = Frame(F52)
        F522 = Frame(F52)

        F521.pack(fill=BOTH)
        F522.pack(fill=BOTH)
#        F523 = Frame(F52)
#        F523.pack(fill=BOTH)

        self.K = StringVar()
        self.K.set(self.options['K'])
#        f = Frame(F52)
#        f.pack(side=TOP, fill=BOTH,  expand=0)
        Label(F521, text="K : ").pack(side=LEFT)
        Entry(F521, textvariable=self.K, width=3, justify=CENTER).pack(side=LEFT, padx=4)
#        Label(F52, text="Init", padx=20).pack(anchor=W)
#        Frame(F521,borderwidth=1,relief=GROOVE,width=2).pack(side=RIGHT, fill=Y, padx=2)

        self.Fitting = StringVar()
        self.Fitting.set(self.options[FITTING['key']]) 
#        f = Frame(F52)
#        f.pack(side=RIGHT, fill=BOTH,  expand=0)
        for text, mode in FITTING['options_gui']:
            b = Radiobutton(F521, text=text, variable=self.Fitting, value=mode, command=self.Results)
            b.pack(anchor=W,padx=4)        
            
        Frame(F522,borderwidth=1,relief=GROOVE,height=2).pack(fill=X)
        Label(F522, text="Centroid Init", padx=20).pack()
        Frame(F522,borderwidth=1,relief=GROOVE,height=2).pack(fill=X)
#        IM = [('First points','first'),('Randomize', 'random'),('Distribtued','distributed'),('Other','Other')]

        self.KMInit = StringVar()
        self.KMInit.set(self.options[CENTROIDSINIT['key']]) 
        for text, mode in CENTROIDSINIT['options_gui']:
            b = Radiobutton(F522, text=text, variable=self.KMInit, value=mode, command=self.Results)
            b.pack(anchor=W)        

        
        self.Thr = StringVar()
        self.Thr.set(self.options['single_thr'])
        f = Frame(F53)
        f.pack(side=TOP, fill=BOTH,  expand=0)
        Label(f, text="single\nbasic   >\ncolor", justify=LEFT).pack(side=LEFT)
        Entry(f, textvariable=self.Thr, width=4, justify=CENTER).pack(side=LEFT)
#        self.Synonymous = BooleanVar()
#        self.Synonymous.set(self.options['synonyms'])
#        b = Checkbutton(F53, text='Synonymous', variable=self.Synonymous, command=self.Results)
#        b.pack(anchor=W)
        
        Frame(F53,borderwidth=1,relief=GROOVE,height=2).pack(fill=X)
        Label(F53, text="Similarity Metric", padx=2).pack()
        Frame(F53,borderwidth=1,relief=GROOVE,height=2).pack(fill=X)
        self.Metric = StringVar()
        self.Metric.set(self.options[SIMILARITY['key']]) 
        for text, mode in SIMILARITY['options_gui']:
            b = Radiobutton(F53, text=text, variable=self.Metric, value=mode, command=self.Results)
            b.pack(anchor=W)        
            
        
    def ChooseFile(self):
        filename = tkFileDialog.askopenfilename(initialdir='Images', filetypes=[("Text files","*.txt")])
        if filename =='':
            return
        self.ImageFolder = os.path.dirname(filename)
        self.filename['text'] = os.path.basename(filename) 
        self.GT = lb.loadGT(filename)
        self.num_im = 0
        self.Results()

    def Previous(self):
        self.num_im -= 1
        if self.num_im<-1:
            self.num_im=0
        elif self.num_im<0:
            self.num_im=0
            return            
        self.Results()       

    def PreviousFast(self):
        self.num_im -= 10
        if self.num_im<0:
            self.num_im=0
        self.Results()       

    def Next(self):
        self.num_im += 1
        if self.num_im>=len(self.GT):
            self.num_im=len(self.GT)-1
            return
        self.Results()       

    def NextFast(self):
        self.num_im += 10
        if self.num_im>=len(self.GT):
            self.num_im=len(self.GT)-1
        self.Results()       

    def Results(self):        
        if self.num_im<0 or self.num_im>=len(self.GT):
            return
        #self.parent.config(cursor="wait")
        self.parent.update()
        
        self.im_name['text'] = self.GT[self.num_im][0]
        im = io.imread(self.ImageFolder+"/"+self.GT[self.num_im][0])    
        
        self.options['K']          = int(self.K.get())
        self.options['single_thr'] = float(self.Thr.get())
#        self.options['synonyms']   = self.Synonymous.get()
        self.options[COLORSSPACE['key']] = self.ColorMode.get()
        self.options[CENTROIDSINIT['key']]    = self.KMInit.get()
        self.options[SIMILARITY['key']]     = self.Metric.get()
        self.options[FITTING['key']]    = self.Fitting.get()

        if self.options['fitting'] != 'None':
            if self.K_ant==0:
                self.K_ant = self.options['K']
            self.options['K'] = 0
            self.K.set(0)
            pass
        else:
            if self.K_ant>0:
                self.K.set(self.K_ant)
                self.options['K'] = self.K_ant
                self.K_ant = 0

        self.labels, self.which, self.km = lb.processImage(im, self.options)
        if self.options['K'] == 0:
            self.K.set(self.km.K)

        if not self.Group.get():
            self.labels = [w for ww in [[w]*len(z) for w,z in zip(self.labels,self.which)] for w in ww]
            self.which = [[w] for ww in self.which for w in ww]
        
 
       # get percentages for every label
        percent = np.zeros((len(self.which),1))
        for i,w in enumerate(self.which):
            for k in w:
                percent[i] += np.sum(self.km.clusters==k)
        percent = percent*100/self.km.clusters.size

       # get color for every label from image
        
        sh = im.shape;
        im = np.reshape(im, (-1, im.shape[2]))
        colors = np.zeros((len(self.which),3))
        for i,w in enumerate(self.which):
            for k in w:
                colors[i] += np.mean(im[self.km.clusters==k,:],0)
            colors[i] /= len(w)
        im = np.reshape(im, sh)
        self.im = im
        
        self.CBgt = self.CreateCheckList(self.F1, self.GT[self.num_im][1])
        self.CBdetected = self.CreateCheckList(self.F2, self.labels, percent, colors)
        l = Label(self.F2,text='similarity: '+'%.2f' % (lb.similarityMetric(self.labels, self.GT[self.num_im][1], self.options)*100)+'%')
        l.grid(columnspan=2,sticky=W+E+N+S)
        self.F2.grid_rowconfigure(self.F2.grid_size()[1]-1, weight=1)

        self.CheckLabels()
        #self.parent.config(cursor="")
        self.parent.update()
        self.Segment()
        
#        self.ShowImage(im)
        
    def Segment(self):
        all_widgetes = self.F2.winfo_children()
        cb = [w for w in all_widgetes if w.winfo_class() == 'Checkbutton']
        lab_colors = np.zeros((len(cb),3))
        for i,c in enumerate(cb):
            lab_colors[i,0] = int(c.cget('bg')[1:3], 16)
            lab_colors[i,1] = int(c.cget('bg')[3:5], 16)
            lab_colors[i,2] = int(c.cget('bg')[5:7], 16)
        ims = np.empty_like(self.im)
        ims = np.reshape(ims, (-1, ims.shape[2]))
        for i,w in enumerate(self.which):
            for k in w:
                ims[self.km.clusters==k]=lab_colors[i]
        ims = np.reshape(ims, self.im.shape)
        self.ShowImage(np.hstack((ims,self.im)))
            
    def ShowImage(self,im):
        self.axes.imshow(im)
        plt.axis('off')
        self.Canvas.draw()        
        
    def CreateCheckList(self, frame, labels, values=None, colors=None):
       
        all_widgetes = [w for w in frame.winfo_children() if w.winfo_class() == 'Checkbutton']
        for w in all_widgetes:
            w.destroy()
        all_widgetes = [w for w in frame.winfo_children() if w.winfo_class() == 'Label']
        for w in all_widgetes[1:]:
            w.destroy()
        var = []
        if values is not None:
            add_text = ["%5.1f" % (values[i]) +'%' for i in range(len(values))]

        for i in range(len(labels)):
            var.append(BooleanVar())
            
            fgc = "black"
            if colors is not None:
                mycolor = '#%02x%02x%02x' % (colors[i,0], colors[i,1], colors[i,2])
                if (colors[i,0]+colors[i,1]+colors[i,2])/3/255<=0.5:
                    fgc = "#FFFFFF"
            else:
                mycolor = frame.cget('bg')
                
            cb = Checkbutton(frame, text=labels[i], state=DISABLED, variable=var[i], bg=mycolor, disabledforeground=fgc, anchor=W)
#            cb.pack(anchor=W, fill=X, padx=2)
            cb.grid(row=i+2,column=0,sticky=W+E)
            if values is not None:
                cb = Label(frame, text=add_text[i], bg=mycolor, fg=fgc, anchor=E)
                cb.grid(row=i+2,column=1,sticky=W+E+N+S)
            frame.grid_rowconfigure(i+2, weight=0)
                
        
        return var

    def CheckLabels(self):
        for i in range(len(self.labels)):
            if self.labels[i] in self.GT[self.num_im][1]:
                self.CBdetected[i].set(True)

        for i in range(len(self.GT[self.num_im][1])):
            if self.GT[self.num_im][1][i] in self.labels:
                self.CBgt[i].set(True)
            

def main():
  
    root = Tk()
    root.geometry("700x500+10+10")
    
    options = {'colorspace':'RGB', 'K':6, 'km_init':'first', 'verbose':False, 'single_thr':0.6, 'metric':'basic', 'fitting':'None'}
    app = Tagging(root, options)

    app.ImageFolder = "Images"
    app.filename['text'] = "LABELSsmall.txt"
    app.GT = lb.loadGT(app.ImageFolder + '/' + app.filename['text'])

    app.num_im=0
    app.Results()
    
    root.mainloop()
    plt.close("all")

    

if __name__ == '__main__':
    main()  