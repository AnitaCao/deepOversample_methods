import os
import numpy as np
import fashion_mnist_suppli as spp
import fashion_mnist_net as nt
from keras.layers import Input
from keras.models import Model
from tensorflow.keras.optimizers import Adam
#from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

gamopth = '/home/tcvcs/VSCODE/GAMO/'

fileName=['fMnist_100_trainData.csv', 'fMnist_100_testData.csv']
fileStart='fMnist_100_Gamo'
fileEnd, savePath='_Model.h5', gamopth+fileStart+'/'
adamOpt=Adam(0.0002, 0.5)
latDim, modelSamplePd, resSamplePd=100, 5000, 500

batchSize, max_step=32, 50000

trainS, labelTr=spp.fileRead(gamopth + fileName[0])
testS, labelTs=spp.fileRead(gamopth + fileName[1])

n, m=trainS.shape[0], testS.shape[0] #n=9000, m=1000

trainS, testS=(trainS-127.5)/127.5, (testS-127.5)/127.5 #normalization
trainS, testS=np.reshape(trainS, (n, 28, 28, 1)), np.reshape(testS, (m, 28, 28, 1))

labelTr, labelTs, c, pInClass, _=spp.relabel(labelTr, labelTs) #c=10
#pInClass: [40   60  100  200  350  500  750 1000 2000 4000]

imbalancedCls, toBalance, imbClsNum, ir=spp.irFind(pInClass, c) #imbClsNum=9

labelsCat=to_categorical(labelTr)

shuffleIndex=np.random.choice(np.arange(n), size=(n,), replace=False)
trainS=trainS[shuffleIndex]
labelTr=labelTr[shuffleIndex]
labelsCat=labelsCat[shuffleIndex]
classMap=list()
for i in range(c):
    classMap.append(np.where(labelTr==i)[0])

#create MLP model: output a 10-dimensional vector of class probabilities
mlp=nt.fMnistGamoMlpCreate()
mlp.compile(loss='mean_squared_error', optimizer=adamOpt)
mlp.trainable=False

# The model is designed to process 28x28 grayscale images through convolutional 
# and pooling layers to extract features, which are then flattened and passed
#  through a dense layer to produce a 512-dimensional output vector. 
gamoConv=nt.fMnistGamoConvCreate()
ip1=Input(shape=(28, 28, 1))
conv_mlp=Model(inputs=gamoConv.inputs, outputs=mlp(gamoConv.outputs))
conv_mlp.compile(loss='mean_squared_error', optimizer=adamOpt)

#The discriminator model takes two inputs—image data and labels—and produce a single 
# probability value. This output represents the model's assessment of the 
# input's authenticity, with values closer to 1 indicating "real" data and 
# values closer to 0 indicating "fake" or generated data.
dis=nt.fMnistGamoDisCreate()
dis.compile(loss='mean_squared_error', optimizer=adamOpt)
dis.trainable=False

#generator gen: takes a noise and generate a 64 dimensional vector
gen=nt.fMnistGamoGenCreate(latDim) 

gen_processed, genP_mlp, genP_dis=list(), list(), list()
for i in range(imbClsNum):
    dataMinor=trainS[classMap[i], :]
    numMinor=dataMinor.shape[0] #number of samples in the current minority class
    gen_processed.append(nt.fMnistGenProcessCreate(numMinor))

    ip1=Input(shape=(latDim,))
    ip2=Input(shape=(c,))
    ip3=Input(shape=(512, numMinor))
    op1=gen([ip1, ip2]) #generate a 64-dimensional vector from input with shape 100 by 10
    op2=gen_processed[i]([op1, ip3])
    op3=mlp(op2)
    genP_mlp.append(Model(inputs=[ip1, ip2, ip3], outputs=op3))
    genP_mlp[i].compile(loss='mean_squared_error', optimizer=adamOpt)#genP_mlp is the list of classifiers

    ip1=Input(shape=(latDim,))
    ip2=Input(shape=(c,))
    ip3=Input(shape=(512, numMinor))
    ip4=Input(shape=(c,))
    op1=gen([ip1, ip2])
    op2=gen_processed[i]([op1, ip3])
    op3=dis([op2, ip4])
    genP_dis.append(Model(inputs=[ip1, ip2, ip3, ip4], outputs=op3))
    genP_dis[i].compile(loss='mean_squared_error', optimizer=adamOpt)# the list of discriminators

batchDiv, numBatches, bSStore=spp.batchDivision(n, batchSize)
genClassPoints=int(np.ceil(batchSize/c)) #number of points to generate for each class in each batch

inClassPoints=list()
inClassPoints=[None]*imbClsNum

validA=np.ones((genClassPoints, 1))

if not os.path.exists(fileStart):
    os.makedirs(fileStart)
iter=np.int(np.ceil(max_step/resSamplePd)+1)
acsaSaveTr, gmSaveTr, accSaveTr=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
acsaSaveTs, gmSaveTs, accSaveTs=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
confMatSaveTr, confMatSaveTs=np.zeros((iter, c, c)), np.zeros((iter, c, c))
tprSaveTr, tprSaveTs=np.zeros((iter, c)), np.zeros((iter, c))

step=0
max_step_ = 2
while step<max_step_:

    conv_mlp.fit(trainS, labelsCat, batch_size=batchSize, verbose=0)
    trainSFeatures=gamoConv.predict(trainS, batch_size=500)
    for i1 in range(imbClsNum):
        inClassPoints[i1]=np.expand_dims(np.transpose(trainSFeatures[classMap[i1]]), axis=0)

    for j in range(numBatches):
        x1, x2=batchDiv[j, 0], batchDiv[j+1, 0] #x1=0, x2=32, index of the first and the next batch
        validR=np.ones((bSStore[j, 0],1))-np.random.uniform(0,0.1, size=(bSStore[j, 0], 1))
        mlp.train_on_batch(trainSFeatures[x1:x2], labelsCat[x1:x2])#trainSFeatures is the output of the convolutional layer, has shape (9000, 512)
        dis.train_on_batch([trainSFeatures[x1:x2], labelsCat[x1:x2]], validR)

        invalid=np.zeros((bSStore[j, 0], 1))+np.random.uniform(0, 0.1, size=(bSStore[j, 0], 1))
        randNoise=np.random.normal(0, 1, (bSStore[j, 0], latDim)) #shape (4, 100)
        fakeLabel=spp.randomLabelGen(toBalance, bSStore[j, 0], c) #generate random labels, shape (32, 10), one-hot encoded
        rLPerClass=spp.rearrange(fakeLabel, imbClsNum) #rLPerClass is a list of arrays, each array contains the indices of the samples that belong to the corresponding class
        fakePoints=np.zeros((bSStore[j, 0], 512)) #initialize the fakePoints array,same shape as trainSFeatures
        genFinal=gen.predict([randNoise, fakeLabel])# generate 32 samples, each sample is a 64 by 1 vector. 
        for i1 in range(imbClsNum):
            if rLPerClass[i1].shape[0]!=0: #if the current class has samples
                temp=genFinal[rLPerClass[i1]] #get the generated samples that belong to the current class, shape number of sample by 64
                minorPoints=np.repeat(inClassPoints[i1], rLPerClass[i1].shape[0], axis=0)
                fakePoints[rLPerClass[i1]]=gen_processed[i1].predict([temp, minorPoints])

        mlp.train_on_batch(fakePoints, fakeLabel)
        dis.train_on_batch([fakePoints, fakeLabel], invalid)

        for i1 in range(imbClsNum):
            rLabel=np.zeros((genClassPoints, c))
            rLabel[:, i1]=1
            randNoise=np.random.normal(0, 1, (genClassPoints, latDim))
            oppositeLabel=np.ones((genClassPoints, c))-rLabel
            minorPoints=np.repeat(inClassPoints[i1], genClassPoints, axis=0)
            genP_mlp[i1].train_on_batch([randNoise, rLabel, minorPoints], oppositeLabel)
            genP_dis[i1].train_on_batch([randNoise, rLabel, minorPoints, rLabel], validA)

        if step%resSamplePd==0:
            saveStep=int(step//resSamplePd)

            pLabel=np.argmax(conv_mlp.predict(trainS), axis=1)

            print('here:::::::::::::::::::')
            print(pLabel)
            print('here:::::::::::::::::::')
            print(labelTr)

            acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTr)
            print('Train: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
            print('TPR: ', np.round(tpr, 2))
            acsaSaveTr[saveStep], gmSaveTr[saveStep], accSaveTr[saveStep]=acsa, gm, acc
            confMatSaveTr[saveStep]=confMat
            tprSaveTr[saveStep]=tpr

            pLabel=np.argmax(conv_mlp.predict(testS), axis=1)
            acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTs)
            print('Test: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
            print('TPR: ', np.round(tpr, 2))
            acsaSaveTs[saveStep], gmSaveTs[saveStep], accSaveTs[saveStep]=acsa, gm, acc
            confMatSaveTs[saveStep]=confMat
            tprSaveTs[saveStep]=tpr

        if step%modelSamplePd==0 and step!=0:
            direcPath=savePath+'gamo_models_'+str(step)
            if not os.path.exists(direcPath):
                os.makedirs(direcPath)
            dis.save(direcPath+'/DIS_'+str(step)+fileEnd)
            gen.save(direcPath+'/GEN_'+str(step)+fileEnd)
            mlp.save(direcPath+'/MLP_'+str(step)+fileEnd)
            gamoConv.save(direcPath+'/Conv_'+str(step)+fileEnd)
            for i in range(imbClsNum):
                gen_processed[i].save(direcPath+'/GenP_'+str(i)+'_'+str(step)+fileEnd)

        step=step+2
        if step>=max_step: break

pLabel=np.argmax(conv_mlp.predict(trainS), axis=1)
acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTr)
print('Performance on Train Set: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
print('TPR: ', np.round(tpr, 2))
acsaSaveTr[-1], gmSaveTr[-1], accSaveTr[-1]=acsa, gm, acc
confMatSaveTr[-1]=confMat
tprSaveTr[-1]=tpr

pLabel=np.argmax(conv_mlp.predict(testS), axis=1)
acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTs)
print('Performance on Test Set: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
print('TPR: ', np.round(tpr, 2))
acsaSaveTs[-1], gmSaveTs[-1], accSaveTs[-1]=acsa, gm, acc
confMatSaveTs[-1]=confMat
tprSaveTs[-1]=tpr

direcPath=savePath+'gamo_models_'+str(step)
if not os.path.exists(direcPath):
    os.makedirs(direcPath)
dis.save(direcPath+'/DIS_'+str(step)+fileEnd)
gen.save(direcPath+'/GEN_'+str(step)+fileEnd)
mlp.save(direcPath+'/MLP_'+str(step)+fileEnd)
gamoConv.save(direcPath+'/Conv_'+str(step)+fileEnd)
for i in range(imbClsNum):
    gen_processed[i].save(direcPath+'/GenP_'+str(i)+'_'+str(step)+fileEnd)

resSave=savePath+'Results'
np.savez(resSave, acsa=acsa, gm=gm, tpr=tpr, confMat=confMat, acc=acc)
recordSave=savePath+'Record'
np.savez(recordSave, acsaSaveTr=acsaSaveTr, gmSaveTr=gmSaveTr, accSaveTr=accSaveTr, acsaSaveTs=acsaSaveTs, 
    gmSaveTs=gmSaveTs, accSaveTs=accSaveTs, confMatSaveTr=confMatSaveTr, confMatSaveTs=confMatSaveTs, tprSaveTr=tprSaveTr, tprSaveTs=tprSaveTs)