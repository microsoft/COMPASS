from __future__ import print_function

import os
import torch

from workflow import WorkFlow

class TorchFlow(WorkFlow.WorkFlow):
    """Add pytorch support based on WorkFlow.WorkFlow"""
    def __init__(self, workingDir, prefix = "", suffix = "", disableStreamLogger = False, plotterType='Visdom'):
        super(TorchFlow, self).__init__(workingDir, prefix, suffix, None, disableStreamLogger)

        self.plotterType = plotterType
    
    # Overload the function initialize().
    def initialize(self):
        super(TorchFlow, self).initialize()

    # Overload the function train().
    def train(self):
        super(TorchFlow, self).train()

    # Overload the function test().
    def test(self):
        super(TorchFlow, self).test()

    # Overload the function finalize().
    def finalize(self):
        super(TorchFlow, self).finalize()

    # ========== Class specific member functions. ==========
    def load_model(self, model, modelname):
        preTrainDict = torch.load(modelname)
        model_dict = model.state_dict()
        # print 'preTrainDict:',preTrainDict.keys()
        # print 'modelDict:',model_dict.keys()
        preTrainDictTemp = {k:v for k,v in preTrainDict.items() if k in model_dict}

        if( 0 == len(preTrainDictTemp) ):
            self.logger.info("Does not find any module to load. Try DataParallel version.")
            for k, v in preTrainDict.items():
                kk = k[7:]

                if ( kk in model_dict ):
                    preTrainDictTemp[kk] = v

        if ( 0 == len(preTrainDictTemp) ):
            raise WorkFlow.WFException("Could not load model from %s." % (modelname), "load_model")

        for item in preTrainDictTemp:
            self.logger.info("Load pretrained layer:{}".format(item) )
        model_dict.update(preTrainDictTemp)
        model.load_state_dict(model_dict)
        return model

    def save_model(self, model, modelname):
        modelname = self.prefix + modelname + self.suffix + '.pkl'
        torch.save(model.state_dict(), os.path.join(self.modeldir, modelname))

    def append_plotter(self, plotName, valueNameList, dispType, semiLog=False):
        if self.plotterType == 'Visdom':
            self.AVP.append(WorkFlow.VisdomLinePlotter(plotName, self.AV, valueNameList, dispType, semiLog=semiLog))
        elif self.plotterType == 'Int':
            self.AVP.append(WorkFlow.PLTIntermittentPlotter(self.workingDir + "/IntPlot", plotName, self.AV, valueNameList, dispType, semiLog=semiLog))

