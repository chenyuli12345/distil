import torch

from .score_streaming_strategy import ScoreStreamingStrategy

class EntropySampling(ScoreStreamingStrategy):    #定义一个类，继承自ScoreStreamingStrategy类
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(EntropySampling, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
    
    def acquire_scores(self, unlabeled_buffer):
        
        probs = self.predict_prob(unlabeled_buffer)
        log_probs = torch.log(probs)
        U = -(probs*log_probs).sum(1)
        return U