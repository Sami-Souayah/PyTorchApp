from models.eval_model import Evaluate
from models.train_model import Training_Model
from Dsets.dataset import Dataset
from Dsets.training_dataset import Training_Dataset


def evaluate_stuff():
    inste = Evaluate('META')
    return inste.eval()


def train():
    trnmdl = Training_Model()
    trnmdl.train(35)
    trnmdl.test()


print(train())