from models.eval_model import Evaluate
from models.train_model import Training_Model
from Dsets.dataset import Dataset


def evaluate_stuff():
    inste = Evaluate('NVDA')
    return inste.eval()


def train():
    trnmdl = Training_Model()
    trnmdl.train(90)
    trnmdl.test()


print(evaluate_stuff())