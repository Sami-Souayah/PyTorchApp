from models.eval_model import Evaluate
from models.train_model import Training_Model
from Dsets.dataset import Dataset


def evaluate_stuff():
    inste = Evaluate('AAPL  ')
    return inste.eval()


def train():
    trnmdl = Training_Model()
    trnmdl.train(90)
    trnmdl.test()
    trnmdl.graph_test()


print(evaluate_stuff())