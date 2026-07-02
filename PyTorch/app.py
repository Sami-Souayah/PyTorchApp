from models.eval_model import Evaluate
from models.train_model import Training_Model
from Dsets.dataset import Dataset


def evaluate_stuff():
    inste = Evaluate('META')
    return inste.eval()


def train(epoch):
    trnmdl = Training_Model()
    trnmdl.train(epoch)
    trnmdl.test()


if __name__ == "__main__":
    print(train())
