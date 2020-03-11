import torch
import torchtext

import os
import torch.nn as nn
import torch.nn.functional as F
import re
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
import time
from torch.utils.data.dataset import random_split
import numpy as np
import pandas as pd

# load __init__.py from swag models, I had a nlp_model file with a network inside
from swag import data, models, utils, losses

# load __init__ .py from swag posteriors
from swag.posteriors import SWAG


import argparse
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument("--no_schedule", action="store_true", help="store schedule")
dictionnary_arguments = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dir": r"D:\dossier important 2020\main_folder\results",
    "seed": 1,
    "model": "TextSentiment_16",
    "dataset": "NLP",
    "data_path": "qzdq",
    "num_workers": 0,
    "no_cov_mat": True,
    "cov_mat": False,
    "momentum": 0.9,
    "lr_init": 4.0,
    "eval_freq": 1,
    "save_freq": 10,
    "loss": "CE",
    "swa_resume": None,
    "resume": None,
    "epochs": 30,
    "swa_lr": 0.01,
    "swa_start": 10,
    "batch_size": 128,
    "use_test": True,
    "split_classes": None,
    "wd": 1e-4,
    "swa": True,
    "swa_c_epochs": 1,
    "max_num_models": 20,
    "no_schedule": False
}

# use cuda or not
args = parser.parse_args()
args.device = dictionnary_arguments["device"]
args.dir = dictionnary_arguments["dir"]
args.seed = dictionnary_arguments["seed"]
args.model = dictionnary_arguments["model"]
args.dataset = dictionnary_arguments["dataset"]
args.data_path = dictionnary_arguments["data_path"]
args.num_workers = dictionnary_arguments["num_workers"]
args.no_cov_mat = dictionnary_arguments["no_cov_mat"]
args.cov_mat = dictionnary_arguments["cov_mat"]
args.momentum = dictionnary_arguments["cov_mat"]
args.lr_init = dictionnary_arguments["lr_init"]
args.eval_freq = dictionnary_arguments["eval_freq"]
args.save_freq = dictionnary_arguments["save_freq"]
args.loss = dictionnary_arguments["loss"]
args.swa_resume = dictionnary_arguments["swa_resume"]
args.epochs = dictionnary_arguments["epochs"]
args.swa_lr = dictionnary_arguments["swa_lr"]
args.swa_start = dictionnary_arguments["swa_start"]
args.batch_size = dictionnary_arguments["batch_size"]
args.resume = dictionnary_arguments["resume"]
args.use_test = dictionnary_arguments["use_test"]
args.wd = dictionnary_arguments["wd"]
args.swa = dictionnary_arguments["swa"]
args.swa_c_epochs = dictionnary_arguments["swa_c_epochs"]
args.max_num_models = dictionnary_arguments["max_num_models"]
args.no_schedule = dictionnary_arguments["no_schedule"]
args.split_classes = dictionnary_arguments["split_classes"]


NGRAMS = 2
N_EPOCHS = args.epochs


if not os.path.isdir('./.data_sentiment'):
    os.mkdir('./.data_sentiment')

train_dataset, test_dataset = torchtext.datasets.AG_NEWS(
    root='./.data_sentiment', ngrams=NGRAMS, vocab=None)
BATCH_SIZE = 16
device = args.device

VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_dataset.get_labels())

class TextSentiment(nn.Module):
    args = list()
    kwargs = dict()
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_classes=NUN_CLASS):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


model_NLP = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)

# initiate the swag model
num_classes = 4
print("SWAG training")
swag_model = SWAG(TextSentiment, no_cov_mat=args.no_cov_mat, max_num_models=args.max_num_models, *model_NLP.args, num_classes=num_classes, **model_NLP.kwargs)
swag_model.to(args.device)


# choose the columns to plot at the end
columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time", "mem_usage"]
columns = columns[:-2] + ["swa_te_loss", "swa_te_acc"] + columns[-2:]
swag_res = {"loss": None, "accuracy": None}


def schedule(epoch):
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if t <= 0.5:
        # keep the initial learning rate
        factor = 1.0
    elif t <= 0.9:
        # decreasing learning rate
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        # final stable learning rate (cf image)
        factor = lr_ratio
    return args.lr_init * factor


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def train_func(sub_train_):
    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model_NLP(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc += (output.argmax(1) == cls).sum().item()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)


def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model_NLP(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()
    return loss / len(data_), acc / len(data_)


min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model_NLP.parameters(), lr=4.0, momentum=args.momentum)#, weight_decay=args.wd)

# save the current model
start_epoch = 0
utils.save_checkpoint(
    args.dir,
    start_epoch,
    state_dict=model_NLP.state_dict(),
    optimizer=optimizer.state_dict(),
)

sgd_ens_preds = None
sgd_targets = None
n_ensembled = 0.0


train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])
loaders, num_classes = data.loaders(dataset=args.dataset, path=args.data_path, batch_size = BATCH_SIZE, num_workers = args.num_workers, transform_train=sub_train_, transform_test=sub_valid_)

train_swag_loss = []
train_swag_accuracy = []

test_swag_loss = []
test_swag_accuracy = []




for epoch in range(N_EPOCHS):
    lr = schedule(epoch)
    utils.adjust_learning_rate(optimizer, lr)

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')


    # save the model every swa_c_epochs itteration

    if ((epoch + 1) > args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0):
        # sgd_res contain two element : prediction of the model and targets which is the true value

        sgd_res = utils.predict(loaders["test"], model_NLP, type_model="NLP")
        sgd_preds = sgd_res["predictions"]
        sgd_targets = sgd_res["targets"]


        print("updating sgd_ens")
        if sgd_ens_preds is None:
            # initiate sgd_ens_preds
            sgd_ens_preds = sgd_preds.copy()
        else:
            # update sgd_ens_preds the mean of the pred
            sgd_ens_preds = sgd_ens_preds * n_ensembled / (n_ensembled + 1) + sgd_preds / (n_ensembled + 1)

        n_ensembled += 1
        ## store the model to use the parameters later
        swag_model.collect_model(model_NLP)

        # try swag !!!

        swag_model.sample(0.0)
        utils.bn_update(loaders["train"], swag_model)

        swag_res_test = utils.eval(loaders["test"], swag_model, criterion, model_type = "NLP")
        swag_res_train = utils.eval(loaders["train"], swag_model, criterion, model_type="NLP")

        print("swag_res_test : ", swag_res_test)
        print("swag_res_train : ", swag_res_train)
        train_swag_loss.append(swag_res_train['loss'])
        train_swag_accuracy.append(swag_res_train['accuracy'])

        test_swag_loss.append(swag_res_test['loss'])
        test_swag_accuracy.append(swag_res_test['accuracy'])



    # save the model at a given frequence args.save_freq
    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(args.dir, epoch + 1, state_dict=model_NLP.state_dict(), optimizer=optimizer.state_dict())
        utils.save_checkpoint(args.dir, epoch + 1, name="swag", state_dict=swag_model.state_dict())


# save the model at the end
if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        state_dict=model_NLP.state_dict(),
        optimizer=optimizer.state_dict(),
    )
    if args.epochs > args.swa_start:
        utils.save_checkpoint(
            args.dir, args.epochs, name="swag", state_dict=swag_model.state_dict()
        )

# save arrays : Save several arrays into a single file in uncompressed .npz format : the prediction and the target
# of the model
print(sgd_ens_preds)
print(sgd_targets)
np.savez(os.path.join(args.dir, "sgd_ens_preds.npz"), predictions=sgd_ens_preds,targets=sgd_targets)

dict_data = {
    "train_swag_loss": train_swag_loss,
    "train_swag_accuracy": train_swag_accuracy,
    "test_swag_loss": test_swag_loss,
    "test_swag_accuracy": test_swag_accuracy
}


df = pd.DataFrame(dict_data, columns=dict_data.keys())
print(df)
path = "D:\\dossier important 2020\\swa_gaussian-master\\optimizer_results\\" + "SWAG" + ".csv"
df.to_csv(path, index=False, header=True)

print('Checking the results of test dataset...')
test_loss, test_acc = test(test_dataset)
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')


ag_news_label = {1 : "World",
                 2 : "Sports",
                 3 : "Business",
                 4 : "Sci/Tec"}


def predict(text, model_NLP, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model_NLP(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."


vocab = train_dataset.get_vocab()
model_NLP = model_NLP.to("cpu")

print("This is a %s news" %ag_news_label[predict(ex_text_str, model_NLP, vocab, 2)])