'''Rhyme-detecting neural net

Use with rhymesnet management command:
    `./manage.py rhymesnet --train`
    `./manage.py rhymesnet --test`
    `./manage.py rhymesnet --predict "word1" "word2"`
'''

import time
import re
import random
import torch
import numpy as np
from typing import List
from functools import lru_cache
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from statistics import mean
from dataclasses import dataclass
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from wonderwords import RandomWord
from sklearn import metrics
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from songisms import utils
import pronouncing as pron


@dataclass
class Config:
    random_seed: int = 5050
    model_file_full: str = './data/rhymes_full.pt'
    model_file_script: str = './data/rhymes.pt'
    distances_file: str = './data/rhymes_distances.pt'
    train_file: str = './data/rhymes_train.csv'
    test_misses_file: str = './data/rhymes_test_misses.csv'
    train_rows: int = 3000  # number of rows to use for training/validation
    test_rows: int = 2000  # number of rows to use for testing
    batch_size: int = 64
    epochs: int = 10
    lr: float = 0.001
    loss_margin: float = 1.5
    workers: int = 1
    positional_encoding: bool = False
    early_stop_epochs: int = 3  # stop training after n epochs of no validation improvement
    device: str = "cpu"  # str(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))

    # these can't be changed right now without also adjusting the network
    max_len: int = 15  # width (ipa characters)
    ipa_feature_len: int = 25  # channels


config = Config()
torch.manual_seed(config.random_seed)
random.seed(config.random_seed)
ipa_cache = None


def to_ipa(text):
    return re.sub(r'[\s.]+', '', utils.data.ipa.get(text, '')).strip() or None


class RhymesTrainDataset(Dataset):
    def __init__(self, pad_to=config.max_len):
        self.pad_to = pad_to
        self.rw = RandomWord()
        self.train_data = utils.data.rhymes_train
        random.shuffle(self.train_data)

        from rhymes.models import Vote
        votes = Vote.objects.exclude(alt2=None).filter(label__in=['alt1', 'alt2'])
        rlhf = []
        for v in votes:
            pos = v.alt1 if v.label == 'alt1' else v.alt2
            neg = v.alt1 if v.label == 'alt2' else v.alt2
            rlhf.append((0.8, v.anchor, pos, neg))
        random.shuffle(rlhf)
        self.rlhf = rlhf
        print("*counts:", 'RLHF =>', len(self.rlhf), 'TRAIN =>', len(self.train_data))

    def __len__(self):
        return config.train_rows

    def __getitem__(self, idx):
        score, anc_ipa, pos_ipa, neg_ipa = None, None, None, None
        while len(self.train_data) and (not anc_ipa or not pos_ipa or not neg_ipa):
            if random.random() > .1 and len(self.rlhf):
                score, anchor, pos, neg = self.rlhf.pop()
                anc_ipa, pos_ipa, neg_ipa = to_ipa(anchor), to_ipa(pos), to_ipa(neg)
            else:
                score, anchor, other = self.train_data.pop()
                if score > .5:
                    pos = other
                    neg = random.choice(list(utils.data.gpt_ipa.keys()))
                else:
                    neg = other
                    rhymes = pron.rhymes(anchor)
                    if not rhymes or not len(rhymes):
                        continue
                    pos = random.choice(rhymes)
                anc_ipa, pos_ipa, neg_ipa = to_ipa(anchor), to_ipa(pos), to_ipa(neg)

        label = torch.tensor(score, dtype=torch.float32)
        return label, *make_rhyme_tensors(anc_ipa, pos_ipa, neg_ipa, pad_to=self.pad_to)


class RhymesTestDataset(Dataset):
    '''More real-world dataset for final testing
    '''
    def __init__(self):
        self.data = utils.data.rhymes_test
        random.shuffle(self.data)

    def __len__(self):
        return config.test_rows

    def __getitem__(self, idx):
        score, anchor, other = self.data.pop()
        return score, anchor, other


def make_rhyme_tensors(anchor_ipa, pos_ipa, neg_ipa=None, pad_to=config.max_len) -> List[torch.Tensor]:
    '''Aligns IPA words and returns feature vectors for each IPA character.
    '''
    # align all 3 if neg is provided, otherwise just 2
    if neg_ipa is not None:
        anchor_vec, pos_vec, _ = utils.align_vals(anchor_ipa, pos_ipa)
        anchor_vec, neg_vec, _ = utils.align_vals(anchor_ipa, neg_ipa)
        vecs = [anchor_vec, pos_vec, neg_vec]
    else:
        anchor_vec, pos_vec, _ = utils.align_vals(anchor_ipa, pos_ipa)
        vecs = [anchor_vec, pos_vec]

    tensors: List[torch.Tensor] = []
    for vec in vecs:
        # pad, or chop if it's too long (but it shouldn't be)
        vec = (['_'] * (pad_to - len(vec))) + [str(c) for c in vec][:config.max_len]
        # replace IPA characters with feature arrays
        vec = np.array(utils.get_ipa_features_vector(vec))
        # convert to tensor
        tensor = torch.tensor(vec, dtype=torch.float32)
        tensor = tensor.transpose(0, 1)

        if config.positional_encoding:
            encoder = Summer(PositionalEncoding1D(config.max_len))
            tensor = tensor.unsqueeze(0) # encoder requires a batch dimension
            tensor = encoder(tensor)
            tensor = tensor.squeeze(0)

        tensors.append(tensor)

    return tensors


class SiameseTripletNet(nn.Module):
    def __init__(self):
        super(SiameseTripletNet, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv1d(config.ipa_feature_len, 64, kernel_size=4, padding=2, dilation=2),
            nn.Tanh(),
            nn.BatchNorm1d(64),
            nn.Dropout1d(p=.2),

            nn.Conv1d(64, 64, kernel_size=4, padding=2, dilation=2),
            nn.Tanh(),
            nn.BatchNorm1d(64),
            nn.Dropout1d(p=.2),

            nn.Conv1d(64, 32, kernel_size=4, padding=2, dilation=2),
            nn.Tanh(),
            nn.BatchNorm1d(32),
            nn.Dropout1d(p=.2),
        )
        self.fc1 = nn.Linear(288, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

    def forward_once(self, x):
        z = self.cnn1(x)
        z = z.view(z.size()[0], -1)
        z = torch.tanh(self.fc1(z))
        z = torch.tanh(self.fc2(z))
        z = self.fc3(z)
        return z

    def forward(self, anchor, pos, neg=None):
        anchor_out = self.forward_once(anchor)
        pos_out = self.forward_once(pos)
        if neg is None:
            return anchor_out, pos_out
        neg_out = self.forward_once(neg)
        return anchor_out, pos_out, neg_out


class WeightedDistance:
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, x1, x2):
        distance = torch.norm(x1 - x2, p=2, dim=1)
        return distance * self.weight


def train():
    model = SiameseTripletNet().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # prepare training and validation data loaders
    dataset = RhymesTrainDataset()
    train_data, validation_data = random_split(dataset, [0.8, 0.2])
    loader = DataLoader(train_data, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.workers)
    validation_loader = DataLoader(validation_data, batch_size=config.batch_size,
                                   shuffle=True, num_workers=config.workers)
    early_stop_counter = 0
    all_losses = []
    all_validation_losses = []
    distances = []  # track distances for norming later
    start_time = time.time()

    for epoch in range(config.epochs):
        # training set
        prog_bar = tqdm(loader)
        losses = []

        for batch in prog_bar:
            weight, anchor, pos, neg = [b.to(config.device) for b in batch]
            anchor_out, pos_out, neg_out = model(anchor, pos, neg)

            distance_fn = WeightedDistance(weight=weight)
            criterion = nn.TripletMarginWithDistanceLoss(distance_function=distance_fn,
                                                         margin=config.loss_margin)
            loss = criterion(anchor_out, pos_out, neg_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            prog_bar.set_description(f"[E{epoch+1}-T] L={mean(losses):.3f}")

            distances += [d.item() for d in distance_fn(anchor_out.squeeze(0), pos_out.squeeze(0))]
            distances += [d.item() for d in distance_fn(anchor_out.squeeze(0), neg_out.squeeze(0))]

        prog_bar.close()
        all_losses.append(losses)

        # validation set
        prog_bar = tqdm(validation_loader)
        losses = []

        with torch.no_grad():
            for batch in prog_bar:
                weight, anchor, pos, neg = [b.to(config.device) for b in batch]
                anchor_out, pos_out, neg_out = model(anchor, pos, neg)

                distance_fn = WeightedDistance(weight=weight)
                criterion = nn.TripletMarginWithDistanceLoss(distance_function=distance_fn,
                                                             margin=config.loss_margin)
                loss = criterion(anchor_out, pos_out, neg_out)
                losses.append(loss.item())
                prog_bar.set_description(f"[E{epoch+1}-v] L={mean(losses):.3f} es:{early_stop_counter}")
                distances += [d.item() for d in distance_fn(anchor_out.squeeze(0), pos_out.squeeze(0))]
                distances += [d.item() for d in distance_fn(anchor_out.squeeze(0), neg_out.squeeze(0))]

        prog_bar.close()
        all_validation_losses.append(losses)

        # check for early stop
        went_down = mean(all_validation_losses[-1]) < mean(all_validation_losses[-2]) \
            if len(all_validation_losses) > 1 else True
        early_stop_counter = 0 if went_down else early_stop_counter + 1
        if early_stop_counter >= config.early_stop_epochs:
            print("Early stopping")
            break

    elapsed = time.time() - start_time
    print(f"Training took {elapsed/60.0:.2f} mins | ~epoch: {elapsed/config.epochs:.2f} secs")

    save_full_model(model, optimizer, distances)
    save_script_model(model, list(distances))

    # plot training/validation losses averaged per epoch
    plt.plot([mean(epoch) for epoch in all_losses], label='Training Loss')
    plt.plot([mean(epoch) for epoch in all_validation_losses], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


class RhymeScorer():
    '''Normalize distance into a ascore between 0 and 1,
       where 0 is not a rhyme and 1 is a perfect rhyme
    '''
    def __init__(self, distances):
        self.robust = RobustScaler()
        scaled = self.robust.fit_transform(distances.reshape(-1, 1))
        self.minmax = MinMaxScaler().fit(scaled)

    def __call__(self, distance):
        val = np.array([[distance]])
        val = self.robust.transform(val)
        val = self.minmax.transform(val)
        return 1.0 - val[0][0]


def save_full_model(model, optimizer, distances):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'distances': tuple(distances),
    }, config.model_file_full)


def load_full_model():
    model = SiameseTripletNet().to(config.device)
    model_params = torch.load(config.model_file_full)
    model.load_state_dict(model_params.get('model_state_dict'))
    model.eval()
    # make scorer using distances from training
    scorer = RhymeScorer(np.array(model_params.get('distances')))
    return model, scorer


def save_script_model(model, distances):
    model.eval()
    inp1 = torch.randn((1, config.ipa_feature_len, config.max_len))
    inp2 = torch.randn((1, config.ipa_feature_len, config.max_len))
    traced_model = torch.jit.trace(model, (inp1, inp2))
    torch.jit.save(traced_model, config.model_file_script)
    torch.save({
        'distances': tuple(distances),
    }, config.distances_file)


def load_script_model():
    model = torch.jit.load(config.model_file_script)
    model.to(config.device)
    model.eval()
    distances = torch.load(config.distances_file)['distances']
    # make scorer using distances from training
    scorer = RhymeScorer(np.array(distances))
    return model, scorer


def test():
    dataset = RhymesTestDataset()
    loader = DataLoader(dataset, shuffle=True, num_workers=config.workers)
    model, scorer = load_script_model()
    prediction_times = []
    prog_bar = tqdm(loader, "Test Rhymes")
    true_labels = []
    predicted_scores = []
    correct = []
    wrong = []

    for batch in prog_bar:
        label, anchor, other = batch[0].item(), batch[1][0], batch[2][0]
        start_time = time.time()
        score = predict(anchor, other, model=model, scorer=scorer)
        prediction_times.append(time.time() - start_time)
        predicted = score >= .5
        true_labels.append(label)
        predicted_scores.append(score)
        if (predicted and label == 1.0) or (not predicted and label == 0.0):
            correct.append((anchor, other, predicted, score))
        else:
            wrong.append((anchor, other, predicted, score))

        prog_bar.set_description(f"√: {len(correct)} X: {len(wrong)} "
                                 f"%: {(len(correct)/(len(correct)+len(wrong))*100):.1f}")
    prog_bar.close()

    # metrics
    predicted_labels = [1.0 if s >= .5 else 0.0 for s in predicted_scores]
    acc = metrics.accuracy_score(true_labels, predicted_labels)
    prec = metrics.precision_score(true_labels, predicted_labels)
    nprec = metrics.precision_score(true_labels, predicted_labels, pos_label=0)
    recall = metrics.recall_score(true_labels, predicted_labels)
    nrecall = metrics.recall_score(true_labels, predicted_labels, pos_label=0)
    f1 = metrics.f1_score(true_labels, predicted_labels)
    nf1 = metrics.f1_score(true_labels, predicted_labels, pos_label=0)
    auc = metrics.roc_auc_score(true_labels, predicted_scores)
    loss = metrics.log_loss(true_labels, predicted_scores)

    print("\n* Correct:", len(correct), "| Wrong:", len(wrong))
    print("* Avg wrong score:", mean([s[-1] for s in wrong]))
    print("* Avg correct score:", mean([s[-1] for s in correct]))
    print("* Avg prediction time:", f"{mean(prediction_times)*1000.0:.2f}ms")
    print("* Tough calls:", len([s for s in wrong if s[-1] > .45 and s[-1] < .55]))
    print("\n" + "=" * 40 + "\n")
    print("* Accuracy:", f"{acc:.3f}")
    print("* Precision:", f"{prec:.3f}", f"/ neg: {nprec:.3f}")
    print("* Recall:", f"{recall:.3f}", f"/ neg: {nrecall:.3f}")
    print("* F1:", f"{f1:.3f}", f"/ neg: {nf1:.3f}")
    print("* AUC:", f"{auc:.3f}")
    print("* Log Loss:", f"{loss:.3f}")
    print("\nWritng misses to:", config.test_misses_file)

    with open(config.test_misses_file, 'w') as f:
        f.write('text1,text2,ipa1,ipa2,pred,score\n')
        for vals in wrong:
            f.write(f"{vals[0]},{vals[1]},"
                    f"{to_ipa(vals[0])},{to_ipa(vals[1])},"
                    f"{'Y' if vals[2] else 'N'},{vals[3]:.3f}\n")


SCORE_LABELS = (
    (.2, "Not a rhyme"),
    (.3, "Unlikely rhyme"),
    (.5, "Near rhyme"),
    (.7, "Weak rhyme"),
    (.9, "Slant rhyme"),
    (1.0, "Perfect rhyme"),
)

def predict(text1, text2, model=None, scorer=lambda x: x):
    if not model:
        model, scorer = load_script_model()

    anchor_ipa, other_ipa = to_ipa(text1), to_ipa(text2)
    if not anchor_ipa or not other_ipa:
        print(f"Could not find IPA for {text1} or {text2}")
        return 0.0
    anchor, other = make_rhyme_tensors(anchor_ipa, other_ipa)

    # fake a batch dimension here
    anchor = anchor.unsqueeze(0).to(config.device)
    other = other.unsqueeze(0).to(config.device)

    anchor_out, other_out = model(anchor, other)
    distance = torch.norm(anchor_out - other_out, p=2, dim=1)
    distance = distance.squeeze(0).detach().numpy()
    score = scorer(distance)

    return score


def score_to_label(score):
    label = None
    for thresh, val in SCORE_LABELS:
        if score <= thresh:
            label = val
            break
    return label


def calc_cnn_output_size():
    model = SiameseTripletNet()
    dummy_input = torch.randn((1, config.ipa_feature_len, config.max_len))
    output = model.cnn1(dummy_input)
    output = output.view(output.size()[0], -1)
    return output.size(1)


if __name__ == '__main__':
    print(calc_cnn_output_size())
