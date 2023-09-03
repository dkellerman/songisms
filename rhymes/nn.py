'''Rhyme-detecting neural net

Use with rhymesnet management command:
    `./manage.py rhymesnet --data` -> generate training data
    `./manage.py rhymesnet --train` -> train model
    `./manage.py rhymesnet --test` -> test model
    `./manage.py rhymesnet --predict "word1" "word2"` -> predict rhyme
'''

import time
import random
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
import matplotlib.pyplot as plt
from wonderwords import RandomWord
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from songisms import utils

# to be configurable later
MODEL_FILE = './data/rhymes.pt'
TRAIN_FILE = './data/rhymes_train.csv'
DATA_TOTAL_SIZE = 3000  # number of rows to generate
ROWS = 2000  # number of rows to use for training/validation
TEST_SIZE = 2000
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.001
LOSS_MARGIN = 1.0
WORKERS = 2
POSITIONAL_ENCODING = False
EARLY_STOP_EPOCHS = 3  # stop training after n epochs of no validation improvement
USE_TAILS = False  # use IPA stress tails
DATAMUSE_CACHED_ONLY = True  # set false for first few times generating training data
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# these can't be changed right now without also adjusting the network
MAX_LEN = 15  # width (ipa characters)
IPA_FEATURE_LEN = 25  # channels


class RhymesTrainDataset(Dataset):
    '''Loads training data triples (anchor/pos/neg)
    '''
    def __init__(self, pad_to=MAX_LEN):
        self.pad_to = pad_to
        self.triples = utils.data.rhymes_train[:ROWS]

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        anchor, pos, neg = self.triples[idx]
        anchor, pos, neg = make_rhyme_tensors(anchor, pos, neg, pad_to=self.pad_to)
        return anchor, pos, neg


def make_rhyme_tensors(anchor, pos, neg=None, pad_to=MAX_LEN):
    '''Aligns IPA words and returns feature vectors for each IPA character.
    '''
    ipa_conversion_fn = utils.get_ipa_tail if USE_TAILS else utils.get_ipa_text

    # align all 3 if neg is provided, otherwise just 2
    if neg is not None:
        anchor_ipa, pos_ipa, neg_ipa = [ipa_conversion_fn(text) for text in [anchor, pos, neg]]
        anchor_vec, pos_vec, _ = utils.align_vals(anchor_ipa, pos_ipa)
        anchor_vec, neg_vec, _ = utils.align_vals(anchor_ipa, neg_ipa)
        vecs = [anchor_vec, pos_vec, neg_vec]
    else:
        anchor_ipa, pos_ipa = [ipa_conversion_fn(text) for text in [anchor, pos]]
        anchor_vec, pos_vec, _ = utils.align_vals(anchor_ipa, pos_ipa)
        vecs = [anchor_vec, pos_vec]

    for i, vec in enumerate(vecs):
        # pad, or chop if it's too long (but it shouldn't be)
        vec = (['_'] * (pad_to - len(vec))) + [str(c) for c in vec][:MAX_LEN]
        # replace IPA characters with feature arrays
        vec = utils.get_ipa_features_vector(vec)
        # convert to tensor
        vec = torch.tensor(vec, dtype=torch.float)
        vec = vec.transpose(0, 1)

        if POSITIONAL_ENCODING:
            encoder = Summer(PositionalEncoding1D(MAX_LEN))
            vec = vec.unsqueeze(0) # encoder requires a batch dimension
            vec = encoder(vec)
            vec = vec.squeeze(0)

        vecs[i] = vec

    return vecs


class SiameseTripletNet(nn.Module):
    def __init__(self):
        super(SiameseTripletNet, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv1d(IPA_FEATURE_LEN, 64, kernel_size=3, padding=2),
            nn.Tanh(),
            nn.BatchNorm1d(64),
            nn.Dropout1d(p=.2),

            nn.Conv1d(64, 64, kernel_size=3, padding=2),
            nn.Tanh(),
            nn.BatchNorm1d(64),
            nn.Dropout1d(p=.2),

            nn.Conv1d(64, 32, kernel_size=3, padding=2),
            nn.Tanh(),
            nn.BatchNorm1d(32),
            nn.Dropout1d(p=.2),
        )
        self.fc1 = nn.Linear(672, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

    def forward_once(self, x):
        z = self.cnn1(x)
        z = z.view(z.size()[0], -1)
        # print(z.shape)
        z = F.tanh(self.fc1(z))
        z = F.tanh(self.fc2(z))
        z = self.fc3(z)
        return z

    def forward(self, anchor, pos, neg=None):
        anchor_out = self.forward_once(anchor)
        pos_out = self.forward_once(pos)
        if neg is None:
            return anchor_out, pos_out
        neg_out = self.forward_once(neg)
        return anchor_out, pos_out, neg_out


class TripletMarginLoss(nn.Module):
    def __init__(self, margin=LOSS_MARGIN):
        self.margin = margin
        super(TripletMarginLoss, self).__init__()

    def forward(self, anchor, pos, neg):
        return nn.functional.triplet_margin_loss(anchor, pos, neg, margin=self.margin)


def train():
    model = SiameseTripletNet().to(DEVICE)
    criterion = TripletMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # prepare training and validation data loaders
    dataset = RhymesTrainDataset()
    train_data, validation_data = random_split(dataset, [0.8, 0.2])
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    early_stop_counter = 0
    all_losses = []
    all_validation_losses = []
    distances = []  # track distances for norming later
    start_time = time.time()

    for epoch in range(EPOCHS):
        # training set
        prog_bar = tqdm(loader)
        losses = []

        for anchor, pos, neg in prog_bar:
            anchor_out, pos_out, neg_out = model(anchor.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE))
            loss = criterion(anchor_out, pos_out, neg_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            prog_bar.set_description(f"[E{epoch+1}-T] L={sum(losses)/len(losses):.3f}")
            distances += [d.item() for d in pairwise_distance_ignore_batch_dim(anchor_out, pos_out)]
            distances += [d.item() for d in pairwise_distance_ignore_batch_dim(anchor_out, neg_out)]

        all_losses.append(losses)

        # validation set
        prog_bar = tqdm(validation_loader)
        losses = []

        with torch.no_grad():
            for anchor, pos, neg in prog_bar:
                anchor_out, pos_out, neg_out = model(anchor.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE))
                loss = criterion(anchor_out, pos_out, neg_out)
                losses.append(loss.item())
                went_down = losses[-1] < losses[-2] if len(losses) > 1 else True
                prog_bar.set_description(f"[E{epoch+1}-v] L={sum(losses)/len(losses):.3f}"
                                         f"{'-' if went_down else '+'}")
                distances += [d.item() for d in pairwise_distance_ignore_batch_dim(anchor_out, pos_out)]
                distances += [d.item() for d in pairwise_distance_ignore_batch_dim(anchor_out, neg_out)]

        all_validation_losses.append(losses)
        early_stop_counter = 0 if went_down else early_stop_counter + 1
        if early_stop_counter >= EARLY_STOP_EPOCHS:
            print("Early stopping")
            break

    elapsed = time.time() - start_time
    print(f"Training took {elapsed/60.0:.2f} mins | ~epoch: {elapsed/EPOCHS:.2f} secs")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'distances': distances,
    }, MODEL_FILE)

    # plot training/validation losses averaged per epoch
    plt.plot([sum(epoch)/len(epoch) for epoch in all_losses], label='Training Loss')
    plt.plot([sum(epoch)/len(epoch) for epoch in all_validation_losses], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def pairwise_distance_ignore_batch_dim(tensor1, tensor2, *args, **kwargs):
    # wish I could find a better way to do this
    tensor1 = tensor1.squeeze(0)
    tensor2 = tensor2.squeeze(0)
    distance = F.pairwise_distance(tensor1, tensor2, *args, **kwargs)
    tensor1 = tensor1.unsqueeze(0)
    tensor2 = tensor2.unsqueeze(0)
    return distance


class RhymesTestDataset(Dataset):
    '''More real-world dataset for final testing, produces pairs of words and
       a label => (text, text, 1.0|0.0)
    '''
    def __init__(self):
        song_rhyme_sets = utils.data.rhymes
        random.shuffle(song_rhyme_sets)
        labeled_pairs = []

        # half of dataset is positive rhyme pairs
        for rset in song_rhyme_sets:
            pairs = utils.get_rhyme_pairs(';'.join(rset))
            for text1, text2 in pairs:
                if (text1, text2, 1.0) not in labeled_pairs and \
                   (text2, text1, 1.0) not in labeled_pairs:
                    labeled_pairs.append((text1, text2, 1.0))

            if len(labeled_pairs) >= TEST_SIZE // 2:
                labeled_pairs = labeled_pairs[:TEST_SIZE // 2]
                break

        # second half of dataset are random negatives, probably...
        # quick sample showed me about 5% maybe-rhymes
        rw = RandomWord()
        while len(labeled_pairs) < TEST_SIZE:
            w1, w2 = rw.word(), rw.word()

            # spot check
            if w1 == w2 or (utils.get_ipa_tail(w1) == utils.get_ipa_tail(w2)):
                continue

            if (w1, w2, 0.0) not in labeled_pairs and \
               (w2, w1, 0.0) not in labeled_pairs and \
               (w1, w2, 1.0) not in labeled_pairs and \
               (w2, w1, 1.0) not in labeled_pairs:
                labeled_pairs.append((w1, w2, 0.0))

        print("Loaded/created", len(labeled_pairs), "test rhymes")
        self.labeled_pairs = labeled_pairs[:TEST_SIZE]

    def __len__(self):
        return len(self.labeled_pairs)

    def __getitem__(self, idx):
        return self.labeled_pairs[idx]


class RhymeScorer():
    '''Normalize distance into a ascore between 0 and 1,
       where 0 is not a rhyme and 1 is a perfect rhyme
    '''
    def __init__(self, distances):
        self.robust = RobustScaler()
        scaled = self.robust.fit_transform([[d] for d in distances])
        self.minmax = MinMaxScaler().fit(scaled)

    def __call__(self, distance):
        val = [[distance]]
        val = self.robust.transform(val)
        val = self.minmax.transform(val)
        return 1.0 - val[0][0]


def load_model():
    # TODO: use torch script instead of loading full model
    model = SiameseTripletNet().to(DEVICE)
    model_params = torch.load(MODEL_FILE)
    model.load_state_dict(model_params.get('model_state_dict'))
    model.eval()

    # make scorer using distances from training
    scorer = RhymeScorer(model_params.get('distances'))

    return model, scorer


def test():
    dataset = RhymesTestDataset()
    loader = DataLoader(dataset, shuffle=True, num_workers=WORKERS)
    model, scorer = load_model()
    prog_bar = tqdm(loader, "Test Rhymes")

    correct = []
    wrong = []

    for batch in prog_bar:
        text1, text2, label = batch[0][0], batch[1][0], batch[2].item()
        pred, score, _ = predict(text1, text2, model=model, scorer=scorer)

        if (pred and label == 1.0) or (not pred and label == 0.0):
            correct.append((text1, text2, pred, score))
        else:
            wrong.append((text1, text2, pred, score))

        prog_bar.set_description(f"√: {len(correct)} X: {len(wrong)} "
                                 f"%: {(len(correct)/(len(correct)+len(wrong))*100):.1f}")

    # print stats and wrong predictions for inspection
    for text1, text2, pred, score in wrong:
        print('[X]', f'[PRED={"Y" if pred else "N"}]', text1, '=>', text2, f'[{score:.3f}]')

    total = len(correct) + len(wrong)
    pct = (len(correct) / total) * 100
    print("\nCorrect:", len(correct), "| Wrong:", len(wrong), "| Pct:", f"{pct:.3f}%")
    print("Average wrong score:", sum([s[-1] for s in wrong]) / len(wrong))
    print("Average correct score:", sum([s[-1] for s in correct]) / len(correct))
    print("Tough calls:", len([s for s in wrong if s[-1] > .45 and s[-1] < .55]))


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
        model, scorer = load_model()

    anchor_vec, other_vec = make_rhyme_tensors(text1, text2)
    # fake a batch dimension here
    anchor_vec = anchor_vec.unsqueeze(0)
    other_vec = other_vec.unsqueeze(0)

    anchor_out, other_out = model(anchor_vec.to(DEVICE), other_vec.to(DEVICE))
    distance = pairwise_distance_ignore_batch_dim(anchor_out, other_out).item()
    score = scorer(distance)

    pred = score > .5
    label = None
    for thresh, val in SCORE_LABELS:
        if score <= thresh:
            label = val
            break

    return pred, score, label


def make_training_data():
    '''Output siamese neural net training data triples to CSV file
    '''
    rw = RandomWord()
    lines = []

    for _ in tqdm(range(DATA_TOTAL_SIZE)):
        anchor = None
        positive = None

        # lookup a positive value using datamuse
        while positive is None:
            anchor = rw.word()
            anchor = utils.normalize_lyric(anchor)
            rhymes = utils.get_datamuse_rhymes(anchor, cache_only=DATAMUSE_CACHED_ONLY)
            if rhymes:
                positive = random.choice(rhymes)['word']
                if positive.endswith(anchor) or positive.endswith(anchor + 's'):
                   positive = None

        positive = utils.normalize_lyric(positive)

        # random negative
        negative = utils.normalize_lyric(rw.word())
        # check IPA length doesn't exceed max length of net
        if any([len(utils.get_ipa_text(w)) > MAX_LEN - 1 for w in [anchor, positive, negative]]):
             continue
        # and quick spot check
        if utils.get_ipa_tail(anchor) == utils.get_ipa_tail(negative):
            continue

        lines.append((anchor, positive, negative))

    with open(TRAIN_FILE, 'w') as f:
        f.write('\n'.join(';'.join(l) for l in lines))
