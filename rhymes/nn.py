'''Rhyme-detecting neural net

Use with rhymesnet management command:
    `./manage.py rhymesnet --data` -> generate training data
    `./manage.py rhymesnet --train` -> train model
    `./manage.py rhymesnet --test` -> test model
    `./manage.py rhymesnet --predict "word1" "word2"` -> predict rhyme
'''

import random
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
import matplotlib.pyplot as plt
from wonderwords import RandomWord
from songisms import utils

# to be configurable later
MODEL_FILE = './data/rhymes.pt'
TRAIN_FILE = './data/rhymes_train.csv'
DATA_TOTAL_SIZE = 5000  # number of rows to generate
ROWS = 5000  # number of rows to use for training/validation
TEST_SIZE = 1000
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
LOSS_MARGIN = 1.0
WORKERS = 4
POSITIONAL_ENCODING = False
USE_TAILS = False  # use IPA stress tails
DATAMUSE_CACHED_ONLY = True  # set false for first few times generating training data
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# this can't be changed right now without also adjusting the network
MAX_LEN = 20


class RhymesDataset(Dataset):
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
            nn.Conv1d(25, 64, kernel_size=3),
            nn.Tanh(),
            nn.BatchNorm1d(64),
            nn.Dropout1d(p=.2),

            nn.Conv1d(64, 64, kernel_size=3),
            nn.Tanh(),
            nn.BatchNorm1d(64),
            nn.Dropout1d(p=.2),

            nn.Conv1d(64, 32, kernel_size=3),
            nn.Tanh(),
            nn.BatchNorm1d(32),
            nn.Dropout1d(p=.2),
        )
        self.fc1 = nn.Linear(448, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 50)

    def forward_once(self, x):
        z = self.cnn1(x)
        z = z.view(z.size()[0], -1)
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
    model = SiameseTripletNet()
    criterion = TripletMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # prepare training and validation data loaders
    dataset = RhymesDataset()
    train_data, validation_data = random_split(dataset, [0.8, 0.2])
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    val_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    all_losses = []
    all_validation_losses = []
    distances = []  # track distances for norming later

    for epoch in range(EPOCHS):
        # training set
        prog_bar = tqdm(loader)
        losses = []

        for (anchor, pos, neg) in prog_bar:
            anchor_out, pos_out, neg_out = model(anchor, pos, neg)
            loss = criterion(anchor_out, pos_out, neg_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            prog_bar.set_description(f"[E{epoch+1}-T]* L={sum(losses)/len(losses):.3f}")
            distances += [d.item() for d in pairwise_distance_ignore_batch_dim(anchor_out, pos_out)]
            distances += [d.item() for d in pairwise_distance_ignore_batch_dim(anchor_out, neg_out)]

        all_losses.append(losses)

        # validation set
        prog_bar = tqdm(val_loader)
        losses = []
        with torch.no_grad():
            for anchor, pos, neg in prog_bar:
                anchor_out, pos_out, neg_out = model(anchor, pos, neg)
                val_loss = criterion(anchor_out, pos_out, neg_out)
                losses.append(val_loss.item())
                prog_bar.set_description(f"[E{epoch+1}-v] L={sum(losses)/len(losses):.3f}")
                distances += [d.item() for d in pairwise_distance_ignore_batch_dim(anchor_out, pos_out)]
                distances += [d.item() for d in pairwise_distance_ignore_batch_dim(anchor_out, neg_out)]

        all_validation_losses.append(losses)

    # Save model
    min_distance = min(distances)
    max_distance = max(distances)
    print('Saving model', MODEL_FILE, 'mindist:', min_distance, 'maxdist:', max_distance)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'min_distance': min_distance,
        'max_distance': max_distance,
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

        # second half of dataset are random negatives (probably)
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


def load_model():
    # TODO: use torch script instead of loading full model
    model = SiameseTripletNet()
    model_params = torch.load(MODEL_FILE)
    model.load_state_dict(model_params.get('model_state_dict'))
    min_distance = model_params.get('min_distance')
    max_distance = model_params.get('max_distance')
    model.eval()
    distance_norm = max_distance - min_distance
    return model, distance_norm


def test():
    dataset = RhymesTestDataset()
    loader = DataLoader(dataset, shuffle=True, num_workers=WORKERS)
    model, distance_norm = load_model()
    prog_bar = tqdm(loader, "Test Rhymes")

    correct = []
    wrong = []

    for batch in prog_bar:
        text1, text2, label = batch[0][0], batch[1][0], batch[2].item()
        pred, distance, _ = predict(text1, text2, model=model, distance_norm=distance_norm)

        if (pred and label == 1.0) or (not pred and label == 0.0):
            correct.append((text1, text2, pred, distance))
        else:
            wrong.append((text1, text2, pred, distance))

        prog_bar.set_description(f"√: {len(correct)} X: {len(wrong)} "
                                 f"%: {(len(correct)/(len(correct)+len(wrong))*100):.1f}")

    # print stats and wrong predictions for inspection
    for text1, text2, pred, distance in wrong:
        print('[X]', f'[PRED={"Y" if pred else "N"}]', text1, '=>', text2, f'[{distance:.3f}]')

    total = len(correct) + len(wrong)
    pct = (len(correct) / total) * 100
    wrong_distances = [x[3] for x in wrong]
    correct_distances = [x[3] for x in correct]

    print("\n\nCorrect:", len(correct), "| Wrong:", len(wrong), "| Pct:", f"{pct:.3f}%")
    print("Avg wrong distance:", sum(wrong_distances) / len(wrong))
    print("Min wrong distance:", min(wrong_distances))
    print("Max wrong distance:", max(wrong_distances))
    print("Avg correct distance:", sum(correct_distances) / len(correct) + 1)
    print("Min correct distance:", min(correct_distances))
    print("Max correct distance:", max(correct_distances))


DISTANCE_LABELS = (
    (.1, "Perfect rhyme"),
    (.3, "Slant rhyme"),
    (.5, "Weak rhyme"),
    (.6, "Near rhyme"),
    (.7, "Unlikely rhyme"),
    (1.0, "Not a rhyme"),
)

def predict(text1, text2, model=None, distance_norm=1.0):
    if not model:
        model, distance_norm = load_model()

    anchor_vec, other_vec = make_rhyme_tensors(text1, text2)
    # fake a batch dimension here
    anchor_vec = anchor_vec.unsqueeze(0)
    other_vec = other_vec.unsqueeze(0)

    anchor_out, other_out = model(anchor_vec, other_vec)
    distance = pairwise_distance_ignore_batch_dim(anchor_out, other_out).item()
    distance /= distance_norm

    pred = distance < .5
    label = None
    for thresh, val in DISTANCE_LABELS:
        if distance <= thresh:
            label = val
            break

    return pred, distance, label


def make_training_data():
    '''Output siamese neural net training data triples to CSV file
    '''
    rw = RandomWord()

    lines = []
    for _ in tqdm(range(DATA_TOTAL_SIZE), "Generating rhyme triples"):
        anchor = None
        positive = None

        # lookup a positive value using datamuse
        while positive is None:
            anchor = utils.normalize_lyric(rw.word())
            if anchor in lines:
                continue
            rhymes = utils.get_datamuse_rhymes(anchor, cache_only=DATAMUSE_CACHED_ONLY)
            if rhymes:
                # get the lowest scoring result to encourage real-world-esque rhymes
                positive = min(rhymes, key=lambda x: x.get('score', 1000))['word'] or None
                if positive == anchor or (positive == positive + 's'):
                    positive = None

        positive = utils.normalize_lyric(positive)
        negative = utils.normalize_lyric(rw.word())

        # check IPA length doesn't exceed max length of net
        if any([len(utils.get_ipa_text(w)) > MAX_LEN - 1 for w in [anchor, positive, negative]]):
            continue

        lines.append((anchor, positive, negative))

    with open(TRAIN_FILE, 'w') as f:
        f.write('\n'.join(';'.join(l) for l in lines))
