'''Rhyme-detecting neural net
'''

import random
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from wonderwords import RandomWord
from songisms import utils

# to be configurable later
MODEL_FILE = './data/rhymes.torch'
TRAIN_FILE = './data/rhymes_train.csv'
DATA_TOTAL_SIZE = 20000
ROWS = 5000
BATCH_SIZE = 64
EPOCHS = 10
WORKERS = 4
THRESHOLD = 0.5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# these can't change right now without also adjusting the network
MAX_LEN = 30
USE_TAILS = False
NORM_SCORE = 11.5  # ???

SCORE_LABELS = [
    [.1, "Perfect Rhyme"],
    [.3, "Slant rhyme"],
    [.5, "Weak rhyme"],
    [.6, "Near rhyme"],
    [.7, "Unlikely rhyme"],
    [1.0, "Not a rhyme"],
]

class RhymesDataset(Dataset):
    def __init__(self, pad_to=MAX_LEN):
        self.pad_to = pad_to
        self.triplets = utils.data.rhymes_train
        if ROWS:
            self.triplets = self.triplets[:ROWS]

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, pos, neg = self.triplets[idx]
        anchor, pos, neg = make_rhyme_tensors(anchor, pos, neg, pad_to=self.pad_to)
        anchor, pos, neg = anchor.transpose(0, 1), pos.transpose(0, 1), neg.transpose(0, 1)
        return anchor, pos, neg


class SiameseTripletNet(torch.nn.Module):
    def __init__(self):
        super(SiameseTripletNet, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv1d(25, 64, kernel_size=3),
            nn.Tanh(),
            nn.BatchNorm1d(64),
            nn.Dropout1d(p=.25),

            nn.Conv1d(64, 64, kernel_size=3),
            nn.Tanh(),
            nn.BatchNorm1d(64),
            nn.Dropout1d(p=.25),

            nn.Conv1d(64, 32, kernel_size=3),
            nn.Tanh(),
            nn.BatchNorm1d(32),
            nn.Dropout1d(p=.25),
        )
        self.fc1 = nn.Linear(768, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)

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
    def __init__(self, margin=1.0):
        self.margin = margin
        super(TripletMarginLoss, self).__init__()

    def forward(self, anchor, pos, neg):
        return nn.functional.triplet_margin_loss(anchor, pos, neg, margin=self.margin)


def train():
    dataset = RhymesDataset()
    train_data, validation_data = random_split(dataset, [0.8, 0.2])
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    val_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)

    model = SiameseTripletNet()
    criterion = TripletMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    all_losses = []
    all_val_losses = []

    for epoch in range(EPOCHS):
        # training data
        loading = tqdm(loader, f"[E{epoch+1}-T] L=...")
        losses_cur_epoch = []
        all_losses.append(0)

        for (anchor, pos, neg) in loading:
            anchor_out, pos_out, neg_out = model(anchor, pos, neg)
            loss = criterion(anchor_out, pos_out, neg_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_cur_epoch.append(loss.item())
            all_losses[-1] = sum(losses_cur_epoch) / len(losses_cur_epoch)
            loading.set_description(f"[E{epoch+1}-T] L={all_losses[-1]:.2f}")

        # validation data
        val_loading = tqdm(val_loader, f"[E{epoch+1}-v] L=...")
        val_losses_cur_epoch = []
        all_val_losses.append(0)

        with torch.no_grad():
            for anchor, pos, neg in val_loading:
                anchor_out, pos_out, neg_out = model(anchor, pos, neg)
                val_loss = criterion(anchor_out, pos_out, neg_out)
                val_losses_cur_epoch.append(val_loss.item())
                all_val_losses[-1] = sum(val_losses_cur_epoch) / len(val_losses_cur_epoch)
                val_loading.set_description(f"[E{epoch+1}-v] L={all_val_losses[-1]:.2f}")

    # Save model
    torch.save(model.state_dict(), MODEL_FILE)

    # plot train/validation losses
    plt.plot(all_losses, label='Training Loss')
    plt.plot(all_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def make_rhyme_tensors(anchor, pos, neg=None, pad_to=MAX_LEN):
    proc_fn = utils.get_ipa_tail if USE_TAILS else utils.get_ipa_text

    if neg is not None:
        anchor_ipa, pos_ipa, neg_ipa = [proc_fn(text) for text in [anchor, pos, neg]]
        anchor_vec, pos_vec, _ = utils.align_vals(anchor_ipa, pos_ipa)
        anchor_vec, neg_vec, _ = utils.align_vals(anchor_ipa, neg_ipa)
        vecs = [anchor_vec, pos_vec, neg_vec]
    else:
        anchor_ipa, pos_ipa = [proc_fn(text) for text in [anchor, pos]]
        anchor_vec, pos_vec, _ = utils.align_vals(anchor_ipa, pos_ipa)
        vecs = [anchor_vec, pos_vec]

    for i, vec in enumerate(vecs):
        # pad
        vec = (['_'] * (pad_to - len(vec))) + [str(c) for c in vec]
        # replace IPA characters with feature arrays
        vec = utils.get_ipa_features_vector(vec)
        # convert to tensor
        vecs[i] = torch.tensor(vec, dtype=torch.float)

    return vecs


class RhymesTestDataset(Dataset):
    def __init__(self):
        labeled_pairs = []
        for rset in utils.data.rhymes:
            pairs = utils.get_rhyme_pairs(';'.join(rset))
            for w1, w2 in pairs:
                if (w1, w2, 1.0) not in labeled_pairs and \
                   (w2, w1, 1.0) not in labeled_pairs:
                    labeled_pairs.append((w1, w2, 1.0))

        # add some random non-rhymes (probably)
        rw = RandomWord()
        target_ct = (len(labeled_pairs) * 1.5)
        while len(labeled_pairs) < target_ct:
            w1, w2 = rw.word(), rw.word()
            if w1 == w2 or (utils.get_ipa_tail(w1) == utils.get_ipa_tail(w2)):
                continue

            if (w1, w2, 0.0) not in labeled_pairs and \
               (w2, w1, 0.0) not in labeled_pairs and \
               (w1, w2, 1.0) not in labeled_pairs and \
               (w2, w1, 1.0) not in labeled_pairs:
                labeled_pairs.append((w1, w2, 0.0))

        print("Loaded/created", len(labeled_pairs), "rhymes")
        self.labeled_pairs = labeled_pairs

    def __len__(self):
        return len(self.labeled_pairs)

    def __getitem__(self, idx):
        return self.labeled_pairs[idx]


def test():
    correct = []
    wrong = []
    test_dataset = RhymesTestDataset()
    loader = DataLoader(test_dataset, shuffle=True, num_workers=WORKERS)

    for text1, text2, label in tqdm(loader, "Test Rhymes"):
        pred, score = predict(text1, text2)
        if (pred and label == 1.0) or (not pred and label == 0.0):
            correct.append((text1, text2, score))
        else:
            wrong.append((text1, text2, score))

    for text1, text2, score in wrong:
        print('[X]', text1, '=>', text2, f"[{score}]")

    total = len(correct) + len(wrong)
    pct = (len(correct) / total) * 100

    print("\n\nCorrect:", len(correct), "Wrong:", len(wrong), "Pct:", f"{pct:.2f}%")
    print("Avg wrong score:", sum([x[2] for x in wrong]) / len(wrong))
    print("Min wrong score:", min([x[2] for x in wrong]))
    print("Max wrong score:", max([x[2] for x in wrong]))
    print("Avg correct score:", sum([x[2] for x in correct]) / len(correct) + 1)
    print("Min correct score:", min([x[2] for x in correct]))
    print("Max correct score:", max([x[2] for x in correct]))


def predict(word1, word2):
    model = SiameseTripletNet()
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()
    anchor_vec, other_vec = make_rhyme_tensors(word1, word2)
    anchor_vec = anchor_vec.transpose(0, 1).unsqueeze(0)
    other_vec = other_vec.transpose(0, 1).unsqueeze(0)
    anchor_out, other_out = model(anchor_vec, other_vec)
    anchor_out = anchor_out.squeeze(0)
    other_out = other_out.squeeze(0)
    distance = F.pairwise_distance(anchor_out, other_out, p=2).item() / NORM_SCORE

    pred = distance < THRESHOLD
    label = None
    for thresh, val in SCORE_LABELS:
        if distance < thresh:
            label = val
            break
    return pred, distance, label



def make_training_data():
    '''Output siamese neural net training data triples to CSV file
    '''
    from rhymes.models import Rhyme

    rhymes = list(Rhyme.objects.filter(level=1).prefetch_related('to_ngram', 'from_ngram'))
    random.shuffle(rhymes)
    lines = set()
    i = 0
    rw = RandomWord()

    while len(lines) < DATA_TOTAL_SIZE:
        i += 1
        r = rhymes[i]
        anchor = r.from_ngram.text
        positive = r.to_ngram.text

        if i % 2 == 0:
            negative = rhymes[random.randint(0, len(rhymes) - 1)].to_ngram.text
        else:
            negative = rw.word()

        if anchor != negative:
            entry = (anchor, positive, negative)
            lines.add(entry)

        pos_vars = utils.make_variants(positive)
        neg_vars = utils.make_variants(negative)
        for pos_var, neg_var in zip(pos_vars[:2], neg_vars[:2]):
            if anchor != pos_var:
                entry = (anchor, pos_var, rw.word())
                lines.add(entry)
            if anchor != neg_var:
                entry = (anchor, positive, neg_var)
                lines.add(entry)

    with open(TRAIN_FILE, 'w') as f:
        f.write('\n'.join(';'.join(l) for l in lines))
