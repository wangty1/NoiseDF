""" Code to load checkpoint and perform model testing. """

import cv2
import torch
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from network import NoiseDF
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import auc
import pickle
from torchvision import datasets
from utils import SiameseDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--dataset', type=str, default='dataset',
                        help='The directory of testing')
    parser.add_argument('-c', '--checkpoints', type=str, default='checkpoints/noise_df.pth',
                        help='The weight of network')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='the size of every batch')
    parser.add_argument('-i', '--image_size', type=int, default=64,
                        help='The size of the picture entering the network')
    parser.add_argument('-d', '--dropout', type=float, default=0.4,
                        help='The ratio of discard.')
    args = parser.parse_args()
    return args


def test(args):
    folder_dataset_face = datasets.ImageFolder(root=args.dataset + '/face')
    folder_dataset_background = datasets.ImageFolder(root=args.dataset + '/background')
    test_dataset = SiameseDataset(imageFolderDataset_face=folder_dataset_face,
                                  imageFolderDataset_background=folder_dataset_background,
                                  transform=None,
                                  should_invert=False
                                  )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    print("Finished Constructing Dataloader.")

    model = NoiseDF(img_size=args.image_size, dropout=args.dropout)

    # Load pre-trained weights.
    checkpoint = torch.load(args.checkpoints)
    model.load_state_dict(checkpoint['state'])
    model.to("cuda")

    y_label = []
    y_score = []
    sigmoid = nn.Sigmoid()

    model.eval()
    total_count = 0
    correct_count = 0
    for m, (face_test, background_test, label_test) in enumerate(test_dataloader):
        face_test = face_test.to("cuda")
        background_test = background_test.to("cuda")
        label_test = label_test.detach().numpy()
        for i in label_test:
            y_label.append(i)

        output_test = model(face_test, background_test)
        output_test = sigmoid(output_test).squeeze(1)
        output_test = output_test.detach().cpu().numpy()
        for k in output_test:
            y_score.append(k)
        for n, pred in enumerate(output_test):
            if pred > 0.5:
                pred = 1
            else:
                pred = 0
            if pred == label_test[n]:
                correct_count += 1

        total_count += output_test.shape[0]

    # Accuracy.
    acc = correct_count * 1.0 / total_count
    print("ACC : ", acc)

    y_real = np.array(y_label)
    y_pros = np.array(y_score)

    # Save the result to a pkl file
    with open('result/y_real_y_pros.pkl', 'wb') as f:
        pickle.dump([y_real, y_pros], f)

    # AUC score.
    fpr, tpr, thresholds = metrics.roc_curve(y_real, y_pros, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print('AUC : ', roc_auc)


if __name__ == "__main__":
    args = parse_args()
    test(args)
