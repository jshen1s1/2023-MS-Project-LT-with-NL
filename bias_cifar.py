from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import copy
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from utils import noisify, noisify_instance,get_img_num_per_cls,gen_imbalanced_data

class CIFAR10_bias(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True,
                 transform=None, transform_eval = None, target_transform=None, 
                 download=False,lt_type=None, lt_ratio=0.01,
                 noise_type=None, noise_rate=0.2, random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.transform_eval = transform_eval
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset='cifar10'
        self.noise_type=noise_type
        self.lt_type = lt_type
        self.noise_rate = noise_rate
        self.lt_ratio = lt_ratio
        self.random_state = random_state
        self.apply_transform_eval = False
        self.t_matrix = None

        self.nb_classes=10
        self.img_num_per_cls = [5000]*10

        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()
            self.true_labels = copy.deepcopy(self.train_labels)
            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

            if (self.lt_type!='None')&(self.noise_type=='None'):
                self.generate_longtail(self.train_data,self.train_labels,self.lt_type,self.lt_ratio,self.nb_classes,self.random_state)
            elif (self.lt_type=='None')&(self.noise_type!='None'):
                self.generate_noisylabels(self.train_data,self.train_labels,self.noise_type,self.noise_rate,self.nb_classes,self.random_state)
            elif (self.lt_type!='None')&(self.noise_type!='None'):
                self.generate_longtail(self.train_data,self.train_labels,self.lt_type,self.lt_ratio,self.nb_classes,self.random_state)
                self.generate_noisylabels(self.train_data,self.train_labels,self.noise_type,self.noise_rate,self.nb_classes,self.random_state)
            else:
                pass

        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC


    def generate_longtail(self,train_data,train_labels,lt_type,lt_ratio,nb_classes,random_state):
        img_num_per_cls = get_img_num_per_cls(train_data,nb_classes,lt_type,lt_ratio)
        self.img_num_per_cls = img_num_per_cls
        longtail_data,longtail_label,indexes = gen_imbalanced_data(train_data, train_labels, img_num_per_cls,random_state)
        self.train_data = longtail_data
        self.train_labels = longtail_label
        self.true_labels = copy.deepcopy(self.train_labels)
        self.indexes = indexes
        self.noise_or_not = np.array([False]*len(self.train_data))

    def generate_noisylabels(self,train_data,train_labels,noise_type,noise_rate,nb_classes,random_state):
        if noise_type in ['symmetric','pairflip']:
            train_labels = np.asarray([[train_labels[i]] for i in range(len(train_labels))])
            train_labels, actual_noise_rate, self.t_matrix = noisify(dataset=self.dataset, train_labels=train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state, nb_classes=nb_classes)
            train_labels = [i[0] for i in train_labels]
            self.train_labels = train_labels
            self.noise_or_not = np.transpose(self.train_labels)!=np.transpose(self.true_labels)
            print('actual noise rate is',actual_noise_rate)
        if noise_type=='instance':
            train_labels, actual_noise_rate = noisify_instance(train_data, train_labels,noise_rate=noise_rate, random_state=random_state)
            print('actual noise rate is',actual_noise_rate)
            self.noise_or_not = np.transpose(self.train_labels)!=np.transpose(self.true_labels)
            self.train_labels = train_labels
        if noise_type=='human_noise':
            train_labels = np.array(torch.load('CIFAR-10_human.pt')['worse_label'])[self.indexes]
            self.train_labels = train_labels
            self.noise_or_not = np.transpose(self.train_labels)!=np.transpose(self.true_labels)
        idx_each_class_noisy = [[] for i in range(self.nb_classes)]
        for i in range(len(self.train_labels)):
            idx_each_class_noisy[self.train_labels[i]].append(i)
            class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(self.nb_classes)]
        self.img_num_per_cls = np.array(class_size_noisy)


    def get_cls_num_list(self):
        return self.img_num_per_cls

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.apply_transform_eval:
            transform = self.transform_eval
        else:
            transform = self.transform 

        if self.transform is not None:
            img = transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        true_target = self.true_labels[index]
        return img, target, true_target, index

    def eval(self):
        self.apply_transform_eval = True

    def train_mode(self):
        self.apply_transform_eval = False

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class CIFAR100_bias(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

    def __init__(self, root, train=True,
                 transform=None, transform_eval = None, target_transform=None,
                 download=False,lt_type=None, lt_ratio=0.01,
                 noise_type=None, noise_rate=0.2, random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.transform_eval = transform_eval
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset='cifar100'
        self.noise_type=noise_type
        self.lt_type = lt_type
        self.noise_rate = noise_rate
        self.lt_ratio = lt_ratio
        self.random_state = random_state
        self.apply_transform_eval = False
        self.t_matrix = None

        self.nb_classes=100
        self.img_num_per_cls = [500]*100

        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()
            self.true_labels = copy.deepcopy(self.train_labels)
            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

            if (self.lt_type!='None')&(self.noise_type=='None'):
                self.generate_longtail(self.train_data,self.train_labels,self.lt_type,self.lt_ratio,self.nb_classes,self.random_state)
            elif (self.lt_type=='None')&(self.noise_type!='None'):
                self.generate_noisylabels(self.train_data,self.train_labels,self.noise_type,self.noise_rate,self.nb_classes,self.random_state)
            elif (self.lt_type!='None')&(self.noise_type!='None'):
                self.generate_longtail(self.train_data,self.train_labels,self.lt_type,self.lt_ratio,self.nb_classes,self.random_state)
                self.generate_noisylabels(self.train_data,self.train_labels,self.noise_type,self.noise_rate,self.nb_classes,self.random_state)
            else:
                pass

        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC


    def generate_longtail(self,train_data,train_labels,lt_type,lt_ratio,nb_classes,random_state):
        img_num_per_cls = get_img_num_per_cls(train_data,nb_classes,lt_type,lt_ratio)
        self.img_num_per_cls = img_num_per_cls
        longtail_data,longtail_label,indexes = gen_imbalanced_data(train_data, train_labels, img_num_per_cls,random_state)
        self.train_data = longtail_data
        self.train_labels = longtail_label
        self.true_labels = copy.deepcopy(self.train_labels)
        self.indexes = indexes

    def generate_noisylabels(self,train_data,train_labels,noise_type,noise_rate,nb_classes,random_state):
        if noise_type in ['symmetric','pairflip']:
            train_labels = np.asarray([[train_labels[i]] for i in range(len(train_labels))])
            train_labels, actual_noise_rate, self.t_matrix = noisify(dataset=self.dataset, train_labels=train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state, nb_classes=nb_classes)
            train_labels = [i[0] for i in train_labels]
            self.train_labels = train_labels
            self.noise_or_not = np.transpose(self.train_labels)!=np.transpose(self.true_labels)
            print('actual noise rate is',actual_noise_rate)
        if noise_type=='instance':
            train_labels, actual_noise_rate = noisify_instance(train_data, train_labels,noise_rate=noise_rate, random_state=random_state)
            print('actual noise rate is',actual_noise_rate)
            self.noise_or_not = np.transpose(self.train_labels)!=np.transpose(self.true_labels)
            self.train_labels = train_labels
        if noise_type=='human_noise':
            train_labels = np.array(torch.load('CIFAR-100_human.pt')['noisy_label'])[self.indexes]
            self.train_labels = train_labels
            self.noise_or_not = np.transpose(self.train_labels)!=np.transpose(self.true_labels)
        idx_each_class_noisy = [[] for i in range(self.nb_classes)]
        for i in range(len(self.train_labels)):
            idx_each_class_noisy[self.train_labels[i]].append(i)
            class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(self.nb_classes)]
        self.img_num_per_cls = np.array(class_size_noisy)


    def get_cls_num_list(self):
        return self.img_num_per_cls

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.apply_transform_eval:
            transform = self.transform_eval
        else:
            transform = self.transform 

        if self.transform is not None:
            img = transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        true_target = self.true_labels[index]
        return img, target, true_target, index

    def eval(self):
        self.apply_transform_eval = True

    def train_mode(self):
        self.apply_transform_eval = False

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)






