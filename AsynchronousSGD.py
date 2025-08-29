import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
import numpy as np
import os
import urllib.request
import sklearn.datasets

############################
# Dataset preparation
############################
class LibSVMDataset(torch.utils.data.Dataset):
    def __init__(self, url, dataset_path, download=False, dimensionality=None, classes=None):
        self.url = url
        self.dataset_path = dataset_path
        self._dimensionality = dimensionality

        self.filename = os.path.basename(url)
        self.dataset_type = os.path.basename(os.path.dirname(url))

        if not os.path.isfile(self.local_filename):
            if download:
                print(f"Downloading {url}")
                self._download()
            else:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it."
                )
        else:
            print("Files already downloaded")

        self.data, y = sklearn.datasets.load_svmlight_file(self.local_filename)

        sparsity = self.data.nnz / (self.data.shape[0] * self.data.shape[1])
        if sparsity > 0.1:
            self.data = self.data.todense().astype(np.float32)
            self._is_sparse = False
        else:
            self._is_sparse = True

        # convert labels to [0, 1]
        if classes is None:
            classes = np.unique(y)
        self.classes = np.sort(classes)
        self.targets = torch.zeros(len(y), dtype=torch.int64)
        for i, label in enumerate(self.classes):
            self.targets[y == label] = i

        self.class_to_idx = {cl: idx for idx, cl in enumerate(self.classes)}

        super().__init__()

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_features(self):
        return self.data.shape[1]
    

    def __getitem__(self, idx):
        if self._is_sparse:
            x = torch.from_numpy(self.data[idx].todense().astype(np.float32)).flatten()
        else:
            x = torch.from_numpy(self.data[idx]).flatten()
        y = self.targets[idx]

        if self._dimensionality is not None:
            if len(x) < self._dimensionality:
                x = torch.cat([x, torch.zeros([self._dimensionality - len(x)], dtype=x.dtype, device=x.device)])
            elif len(x) > self._dimensionality:
                raise RuntimeError("Dimensionality is set wrong.")

        return x, y

    def __len__(self):
        return len(self.targets)

    @property
    def local_filename(self):
        return os.path.join(self.dataset_path, self.dataset_type, self.filename)

    def _download(self):
        os.makedirs(os.path.dirname(self.local_filename), exist_ok=True)
        urllib.request.urlretrieve(self.url, filename=self.local_filename)


class RCV1(LibSVMDataset):
    def __init__(self, split, download=False, dataset_path=None):
        if split == "train":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2"
        elif split == "test":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2"
        else:
            raise RuntimeError(f"Unavailable split {split}")
        super().__init__(url=url, download=download, dataset_path=dataset_path)


class GISETTE(LibSVMDataset):
    def __init__(self, split, download=False, dataset_path=None):
        if split == "train":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2"
        elif split == "test":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.t.bz2"
        else:
            raise RuntimeError(f"Unavailable split {split}")
        super().__init__(url=url, download=download, dataset_path=dataset_path)

class ijcnn(LibSVMDataset):
    def __init__(self, split, download=False, dataset_path=None):
        if split == "train":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2"
        elif split == "test":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2"
        else:
            raise RuntimeError(f"Unavailable split {split}")
        super().__init__(url=url, download=download, dataset_path=dataset_path)

# Data loading
def load_data(dataset_name, dataset_path, split_type):
    if split_type == 'train':
        if dataset_name == 'cifar10':
          train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
          return CIFAR10(root=dataset_path, train=True, download=True, transform=train_transform)
        elif dataset_name == 'mnist':
          train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
          return MNIST(root=dataset_path, train=True, download=True, transform=train_transform)
        elif dataset_name == 'rcv1':
            return RCV1("train", download=True, dataset_path=dataset_path)
        elif dataset_name == 'gisette':
            return GISETTE("train", download=True, dataset_path=dataset_path)
        elif dataset_name == 'ijcnn':
            return ijcnn("train", download=True, dataset_path=dataset_path)

    elif split_type == 'test':
        if dataset_name == 'cifar10':
          test_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
          return CIFAR10(root=dataset_path, train=False, download=True, transform=test_transform)
        elif dataset_name == 'mnist':
          test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
          return MNIST(root=dataset_path, train=False, download=True, transform=test_transform)
        elif dataset_name == 'rcv1':
            return RCV1("test", download=True, dataset_path=dataset_path)
        elif dataset_name == 'gisette':
            return GISETTE("test", download=True, dataset_path=dataset_path)
        elif dataset_name == 'ijcnn':
            return ijcnn("test", download=True, dataset_path=dataset_path)

############################
# Model and loss functions
############################
def mse_loss(pred, target, num_classes=2):
    target = target.long()
    target = nn.functional.one_hot(target, num_classes=num_classes).float()
    f = nn.MSELoss()
    return f(pred, target)

def hinge_loss(pred, target, q=1.5):
    """q-norm hinge loss"""
    target = target.long()
    target = nn.functional.one_hot(target, num_classes=2).float()
    margin = 1 - (2 * target - 1) * (2 * pred - 1)  # make sure the label values in [-1,1]
    loss = torch.clamp(margin, min=0) ** q
    return loss.mean()

class Linear_RCV1(nn.Module):
    def __init__(self, loss='mse',q=1.5, random_seed=0):
        super(Linear_RCV1, self).__init__()
        gen = torch.Generator().manual_seed(42+random_seed)
        self.fc1 = nn.Linear(47236, 2)
        nn.init.xavier_normal_(self.fc1.weight, generator=gen)
        nn.init.constant_(self.fc1.bias, 0.01)
        if loss == 'mse':
            self.loss = lambda pred, target: mse_loss(pred, target)
        elif loss == 'hingeloss':
            if q is None:
                raise ValueError("q must be specified for hinge loss")
            self.loss = lambda pred, target: hinge_loss(pred, target, q)
        else:
            raise ValueError("Unsupported loss function. Use 'mse' or 'hingeloss'.")

    def forward(self, x, target):
        x = x.view(-1, 47236)
        output = self.fc1(x)
        loss = self.loss(output, target)
        return output, loss

class Linear_GISETTE(nn.Module):
    def __init__(self, loss='mse',q=1.5, random_seed=0):
        super(Linear_GISETTE, self).__init__()
        gen = torch.Generator().manual_seed(42+random_seed)
        self.fc1 = nn.Linear(5000, 2)
        nn.init.xavier_normal_(self.fc1.weight, generator=gen)
        nn.init.constant_(self.fc1.bias, 0.01)
        if loss == 'mse':
            self.loss = lambda pred, target: mse_loss(pred, target)
        elif loss == 'hingeloss':
            if q is None:
                raise ValueError("q must be specified for hinge loss")
            self.loss = lambda pred, target: hinge_loss(pred, target, q)
        else:
            raise ValueError("Unsupported loss function. Use 'mse' or 'hingeloss'.")

    def forward(self, x, target):
        output = self.fc1(x)
        loss = self.loss(output, target)
        return output, loss

class FCNET_MNIST(nn.Module):
    def __init__(self, loss='mse',q=1.5, random_seed=0):
        super(FCNET_MNIST, self).__init__()
        gen = torch.Generator().manual_seed(42+random_seed)
        self.fc1 = nn.Linear(32*32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(layer.weight, generator=gen)
            nn.init.constant_(layer.bias, 0.01)
        self.loss = nn.CrossEntropyLoss()
        # print("Only support CE-Loss!")

    def forward(self, x, target):
        target = target.long()
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        loss = self.loss(output, target)
        return output, loss

class Linear_ijcnn(nn.Module):
    def __init__(self, loss='mse',q=1.5, random_seed=0):
        super(Linear_ijcnn, self).__init__()
        gen = torch.Generator().manual_seed(42+random_seed)
        self.fc1 = nn.Linear(22, 2)
        nn.init.kaiming_normal_(self.fc1.weight, generator=gen)
        nn.init.constant_(self.fc1.bias, 0.01)
        if loss == 'mse':
            self.loss = lambda pred, target: mse_loss(pred, target)
        elif loss == 'hingeloss':
            if q is None:
                raise ValueError("q must be specified for hinge loss")
            self.loss = lambda pred, target: hinge_loss(pred, target, q)
        else:
            raise ValueError("Unsupported loss function. Use 'mse' or 'hingeloss'.")

    def forward(self, x, target):
        x = x.view(-1, 22)
        output = self.fc1(x)
        loss = self.loss(output, target)
        return output, loss

def load_model(model_name, loss='mse', q=None, random_seed=0):
    if model_name == 'linear_rcv1':
        model = Linear_RCV1(loss=loss, q=q, random_seed=random_seed)
        return model
    elif model_name == 'linear_gisette':
        model = Linear_GISETTE(loss=loss, q=q, random_seed=random_seed )
        return model
    elif model_name == 'fcnet_mnist':
        model = FCNET_MNIST(loss=loss, q=q, random_seed=random_seed)
        return model
    elif model_name == 'linear_ijcnn':
        model = Linear_ijcnn(loss=loss, q=q, random_seed=random_seed)
        return model
    else:
        raise ValueError("Unsupported model name. Use 'linear_rcv1' or 'linear_gisette', 'fcnet_mnist', 'linear_ijcnn'.")
    
############################
# Data preparation
############################
def create_neibordataset(dataset, n_pairs, seed=None):
    # only work on indices
    if seed is not None:
        np.random.seed(42+seed)
    indices = list(range(len(dataset)))
    print("----Creating neighboring dataset----")
    print(f"Total number of samples: {len(indices)}")
    print(f"Number of pairs: {n_pairs}")
    removed_indices = np.random.choice(indices, n_pairs, replace=False)
    neighbordataset_1 = [i for i in indices if i not in removed_indices]
    datasets_dict = {0: neighbordataset_1.copy()}
    # randomly replace 1 elements from neighbordataset_1 for "n_pairs" times
    replace_indices = np.random.choice(neighbordataset_1, n_pairs, replace=False)
    for i in range(n_pairs):
        new_subset = neighbordataset_1.copy()
        # replace
        idx = new_subset.index(replace_indices[i])
        print(f"Pair {i+1}: Replacing index {replace_indices[i]} with {removed_indices[i]}")
        new_subset[idx] = removed_indices[i]
        datasets_dict[i+1] = new_subset
    return datasets_dict

def distribute_data(n_workers, neighbor_datasets_dict, heter=True, seed=None):
    # neighbor_dataset should be: dict {0:[], 1:[], ...}
    if seed is not None:
        np.random.seed(42+seed)
    print("----Distributing data----")
    distributed = {}
    combined_indices = None
    for key, subset in neighbor_datasets_dict.items():
        if combined_indices is None:
            combined_indices = np.arange(len(subset))
            np.random.shuffle(combined_indices)
        shuffled_subset = [subset[i] for i in combined_indices]
        chunk_size = len(combined_indices) // n_workers
        distributed_subset = {}
        for worker_id in range(n_workers):
            start_idx = worker_id * chunk_size
            end_idx = (worker_id + 1) * chunk_size if worker_id != n_workers - 1 else len(combined_indices)
            if not heter:
                distributed_subset[worker_id] = shuffled_subset
            else:
                distributed_subset[worker_id] = [shuffled_subset[i] for i in range(start_idx, end_idx)]
        distributed[key] = distributed_subset
    return distributed  # {0: {worker_id: ...}, 1: {...}, ...}

class DistributedSampler(Sampler):
    def __init__(self, indices, worker_id=None, seed=0):
        # super(DistributedSampler, self).__init__(indices)
        self.indices = indices
        self.worker_id = worker_id
        self.seed = seed

    def set_seed(self, seed):
        self.seed = seed

    def update_seed(self):
        self.seed += 1

    def __iter__(self):
        seed = int(self.worker_id) * 10000 + int(self.seed)
        indices = torch.tensor(self.indices)
        shuffled_indices = indices[torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))]
        return iter(shuffled_indices.tolist())

    def __len__(self):
        return len(self.indices)
    
class TestSampler(Sampler):
    def __init__(self, length, max_length, seed):
        # super(TestSampler, self).__init__(length)
        self.length = length
        self.max_length = max_length
        self.seed = seed
        self.indices = torch.randperm(self.length, generator=torch.Generator().manual_seed(self.seed))[:self.max_length]

    def __iter__(self):
        return iter(self.indices.tolist())

    def __len__(self):
        return len(self.indices)

############################
# Information logging
############################
class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class Logger:
    def __init__(self, log_dir, model_name='', dataset_name='', loss_name='', lr=0.01, delay=0, bs=1):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.filename = f"log_{model_name}_{dataset_name}_{loss_name}_lr{lr:.0e}_bs{bs}_delay{delay}.txt"
        self.file_path = os.path.join(log_dir, self.filename)

    def update(self, iteration, delay, train_loss, test_loss, stability):
        with open(self.file_path, 'a') as f:
            f.write(f"Iteration: {iteration}, Delay: {delay}, Train Loss: {train_loss}, Test Loss: {test_loss}, Stability: {stability}\n")

    def savelog(self):
        pass

class DataRecorder:
    def __init__(self, rec_dir='', model_name='', dataset_name='', loss_name='', lr=0.01, delay=0, bs=1):
        self.rec_dir = rec_dir
        if not os.path.exists(rec_dir):
            os.makedirs(rec_dir)
        self.filename = f"{model_name}_{dataset_name}_{loss_name}_lr{lr:.0e}_bs{bs}_delay{delay}.pt"
        self.data = {'iteration': [],
                     'delay': [],
                     'train_loss': [],
                     'test_loss': [],
                     'stability': []}
        
    def update(self, iteration, delay, train_loss, test_loss, stability):
        self.data['iteration'].append(iteration)
        self.data['delay'].append(delay)
        self.data['train_loss'].append(train_loss)
        self.data['test_loss'].append(test_loss)
        self.data['stability'].append(stability)

    def save(self):
        filepath = os.path.join(self.rec_dir, self.filename)
        torch.save(self.data, filepath)
        print(f"Data saved to {filepath}")

class ModelCheckpoint:
    def __init__(self, checkpoint_dir, model_name='', dataset_name='', loss_name='', lr=0.01, delay=0, bs=1):
        self.checkpoint_dir =  checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.model_name=model_name
        self.dataset_name=dataset_name
        self.loss_name=loss_name
        self.lr=lr
        self.delay=delay
        self.bs=bs

    def save(self, model, iteration):
        self.filename= f"{self.model_name}_iter{iteration}_{self.dataset_name}_{self.loss_name}_lr{self.lr:.0e}_bs{self.bs}_delay{self.delay}.pt"
        self.filepath = os.path.join(self.checkpoint_dir, self.filename)
        torch.save(model.state_dict(), self.filepath)
        print(f"Model saved to {self.filepath} at iteration {iteration}")

############################
# Server class
############################
class Server:
    def __init__(self, device='cuda', train_type='fixed', n_pairs=1, num_workers=1, batch_size=1,
                  dataset_name=None, dataset_path=None, model_name=None, loss_name='mse', lr=0.01, 
                  iterations=0, evaluation_time=0, random_seed=0, log_dir=None, checkpoint_dir=None, rec_dir=None):
        # train_type = 'fixed' or 'random'
        self.device = device
        self.train_type = train_type
        self.n_pairs = n_pairs  # number of pairs of neighboring datasets
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.loss_name = loss_name
        self.lr=lr
        self.iterations = iterations
        self.evaluation_time = evaluation_time
        self.batch_size = batch_size
        self.logger = Logger(log_dir, model_name=model_name, dataset_name=dataset_name,
                            loss_name=loss_name, lr=lr, delay=num_workers-1, bs=batch_size)
        self.checkpoint = ModelCheckpoint(checkpoint_dir, model_name=model_name, dataset_name=dataset_name,
                                          loss_name=loss_name, lr=lr, delay=num_workers-1, bs=batch_size)
        self.datarecorder = DataRecorder(rec_dir=rec_dir, model_name=model_name, dataset_name=dataset_name,
                                          loss_name=loss_name, lr=lr, delay=num_workers-1, bs=batch_size)
        self.iteration = 1 #iteration starts from 1
        self.delay = 0 # delay factor: if train_type ='fixed', the delay factor = num_workers - 1
        self.random_seed=random_seed

        # Load datasets and models
        self.train_dataset = load_data(dataset_name, dataset_path, 'train')
        self.test_dataset = load_data(dataset_name, dataset_path, 'test')
        self.neighbor_datasets = create_neibordataset(self.train_dataset, self.n_pairs, seed=self.random_seed)
        self.models = {i: load_model(model_name, loss_name) for i in range(self.n_pairs+1)}  #create n_"pairs+1" models
        self.optimizers = {i: optim.SGD(self.models[i].parameters(), lr=self.lr) for i in range(self.n_pairs+1)}

        # Set device
        if self.device == 'cuda':
            if torch.cuda.is_available():
                for i in range(self.n_pairs+1):
                    self.models[i].to(torch.device(self.device))
            else:
                raise RuntimeError("CUDA is not available. Please set device to 'cpu'.")
        elif self.device == 'cpu':
            for i in range(self.n_pairs+1):
                self.models[i].to(torch.device(self.device))
        else:
            raise ValueError("Unsupported device. Use 'cuda' or 'cpu'.")    

        # Distribute data to workers
        # Distributed_datasets is a turple with (dict1, dict2) 
        if self.train_type == 'fixed':
            self.current_worker_id = 0
            self.distributed_datasets = distribute_data(num_workers, self.neighbor_datasets, heter=False, seed=self.random_seed)
        elif self.train_type == 'random':
            self.distributed_datasets = distribute_data(num_workers, self.neighbor_datasets, heter=True, seed=self.random_seed)
        self.workers = [Worker(worker_id, bs=self.batch_size, distributed_datasets=self.distributed_datasets, n_pairs=self.n_pairs, server=self) for worker_id in range(num_workers)]

    def train(self):
        while self.iteration <= self.iterations:
            if self.train_type == 'fixed':
                self.step_fixeddelay()
            elif self.train_type == 'random':
                self.step()
            else:
                raise ValueError("Unsupported training type. Use 'fixed' or 'random'.")
            if self.iteration % 1000 == 0:
                print(f"Iteration: {self.iteration}, Current Worker ID: {self.current_worker_id}, Delay: {self.delay}")
            if self.iteration % self.evaluation_time == 0 or self.iteration == self.iterations or self.iteration == 10:
                self.evaluation()
            self.iteration += 1

    def step_fixeddelay(self):
        selected_worker = self.workers[self.current_worker_id]
        if self.current_worker_id < self.num_workers - 1:
            self.current_worker_id += 1
        else:   
            self.current_worker_id = 0
        
        iteration_last = selected_worker.iteration_last
        self.delay = self.iteration - iteration_last - 1

        # Update model parameters using gradients from the worker
        upd = [True] * (self.n_pairs + 1)
        for n in range(self.n_pairs + 1):
            self.optimizers[n].zero_grad(set_to_none=True)
            for Sparam, Wparam in zip(self.models[n].parameters(), selected_worker.models[n].parameters()):
                if Wparam.grad is None:
                    upd[n] = False
                    continue
                Sparam.grad = Wparam.grad.detach().clone()
            if upd[n]:
                self.optimizers[n].step()

        # Send updated models back to the worker
        selected_worker.update_models()
        selected_worker.gradient_calculate()

    # def step(self):
    #     inactive_workers = [w for w in self.workers if not w.active]
    #     if not inactive_workers:
    #         return
    #     selected_worker = np.random.choice(inactive_workers)
    #     gradient_info = selected_worker.gradient
    #     iteration_last = selected_worker.iteration_last
    #     delay = self.iteration - iteration_last

    #     # Update model parameters using gradients from the worker
    #     for param, grad in zip(self.model1.parameters(), gradient_info['model1']):
    #         param.grad = grad
    #     self.optimizer1.step()
    #     for param, grad in zip(self.model2.parameters(), gradient_info['model2']):
    #         param.grad = grad
    #     self.optimizer2.step()

    #     # Send updated models back to the worker
    #     selected_worker.update_models(self.model1.state_dict(), self.model2.state_dict())
    #     selected_worker.gradient_calculate()

    def evaluation(self):
        # slow evaluation
        train_loss1 = self.evaluate(self.models[0], self.train_dataset)
        test_sampler = None
        if len(self.train_dataset) <= 10*len(self.test_dataset):
            test_sampler = TestSampler(len(self.test_dataset), max_length=len(self.train_dataset), seed=self.iteration+self.random_seed)
        else:
            test_sampler = None
        test_loss1 = self.evaluate(self.models[0], self.test_dataset, test_sampler)

        # #fast evaluation
        # eval_tr_sampler = TestSampler(self.train_dataset.num_samples, max_length=10, seed=self.iteration)
        # eval_te_sampler = TestSampler(self.test_dataset.num_samples, max_length=10, seed=self.iteration)
        # train_loss1 = self.evaluate(self.model1, self.train_dataset, eval_tr_sampler)
        # test_loss1 = self.evaluate(self.model1, self.test_dataset, eval_te_sampler)
        stability = self.stability_calculate()
        print(f"Iteration: {self.iteration}, Delay: {self.delay:.0f}, Train Loss: {train_loss1:.6f}, Test Loss: {test_loss1:.6f}, Generalization Error: {test_loss1 - train_loss1:.6f}, Stability: {stability:.6f}")
        self.logger.update(self.iteration, self.delay, train_loss1, test_loss1, stability)
        self.datarecorder.update(self.iteration, self.delay, train_loss1, test_loss1, stability)
        if self.iteration == self.iterations:
            self.datarecorder.save()
            self.checkpoint.save(self.models[0], self.iteration)

    def evaluate(self, model, dataset, sampler=None):
        if sampler is None:
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        else:
            dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)
        model.eval()
        total_loss = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _ , loss = model(inputs, labels)
                total_loss += loss.item() * inputs.size(0)
                total += labels.size(0)
        model.train()
        return total_loss / total

    def stability_calculate(self):
        diff_norm = 0
        params1 = dict(self.models[0].named_parameters())
        for n in range(self.n_pairs):
            params2 = dict(self.models[n+1].named_parameters())
            diff_norm = diff_norm + sum(torch.norm(params1[name] - params2[name]).item() ** 2 for name in params1.keys()) ** 0.5
        return diff_norm / self.n_pairs

############################
# Worker class
############################
class Worker:
    def __init__(self, worker_id, bs, distributed_datasets, n_pairs, server):
        self.worker_id = worker_id
        self.active = True
        self.n_pairs = n_pairs
        self.iteration_last = 0
        self.batch_size = bs
        self.server = server
        self.device = server.device
        self.models = {i: load_model(server.model_name, server.loss_name) for i in range(self.n_pairs+1)}
        for i in range(self.n_pairs+1):
            self.models[i].to(self.device)
        self.ini_seed = server.random_seed
        self.samplers = {
                i: DistributedSampler(distributed_datasets[i][worker_id], worker_id=self.worker_id, seed=self.ini_seed) for i in range(self.n_pairs+1)
            }
        self.dataloaders = {
                i: DataLoader(server.train_dataset, batch_size=self.batch_size, sampler=self.samplers[i]) for i in range(self.n_pairs+1)
            }
        self.generators = {
                i: enumerate(self.dataloaders[i]) for i in range(self.n_pairs+1)
            }

    def next_data(self, model_name):
        try:
            _ , (data, labels) = next(self.generators[model_name])
        except StopIteration:
            self.samplers[model_name].update_seed()
            self.dataloaders[model_name] = DataLoader(self.server.train_dataset, batch_size=self.batch_size, sampler=self.samplers[model_name])
            self.generators[model_name] = enumerate(self.dataloaders[model_name])
            _ , (data, labels) = next(self.generators[model_name])

        return data.to(self.device), labels.to(self.device)

    def gradient_calculate(self):
        for i in range(self.n_pairs+1):
            model = self.models[i]
            model.zero_grad()
            inputs, labels = self.next_data(i)
            _ , loss = model(inputs, labels)
            loss.backward()
        self.iteration_last = self.server.iteration
        self.active = False

    def update_models(self):
        for i in range(self.n_pairs+1):
            self.models[i].load_state_dict(self.server.models[i].state_dict())
        self.active = True

############################
# Main function
############################
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_type = 'fixed'  # or 'random' 
    n_pairs = 10  # Number of pairs of neighboring datasets
    # num_workers = 41  # Number of workers
    dataset_name = 'rcv1'  # or 'rcv1', 'cifar10', 'mnist', 'gisette', 'ijcnn'
    dataset_path = './data'
    model_name = 'linear_rcv1' # or 'linear_rcv1', 'linear_gisette', 'fcnet_mnist'
    loss_name = 'mse'  # or 'hingeloss'
    lr = 1e-2   #2e-5
    iterations = 30000
    batch_size = 16
    evaluation_time = 150  # Evaluate every 150 iterations
    random_seed =0
    log_dir_all = ['./rcv1/r1/logs', './rcv1/r2/logs', './rcv1/r3/logs', './rcv1/r4/logs']
    checkpoint_dir_all = ['./rcv1/r1/checkpoints','./rcv1/r2/checkpoints','./rcv1/r3/checkpoints','./rcv1/r4/checkpoints']
    rec_dir_all = ['./rcv1/r1/records', './rcv1/r2/records', './rcv1/r3/records', './rcv1/r4/records']
    for i in [0,1,2]:
        log_dir = log_dir_all[i]
        checkpoint_dir = checkpoint_dir_all[i]
        rec_dir = rec_dir_all[i]
        for num_workers in [201, 81, 61, 41, 21, 1]:
            server = Server(device=device, train_type=train_type, n_pairs=n_pairs, num_workers=num_workers, batch_size=batch_size, dataset_name=dataset_name,
                     dataset_path=dataset_path, model_name=model_name, loss_name=loss_name, lr=lr,
                     iterations=iterations, evaluation_time=evaluation_time, random_seed=random_seed, log_dir=log_dir,checkpoint_dir=checkpoint_dir,
                     rec_dir=rec_dir)
            server.train()
            del server
            gc.collect()
            torch.cuda.empty_cache()
        random_seed +=10
    # n_pairs = 10  # Number of pairs of neighboring datasets
    # # num_workers = 41  # Number of workers
    # dataset_name = 'gisette'  # or 'rcv1', 'cifar10', 'mnist', 'gisette'
    # dataset_path = './data'
    # model_name = 'linear_gisette' # or 'linear_rcv1', 'linear_gisette', 'fcnet_mnist'
    # loss_name = 'mse'  # or 'hingeloss'
    # lr = 1e-5   #2e-5
    # iterations = 5000
    # batch_size = 16
    # evaluation_time = 10  # Evaluate every 10 iterations
    # random_seed = 0
    # log_dir = './gisette/r1/logs'
    # checkpoint_dir = './gisette/r1/checkpoints'
    # rec_dir = './gisette/r1/records'

    # for num_workers in [21, 16, 11, 6, 1]:
