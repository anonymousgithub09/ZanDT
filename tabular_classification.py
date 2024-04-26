import numpy as np
import torch
from pathlib import Path

from torch.nn import BCELoss
from torch.utils.data import DataLoader

from qhoptim.pyt import QHAdam

from src.LT_models import LTBinaryClassifier
from src.monitors import MonitorTree
from src.optimization import train_stochastic, evaluate
# from src.datasets import Dataset, TorchDataset
from src.utils import deterministic

import time
import sys

DATA_NAME = sys.argv[1]
TREE_DEPTH = int(sys.argv[2]) #4
REG = float(sys.argv[3]) #1.4837286400170702
MLP_LAYERS = int(sys.argv[4]) #5
DROPOUT = float(sys.argv[5]) #0.07041761417608994

LR = 0.001
BATCH_SIZE = 512 
EPOCHS = 500

pruning = REG > 0

class TorchDataset(torch.utils.data.Dataset):

    def __init__(self, *data, **options):
        
        n_data = len(data)
        if n_data == 0:
            raise ValueError("At least one set required as input")

        self.data = data
        means = options.pop('means', None)
        stds = options.pop('stds', None)
        self.transform = options.pop('transform', None)
        self.test = options.pop('test', False)
        
        if options:
            raise TypeError("Invalid parameters passed: %s" % str(options))
        
        if means is not None:
            assert stds is not None, "must specify both <means> and <stds>"

            self.normalize = lambda data: [(d - m) / s for d, m, s in zip(data, means, stds)]

        else:
            self.normalize = lambda data: data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        data = self.normalize([s[idx] for s in self.data])
        if self.transform:

            if self.test:
                data = sum([[self.transform.test_transform(d)] * 2 for d in data], [])
            else:
                data = sum([self.transform(d) for d in data], [])
            
        return data


def eval_dataset(data):
    print('classes', np.unique(data.y_test))

    test_losses, train_times, test_times = [], [], []
    for SEED in [1225]:
        deterministic(SEED)

        save_dir = Path("./results/tabular-quantile/") / DATA_NAME / "depth={}/reg={}/mlp-layers={}/dropout={}/seed={}".format(TREE_DEPTH, REG, MLP_LAYERS, DROPOUT, SEED)
        save_dir.mkdir(parents=True, exist_ok=True)

        trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=BATCH_SIZE, num_workers=16, shuffle=True)
        valloader = DataLoader(TorchDataset(data.X_valid, data.y_valid), batch_size=BATCH_SIZE*2, num_workers=16, shuffle=False)
        testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=BATCH_SIZE*2, num_workers=16, shuffle=False)

        model = LTBinaryClassifier(TREE_DEPTH, data.X_train.shape[1], reg=REG)

        # init optimizer
        optimizer = QHAdam(model.parameters(), lr=LR, nus=(0.7, 1.0), betas=(0.995, 0.998))

        # init loss
        loss = BCELoss(reduction="sum")
        criterion = lambda x, y: loss(x.float(), y.float())

        # evaluation criterion => error rate
        eval_criterion = lambda x, y: (x != y).sum()

        # init train-eval monitoring 
        monitor = MonitorTree(pruning, save_dir)

        state = {
            'batch-size': BATCH_SIZE,
            'loss-function': 'BCE',
            'learning-rate': LR,
            'seed': SEED,
            'dataset': DATA_NAME,
            'reg': REG,
        }

        best_val_loss = float("inf")
        best_e = -1
        no_improv = 0
        t0 = time.time()
        for e in range(EPOCHS):
            train_stochastic(trainloader, model, optimizer, criterion, epoch=e, monitor=monitor)

            val_loss = evaluate(valloader, model, {'ER': eval_criterion}, epoch=e, monitor=monitor)
            print("Epoch %i: validation loss = %f\n" % (e, val_loss["ER"]))
            no_improv += 1

            if val_loss["ER"] < best_val_loss:
                best_val_loss = val_loss["ER"]
                best_e = e
                no_improv = 0
                LTBinaryClassifier.save_model(model, optimizer, state, save_dir, epoch=e, val_er=best_val_loss)

            if no_improv == EPOCHS // 5:
                break
        t1 = time.time()
        monitor.close()
        print("best validation error rate (epoch {}): {}\n".format(best_e, best_val_loss))

        model = LTBinaryClassifier.load_model(save_dir)
        t2 = time.time()
        test_loss = evaluate(testloader, model, {'ER': eval_criterion})
        print("test error rate (model of epoch {}): {}\n".format(best_e, test_loss['ER']))
        t3 = time.time()
        test_losses.append(test_loss['ER'])
        train_times.append(t1 - t0)
        test_times.append(t3 - t2)

    print(np.mean(test_losses), np.std(test_losses))
    np.save(save_dir / '../test-losses.npy', test_losses)
    print("Avg train time", np.mean(train_times))
    print("Avg test time", np.mean(test_times))





if DATA_NAME =="syn":
    #@title Synthetic data
    def set_npseed(seed):
        np.random.seed(seed)


    def set_torchseed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    #classification data

    def data_gen_decision_tree(num_data=1000, dim=2, seed=0, w_list=None, b_list=None,vals=None, num_levels=2):        
        set_npseed(seed=seed)

        # Construct a complete decision tree with 2**num_levels-1 internal nodes,
        # e.g. num_levels=2 means there are 3 internal nodes.
        # w_list, b_list is a list of size equal to num_internal_nodes
        # vals is a list of size equal to num_leaf_nodes, with values +1 or 0
        num_internal_nodes = 2**num_levels - 1
        num_leaf_nodes = 2**num_levels
        stats = np.zeros(num_internal_nodes+num_leaf_nodes) #stores the num of datapoints at each node so at 0(root) all data points will be present

        if vals is None: #when val i.e., labels are not provided make the labels dynamically
            vals = np.arange(0,num_internal_nodes+num_leaf_nodes,1,dtype=np.int32)%2 #assign 0 or 1 label to the node based on whether its numbering is even or odd
            vals[:num_internal_nodes] = -99 #we put -99 to the internal nodes as only the values of leaf nodes are counted

        if w_list is None: #if the w values of the nodes (hyperplane eqn) are not provided then generate dynamically
            w_list = np.random.standard_normal((num_internal_nodes, dim))
            w_list = w_list/np.linalg.norm(w_list, axis=1)[:, None] #unit norm w vects
            b_list = np.zeros((num_internal_nodes))

        '''
        np.random.random_sample
        ========================
        Return random floats in the half-open interval [0.0, 1.0).

        Results are from the "continuous uniform" distribution over the
        stated interval.  To sample :math:`Unif[a, b), b > a` multiply
        the output of `random_sample` by `(b-a)` and add `a`::

            (b - a) * random_sample() + a
        '''

    #     data_x = np.random.random_sample((num_data, dim))*2 - 1. #generate the datas in range -1 to +1
    #     relevant_stats = data_x @ w_list.T + b_list #stores the x.wT+b value of each nodes for all data points(num_data x num_nodes) to check if > 0 i.e will follow right sub tree route or <0 and will follow left sub tree route
    #     curr_index = np.zeros(shape=(num_data), dtype=int) #stores the curr index for each data point from root to leaf. So initially a datapoint starts from root but then it can go to right or left if it goes to right its curr index will become 2 from 0 else 1 from 0 then in next iteration from say 2 it goes to right then it will become 6

        data_x = np.random.standard_normal((num_data, dim))
        data_x /= np.sqrt(np.sum(data_x**2, axis=1, keepdims=True))
        relevant_stats = data_x @ w_list.T + b_list
        curr_index = np.zeros(shape=(num_data), dtype=int)
        
        for level in range(num_levels):
            nodes_curr_level=list(range(2**level - 1,2**(level+1)-1  ))
            for el in nodes_curr_level:
    #             b_list[el]=-1*np.median(relevant_stats[curr_index==el,el])
                relevant_stats[:,el] += b_list[el]
            decision_variable = np.choose(curr_index, relevant_stats.T) #based on the curr index will choose the corresponding node value of the datapoint

            # Go down and right if wx+b>0 down and left otherwise.
            # i.e. 0 -> 1 if w[0]x+b[0]<0 and 0->2 otherwise
            curr_index = (curr_index+1)*2 - (1-(decision_variable > 0)) #update curr index based on the desc_variable
            

        bound_dist = np.min(np.abs(relevant_stats), axis=1) #finds the abs value of the minm node value of a datapoint. If some node value of a datapoint is 0 then that data point exactly passes through a hyperplane and we remove all such datapoints
        thres = threshold
        labels = vals[curr_index] #finally labels for each datapoint is assigned after traversing the whole tree

        data_x_pruned = data_x[bound_dist>thres] #to distingush the hyperplanes seperately for 0 1 labels (classification)
        #removes all the datapoints that passes through a node hyperplane
        labels_pruned = labels[bound_dist>thres]
        relevant_stats = np.sign(data_x_pruned @ w_list.T + b_list) #storing only +1 or -1 for a particular node if it is active or not
        nodes_active = np.zeros((len(data_x_pruned),  num_internal_nodes+num_leaf_nodes), dtype=np.int32) #stores node actv or not for a data

        for node in range(num_internal_nodes+num_leaf_nodes):
            if node==0:
                stats[node]=len(relevant_stats) #for root node all datapoints are present
                nodes_active[:,0]=1 #root node all data points active status is +1
                continue
            parent = (node-1)//2
            nodes_active[:,node]=nodes_active[:,parent]
            right_child = node-(parent*2)-1 # 0 means left, 1 means right 1 has children 3,4
            #finds if it is a right child or left of the parent
            if right_child==1:
                nodes_active[:,node] *= relevant_stats[:,parent]>0 #if parent node val was >0 then this right child of parent is active
            if right_child==0:
                nodes_active[:,node] *= relevant_stats[:,parent]<0 #else left is active
            stats = nodes_active.sum(axis=0) #updates the status i.e., no of datapoints active in that node (root has all active then gradually divided in left right)
        return ((data_x_pruned, labels_pruned), (w_list, b_list, vals), stats)


    class Dataset_syn:
        def __init__(self, dataset, data_path='./DATA'):
            if dataset =="syn":
                self.X_train = train_data
                self.y_train = train_data_labels
                self.X_valid = vali_data
                self.y_valid = vali_data_labels
                self.X_test = test_data
                self.y_test = test_data_labels
            self.data_path = data_path
            self.dataset = dataset


    # Define dictionaries
    seed=365
    num_levels=4
    threshold = 0 #data seperation distance
    output_dim=1


    data_configs = [
        {"input_dim": 20, "num_data": 40000},
        {"input_dim": 100, "num_data": 60000},
        # {"input_dim": 500, "num_data": 100000}
    ]

    # Code block to run for each dictionary
    for config in data_configs:
        input_dim = config["input_dim"]
        num_data = config["num_data"]
        print("====================================================================================")
        print("input_dim", input_dim, "num_data", num_data)
        
        
        ((data_x, labels), (w_list, b_list, vals), stats) = data_gen_decision_tree(
                                                    dim=input_dim, seed=seed, num_levels=num_levels,
                                                    num_data=num_data)
        seed_set=seed
        w_list_old = np.array(w_list)
        b_list_old = np.array(b_list)
        print(sum(labels==1))
        print(sum(labels==0))
        print("Seed= ",seed_set)
        num_data = len(data_x)
        num_train= num_data//2
        num_vali = num_data//4
        num_test = num_data//4
        
        train_data = data_x[:num_train,:]
        train_data_labels = labels[:num_train]

        vali_data = data_x[num_train:num_train+num_vali,:]
        vali_data_labels = labels[num_train:num_train+num_vali]

        test_data = data_x[num_train+num_vali :,:]
        test_data_labels = labels[num_train+num_vali :]

        data = Dataset_syn(DATA_NAME)
        eval_dataset(data)

if DATA_NAME =="UCI":
    import random
    import requests
    import os
    from tqdm import tqdm
    import numpy as np
    import gzip
    import shutil
    import tarfile
    import bz2
    import pandas as pd
    import gzip
    import shutil
    import warnings

    from pathlib import Path
    from sklearn.datasets import load_svmlight_file
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_swiss_roll
    from sklearn.preprocessing import QuantileTransformer

    from category_encoders import LeaveOneOutEncoder
    from category_encoders.ordinal import OrdinalEncoder
    import os
    import zipfile
    import shutil
    import urllib.request
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import NearestCentroid
    from scipy.io import arff



    def preprocess_data_adult(data_path):
    # Read the data into a DataFrame
        columns = [
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"
        ]
        df = pd.read_csv(data_path, names=columns, na_values=[" ?"])

        # Drop rows with missing values
        df.dropna(inplace=True)

        # Convert categorical features using Label Encoding
        categorical_columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        # Encode the target variable
        df["income"] = df["income"].apply(lambda x: 1 if x == " >50K" else 0)

        return df

    def preprocess_data_bank_marketing(data):
        # Convert categorical features using Label Encoding
        label_encoders = {}
        for col in data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

        return data

    def preprocess_data_credit_card_defaults(data):
        # Convert categorical features using one-hot encoding
        data = pd.get_dummies(data, columns=["SEX", "EDUCATION", "MARRIAGE"], drop_first=True)

        # Standardize numerical features
        scaler = StandardScaler()
        data[["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1",
            "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2",
            "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]] = scaler.fit_transform(
            data[["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1",
                "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2",
                "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]])

        return data
    

    def fetch_ADULT(data_dir="./ADULT_DATA"):
        print("---------------------ADULT--------------------------------------")
        # Create the data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # URL of the dataset zip file
        url = "https://archive.ics.uci.edu/static/public/2/adult.zip"
        zip_file_path = os.path.join(data_dir, "adult.zip")
        # Download the zip file
        urllib.request.urlretrieve(url, zip_file_path)
        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        # Preprocess the data
        train_data_path = os.path.join(data_dir, "adult.data")
    #     test_data_path = os.path.join(data_dir, "adult.test")
        df_train = preprocess_data_adult(train_data_path)
    #     df_test = preprocess_data_adult(test_data_path)

        # Split the data into train, validation, and test sets
        X = df_train.drop("income", axis=1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df_train["income"]
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
    #     X_test = df_test.drop("income", axis=1)
    #     y_test = df_test["income"]

        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_val = (y_val.values.reshape(-1) == 1).astype('int64')
        # Remove the zip file
        os.remove(zip_file_path)

        # Remove the extracted directory and its contents using shutil.rmtree()
        shutil.rmtree(data_dir)

        return dict(
            X_train=X_train.astype('float32'), y_train=y_train, X_valid=X_val.astype('float32'), y_valid=y_val, X_test=X_test.astype('float32'), y_test=y_test
        )

    def fetch_bank_marketing(data_dir="./BANK"):
        print("---------------------BANK--------------------------------------")
        # Create the data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # URL of the dataset zip file
        url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
        zip_file_path = os.path.join(data_dir, "bank_marketing.zip")

        # Download the zip file
        urllib.request.urlretrieve(url, zip_file_path)

        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        
        zip_file_path_bank_add = os.path.join(data_dir, "bank-additional.zip")
        with zipfile.ZipFile(zip_file_path_bank_add, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        # Get the extracted directory path
        extracted_dir = os.path.join(data_dir, "bank-additional")

        # Read the dataset
        data = pd.read_csv(os.path.join(extracted_dir, "bank-additional-full.csv"), sep=';')

        # Preprocess the data
        data = preprocess_data_bank_marketing(data)

        # Split the data into train, validation, and test sets
        X = data.drop("y", axis=1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = data["y"]
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_val = (y_val.values.reshape(-1) == 1).astype('int64')
        # Remove the zip file
        os.remove(zip_file_path)

        # Remove the extracted directory and its contents
        shutil.rmtree(data_dir)

        return dict(
            X_train=X_train.astype('float32'), y_train=y_train,X_test=X_test.astype('float32'), y_test=y_test, X_valid = X_val.astype('float32'), y_valid = y_val
        )

    def fetch_credit_card_defaults(data_dir="./CREDIT"):
        print("---------------------CREDIT--------------------------------------")
        # Create the data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # URL of the dataset zip file
        url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
        zip_file_path = os.path.join(data_dir, "credit_card_defaults.zip")

        # Download the zip file
        urllib.request.urlretrieve(url, zip_file_path)

        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    #     # Get the extracted directory path
    #     extracted_dir = os.path.join(data_dir, "default+of+credit+card+clients")

        # Read the dataset
        data = pd.read_excel(os.path.join(data_dir, "default of credit card clients.xls"), skiprows=1)

        # Preprocess the data
        data = preprocess_data_credit_card_defaults(data)

        # Split the data into train, validation, and test sets
        X = data.drop("default payment next month", axis=1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = data["default payment next month"]
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_val = (y_val.values.reshape(-1) == 1).astype('int64')

        # Remove the zip file
        os.remove(zip_file_path)

        # Remove the extracted directory and its contents
        shutil.rmtree(data_dir)

        return dict(
            X_train=X_train.astype('float32'), y_train=y_train, X_valid=X_val.astype('float32'), y_valid=y_val , X_test=X_test.astype('float32'), y_test=y_test
        )


    def fetch_gamma_telescope(data_dir="./TELESCOPE"):
        print("---------------------TELESCOPE--------------------------------------")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # URL of the dataset zip file
        url = "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip"
        zip_file_path = os.path.join(data_dir, "magic_gamma_telescope.zip")

        # Download the zip file
        urllib.request.urlretrieve(url, zip_file_path)

        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Load the data from CSV
        data_path = os.path.join(data_dir, "magic04.data")
        columns = [
            "fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long",
            "fM3Trans", "fAlpha", "fDist", "class"
        ]
        data = pd.read_csv(data_path, header=None, names=columns)
        
        # Convert the class labels to binary format (g = gamma, h = hadron)
        data["class"] = data["class"].map({"g": 1, "h": 0})
        
        # Split the data into features (X) and target (y)
        X = data.drop("class", axis=1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = data["class"]
        
        # Split the data into train, test, and validation sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_valid = (y_valid.values.reshape(-1) == 1).astype('int64')
        
        # Create a dictionary to store the data splits
        data_splits = {
            "X_train": X_train.astype('float32'), "y_train": y_train,
            "X_valid": X_valid.astype('float32'), "y_valid": y_valid,
            "X_test": X_test.astype('float32'), "y_test": y_test
        }
        
        # Remove the zip file
        os.remove(zip_file_path)

        # Remove the extracted directory and its contents
        shutil.rmtree(data_dir)
        
        return data_splits

    def fetch_rice_dataset(data_dir="./RICE"):
        print("---------------------RICE--------------------------------------")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # URL of the dataset zip file
        url = "https://archive.ics.uci.edu/static/public/545/rice+cammeo+and+osmancik.zip"
        zip_file_path = os.path.join(data_dir, "rice_dataset.zip")

        # Download the zip file
        urllib.request.urlretrieve(url, zip_file_path)

        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
            
        # Load the data from CSV
        arff_file_name = os.path.join(data_dir, "Rice_Cammeo_Osmancik.arff")

        
        # Load the ARFF file using SciPy
        data, meta = arff.loadarff(arff_file_name)
        
        df = pd.DataFrame(data)
        df["Class"] = df["Class"].map({b'Cammeo': 1, b'Osmancik': 0})
        
        # Split the data into features (X) and target (y)
        X = df.drop("Class", axis=1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df["Class"]
        
        # Split the data into train, test, and validation sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_valid = (y_valid.values.reshape(-1) == 1).astype('int64')
        
        # Create a dictionary to store the data splits
        data_splits = {
            "X_train": X_train.astype('float32'), "y_train": y_train,
            "X_valid": X_valid.astype('float32'), "y_valid": y_valid,
            "X_test": X_test.astype('float32'), "y_test": y_test
        }
        
        # Remove the zip file
        os.remove(zip_file_path)

        # Remove the extracted directory and its contents
        shutil.rmtree(data_dir)
        
        return data_splits

    def fetch_german_credit_data(data_dir="./GERMAN"):
        print("---------------------GERMAN--------------------------------------")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # URL of the dataset zip file
        url = "http://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
        zip_file_path = os.path.join(data_dir, "german_credit_data.zip")

        # Download the zip file
        urllib.request.urlretrieve(url, zip_file_path)

        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
            
        # Load the data from CSV
        data_path = os.path.join(data_dir, "german.data")

        columns = [
            "checking_account_status", "duration_months", "credit_history", "purpose",
            "credit_amount", "savings_account_bonds", "employment", "installment_rate",
            "personal_status_sex", "other_debtors_guarantors", "present_residence",
            "property", "age", "other_installment_plans", "housing", "existing_credits",
            "job", "num_dependents", "own_telephone", "foreign_worker", "class"
        ]
        data = pd.read_csv(data_path, sep=' ', header=None, names=columns)
        
        # Convert the class labels to binary format (1 = Good, 2 = Bad)
        data["class"] = data["class"].map({1: 1, 2: 0})
        
        # Handle null values (replace with appropriate values)
        data.fillna(method='ffill', inplace=True)  # Forward fill
        
        # Convert categorical variables to dummy variables
        categorical_columns = [
            "checking_account_status", "credit_history", "purpose", "savings_account_bonds",
            "employment", "personal_status_sex", "other_debtors_guarantors", "property",
            "other_installment_plans", "housing", "job", "own_telephone", "foreign_worker"
        ]
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
        
        # Split the data into features (X) and target (y)
        X = data.drop("class", axis=1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = data["class"]
        
        # Split the data into train, test, and validation sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_valid = (y_valid.values.reshape(-1) == 1).astype('int64')
        
        # Create a dictionary to store the data splits
        data_splits = {
            "X_train": X_train.astype('float32'), "y_train": y_train,
            "X_valid": X_valid.astype('float32'), "y_valid": y_valid,
            "X_test": X_test.astype('float32'), "y_test": y_test
        }
        
        # Remove the zip file
        os.remove(zip_file_path)

        # Remove the extracted directory and its contents
        shutil.rmtree(data_dir)
        
        return data_splits

    def fetch_spambase_dataset(data_dir="./SPAM"):
        print("---------------------SPAM--------------------------------------")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # URL of the dataset zip file
        url = "http://archive.ics.uci.edu/static/public/94/spambase.zip"
        zip_file_path = os.path.join(data_dir, "spambase.zip")

        # Download the zip file
        urllib.request.urlretrieve(url, zip_file_path)

        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
            
        # Load the data from CSV
        data_path = os.path.join(data_dir, "spambase.data")

        columns = [
            f"f{i}" for i in range(57)
        ] + ["spam"]
        data = pd.read_csv(data_path, header=None, names=columns)
        
        # Split the data into features (X) and target (y)
        X = data.drop("spam", axis=1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = data["spam"]
        
        # Split the data into train, test, and validation sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_valid = (y_valid.values.reshape(-1) == 1).astype('int64')
        
        # Create a dictionary to store the data splits
        data_splits = {
            "X_train": X_train.astype('float32'), "y_train": y_train,
            "X_valid": X_valid.astype('float32'), "y_valid": y_valid,
            "X_test": X_test.astype('float32'), "y_test": y_test
        }
        
        # Remove the zip file
        os.remove(zip_file_path)

        # Remove the extracted directory and its contents
        shutil.rmtree(data_dir)
        
        return data_splits

    def fetch_accelerometer_gyro_dataset(data_dir="./GYRO"):
        print("---------------------GYRO--------------------------------------")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # URL of the dataset zip file
        url = "https://archive.ics.uci.edu/static/public/755/accelerometer+gyro+mobile+phone+dataset.zip"
        zip_file_path = os.path.join(data_dir, "accelerometer_gyro_dataset.zip")

        # Download the zip file
        urllib.request.urlretrieve(url, zip_file_path)

        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
            
        # Load the data from CSV
        data_path = os.path.join(data_dir, "accelerometer_gyro_mobile_phone_dataset.csv")
        
        data = pd.read_csv(data_path)
        
        # Convert categorical column to numeric (e.g., label encoding)
        data["timestamp"] = data["timestamp"].astype("category").cat.codes
        
        # Split the data into features (X) and target (y)
        X = data.drop("Activity", axis=1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = data["Activity"]
        
        # Split the data into train, test, and validation sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_valid = (y_valid.values.reshape(-1) == 1).astype('int64')
        
        # Create a dictionary to store the data splits
        data_splits = {
            "X_train": X_train.astype('float32'), "y_train": y_train,
            "X_valid": X_valid.astype('float32'), "y_valid": y_valid,
            "X_test": X_test.astype('float32'), "y_test": y_test
        }
        
        # Remove the zip file
        os.remove(zip_file_path)

        # Remove the extracted directory and its contents
        shutil.rmtree(data_dir)
        
        return data_splits

    def fetch_swarm_behaviour(data_dir="./SWARM"):
        print("---------------------SWARM--------------------------------------")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # URL of the dataset zip file
        url = "https://archive.ics.uci.edu/static/public/524/swarm+behaviour.zip"
        zip_file_path = os.path.join(data_dir, "swarm_behaviour.zip")

        # Download the zip file
        urllib.request.urlretrieve(url, zip_file_path)

        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
            
        # Load the data from CSV
        data_path = os.path.join(data_dir, "Swarm Behavior Data/Grouped.csv")
        
        data = pd.read_csv(data_path)
        
        # Split the data into features (X) and target (y)
        X = data.drop("Class", axis=1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = data["Class"]
        
        # Split the data into train, test, and validation sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_valid = (y_valid.values.reshape(-1) == 1).astype('int64')
        
        # Create a dictionary to store the data splits
        data_splits = {
            "X_train": X_train.astype('float32'), "y_train": y_train,
            "X_valid": X_valid.astype('float32'), "y_valid": y_valid,
            "X_test": X_test.astype('float32'), "y_test": y_test
        }
        
        # Remove the zip file
        os.remove(zip_file_path)
        # Remove the extracted directory and its contents
        shutil.rmtree(data_dir) 
        return data_splits

    def fetch_openml_credit_data(data_dir="./OpenML_Credit"):
        print("---------------------OpenML_Credit DATASET--------------------------------------")
        # Create the data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_url = "https://api.openml.org/data/v1/download/22103185/credit.arff"
        # Download the ARFF file
        arff_file_path = os.path.join(data_dir, "credit.arff")
        urllib.request.urlretrieve(data_url, arff_file_path)

        # Load ARFF file into DataFrame
        data, meta = arff.loadarff(arff_file_path)
        df = pd.DataFrame(data)
        # Convert target variable to int
        last_column = df.columns[-1]

        df[last_column] = df[last_column].astype(int)
        
    #     print("df",df)

        # Split the data into train, validation, and test sets
        X = df.drop(last_column, axis=1)  # Assuming "SeriousDlqin2yrs" is the target variable
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df[last_column]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

    #     y_train = y_train.astype('int64')
    #     y_test = y_test.astype('int64')
    #     y_val = y_val.astype('int64')

        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_val = (y_val.values.reshape(-1) == 1).astype('int64')

        # Remove the ARFF file
        os.remove(arff_file_path)

        # Remove the data directory
        shutil.rmtree(data_dir)

        return dict(
            X_train=X_train.astype('float32'), y_train=y_train,
            X_valid=X_val.astype('float32'), y_valid=y_val,
            X_test=X_test.astype('float32'), y_test=y_test
        )


    def fetch_openml_electricity_data(data_dir="./OpenML_Electricity"):
        print("---------------------OpenML_Electricity DATASET--------------------------------------")
        # Create the data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_url = "https://api.openml.org/data/v1/download/22103245/electricity.arff"
        # Download the ARFF file
        arff_file_path = os.path.join(data_dir, "electricity.arff")
        urllib.request.urlretrieve(data_url, arff_file_path)

        # Load ARFF file into DataFrame
        data, meta = arff.loadarff(arff_file_path)
        df = pd.DataFrame(data)
        # Convert target variable to int
        last_column = df.columns[-1]

        df[last_column] = df[last_column].map({b'DOWN': 0, b'UP': 1})
        
    #     print("df",df)

        # Split the data into train, validation, and test sets
        X = df.drop(last_column, axis=1)  # Assuming "SeriousDlqin2yrs" is the target variable
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df[last_column]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

    #     y_train = y_train.astype('int64')
    #     y_test = y_test.astype('int64')
    #     y_val = y_val.astype('int64')

        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_val = (y_val.values.reshape(-1) == 1).astype('int64')

        # Remove the ARFF file
        os.remove(arff_file_path)

        # Remove the data directory
        shutil.rmtree(data_dir)

        return dict(
            X_train=X_train.astype('float32'), y_train=y_train,
            X_valid=X_val.astype('float32'), y_valid=y_val,
            X_test=X_test.astype('float32'), y_test=y_test
        )


    def fetch_openml_covertype_data(data_dir="./OpenML_Covertype"):
        print("---------------------OpenML_Covertype DATASET--------------------------------------")
        # Create the data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_url = "https://api.openml.org/data/v1/download/22103246/covertype.arff"
        # Download the ARFF file
        arff_file_path = os.path.join(data_dir, "covertype.arff")
        urllib.request.urlretrieve(data_url, arff_file_path)

        # Load ARFF file into DataFrame
        data, meta = arff.loadarff(arff_file_path)
        df = pd.DataFrame(data)
        # Convert target variable to int
        last_column = df.columns[-1]

        df[last_column] = df[last_column].astype(int)
        
    #     print("df",df)

        # Split the data into train, validation, and test sets
        X = df.drop(last_column, axis=1)  # Assuming "SeriousDlqin2yrs" is the target variable
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df[last_column]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

    #     y_train = y_train.astype('int64')
    #     y_test = y_test.astype('int64')
    #     y_val = y_val.astype('int64')

        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_val = (y_val.values.reshape(-1) == 1).astype('int64')

        # Remove the ARFF file
        os.remove(arff_file_path)

        # Remove the data directory
        shutil.rmtree(data_dir)

        return dict(
            X_train=X_train.astype('float32'), y_train=y_train,
            X_valid=X_val.astype('float32'), y_valid=y_val,
            X_test=X_test.astype('float32'), y_test=y_test
        )


    def fetch_openml_pol_data(data_dir="./OpenML_Pol"):
        print("---------------------OpenML_Pol DATASET--------------------------------------")
        # Create the data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_url = "https://api.openml.org/data/v1/download/22103247/pol.arff"
        # Download the ARFF file
        arff_file_path = os.path.join(data_dir, "pol.arff")
        urllib.request.urlretrieve(data_url, arff_file_path)

        # Load ARFF file into DataFrame
        data, meta = arff.loadarff(arff_file_path)
        df = pd.DataFrame(data)
        # Convert target variable to int
        last_column = df.columns[-1]

    #     print("df",df)
        
        df[last_column] = df[last_column].map({b'N':0,b'P':1})
        
        

        # Split the data into train, validation, and test sets
        X = df.drop(last_column, axis=1)  # Assuming "SeriousDlqin2yrs" is the target variable
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df[last_column]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

    #     y_train = y_train.astype('int64')
    #     y_test = y_test.astype('int64')
    #     y_val = y_val.astype('int64')

        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_val = (y_val.values.reshape(-1) == 1).astype('int64')

        # Remove the ARFF file
        os.remove(arff_file_path)

        # Remove the data directory
        shutil.rmtree(data_dir)

        return dict(
            X_train=X_train.astype('float32'), y_train=y_train,
            X_valid=X_val.astype('float32'), y_valid=y_val,
            X_test=X_test.astype('float32'), y_test=y_test
        )


    def fetch_openml_house_16H_data(data_dir="./OpenML_House_16H"):
        print("---------------------OpenML_House_16H DATASET--------------------------------------")
        # Create the data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_url = "https://api.openml.org/data/v1/download/22103248/house_16H.arff"
        # Download the ARFF file
        arff_file_path = os.path.join(data_dir, "house_16H.arff")
        urllib.request.urlretrieve(data_url, arff_file_path)

        # Load ARFF file into DataFrame
        data, meta = arff.loadarff(arff_file_path)
        df = pd.DataFrame(data)
        # Convert target variable to int
        last_column = df.columns[-1]

    #     print("df",df)
        df[last_column] = df[last_column].map({b'N':0,b'P':1})
        
        # Split the data into train, validation, and test sets
        X = df.drop(last_column, axis=1)  # Assuming "SeriousDlqin2yrs" is the target variable
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df[last_column]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

    #     y_train = y_train.astype('int64')
    #     y_test = y_test.astype('int64')
    #     y_val = y_val.astype('int64')

        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_val = (y_val.values.reshape(-1) == 1).astype('int64')

        # Remove the ARFF file
        os.remove(arff_file_path)

        # Remove the data directory
        shutil.rmtree(data_dir)

        return dict(
            X_train=X_train.astype('float32'), y_train=y_train,
            X_valid=X_val.astype('float32'), y_valid=y_val,
            X_test=X_test.astype('float32'), y_test=y_test
        )


    def fetch_openml_MiniBooNE_data(data_dir="./OpenML_MiniBooNE"):
        print("---------------------OpenML_MiniBooNE DATASET--------------------------------------")
        # Create the data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_url = "https://api.openml.org/data/v1/download/22103253/MiniBooNE.arff"
        # Download the ARFF file
        arff_file_path = os.path.join(data_dir, "MiniBooNE.arff")
        urllib.request.urlretrieve(data_url, arff_file_path)

        # Load ARFF file into DataFrame
        data, meta = arff.loadarff(arff_file_path)
        df = pd.DataFrame(data)
        # Convert target variable to int
        last_column = df.columns[-1]

    #     print("df",df)
        
        df[last_column] = df[last_column].map({b'False':0,b'True':1})

        # Split the data into train, validation, and test sets
        X = df.drop(last_column, axis=1)  # Assuming "SeriousDlqin2yrs" is the target variable
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df[last_column]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

    #     y_train = y_train.astype('int64')
    #     y_test = y_test.astype('int64')
    #     y_val = y_val.astype('int64')

        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_val = (y_val.values.reshape(-1) == 1).astype('int64')

        # Remove the ARFF file
        os.remove(arff_file_path)

        # Remove the data directory
        shutil.rmtree(data_dir)

        return dict(
            X_train=X_train.astype('float32'), y_train=y_train,
            X_valid=X_val.astype('float32'), y_valid=y_val,
            X_test=X_test.astype('float32'), y_test=y_test
        )


    def fetch_openml_eye_movements_data(data_dir="./OpenML_Eye_movements"):
        print("---------------------OpenML_Eye_movements DATASET--------------------------------------")
        # Create the data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_url = "https://api.openml.org/data/v1/download/22103255/eye_movements.arff"
        # Download the ARFF file
        arff_file_path = os.path.join(data_dir, "eye_movements.arff")
        urllib.request.urlretrieve(data_url, arff_file_path)

        # Load ARFF file into DataFrame
        data, meta = arff.loadarff(arff_file_path)
        df = pd.DataFrame(data)
        # Convert target variable to int
        last_column = df.columns[-1]

    #     print("df",df)
        df[last_column] = df[last_column].astype(int)

        # Split the data into train, validation, and test sets
        X = df.drop(last_column, axis=1)  # Assuming "SeriousDlqin2yrs" is the target variable
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df[last_column]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

    #     y_train = y_train.astype('int64')
    #     y_test = y_test.astype('int64')
    #     y_val = y_val.astype('int64')

        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_val = (y_val.values.reshape(-1) == 1).astype('int64')

        # Remove the ARFF file
        os.remove(arff_file_path)

        # Remove the data directory
        shutil.rmtree(data_dir)

        return dict(
            X_train=X_train.astype('float32'), y_train=y_train,
            X_valid=X_val.astype('float32'), y_valid=y_val,
            X_test=X_test.astype('float32'), y_test=y_test
        )


    def fetch_openml_Diabetes130US_data(data_dir="./OpenML_Diabetes130US"):
        print("---------------------OpenML_Diabetes130US DATASET--------------------------------------")
        # Create the data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_url = "https://api.openml.org/data/v1/download/22111908/Diabetes130US.arff"
        # Download the ARFF file
        arff_file_path = os.path.join(data_dir, "Diabetes130US.arff")
        urllib.request.urlretrieve(data_url, arff_file_path)

        # Load ARFF file into DataFrame
        data, meta = arff.loadarff(arff_file_path)
        df = pd.DataFrame(data)
        # Convert target variable to int
        last_column = df.columns[-1]
    #     print("df",df)
        df[last_column] = df[last_column].astype(int)
        

        # Split the data into train, validation, and test sets
        X = df.drop(last_column, axis=1)  # Assuming "SeriousDlqin2yrs" is the target variable
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df[last_column]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

    #     y_train = y_train.astype('int64')
    #     y_test = y_test.astype('int64')
    #     y_val = y_val.astype('int64')

        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_val = (y_val.values.reshape(-1) == 1).astype('int64')

        # Remove the ARFF file
        os.remove(arff_file_path)

        # Remove the data directory
        shutil.rmtree(data_dir)

        return dict(
            X_train=X_train.astype('float32'), y_train=y_train,
            X_valid=X_val.astype('float32'), y_valid=y_val,
            X_test=X_test.astype('float32'), y_test=y_test
        )


    def fetch_openml_jannis_data(data_dir="./OpenML_Jannis"):
        print("---------------------OpenML_Jannis DATASET--------------------------------------")
        # Create the data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_url = "https://api.openml.org/data/v1/download/22111907/jannis.arff"
        # Download the ARFF file
        arff_file_path = os.path.join(data_dir, "jannis.arff")
        urllib.request.urlretrieve(data_url, arff_file_path)

        # Load ARFF file into DataFrame
        data, meta = arff.loadarff(arff_file_path)
        df = pd.DataFrame(data)
        # Convert target variable to int
        last_column = df.columns[-1]
    #     print("df",df)

        df[last_column] = df[last_column].astype(int)


        # Split the data into train, validation, and test sets
        X = df.drop(last_column, axis=1)  # Assuming "SeriousDlqin2yrs" is the target variable
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df[last_column]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

    #     y_train = y_train.astype('int64')
    #     y_test = y_test.astype('int64')
    #     y_val = y_val.astype('int64')

        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_val = (y_val.values.reshape(-1) == 1).astype('int64')

        # Remove the ARFF file
        os.remove(arff_file_path)

        # Remove the data directory
        shutil.rmtree(data_dir)

        return dict(
            X_train=X_train.astype('float32'), y_train=y_train,
            X_valid=X_val.astype('float32'), y_valid=y_val,
            X_test=X_test.astype('float32'), y_test=y_test
        )


    def fetch_openml_Bioresponse_data(data_dir="./OpenML_Bioresponse"):
        print("---------------------OpenML_Bioresponse DATASET--------------------------------------")
        # Create the data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_url = "https://api.openml.org/data/v1/download/22111905/Bioresponse.arff"
        # Download the ARFF file
        arff_file_path = os.path.join(data_dir, "Bioresponse.arff")
        urllib.request.urlretrieve(data_url, arff_file_path)

        # Load ARFF file into DataFrame
        data, meta = arff.loadarff(arff_file_path)
        df = pd.DataFrame(data)
        # Convert target variable to int
        last_column = df.columns[-1]
    #     print("df",df)

        df[last_column] = df[last_column].astype(int)

        # Split the data into train, validation, and test sets
        X = df.drop(last_column, axis=1)  # Assuming "SeriousDlqin2yrs" is the target variable
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df[last_column]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

    #     y_train = y_train.astype('int64')
    #     y_test = y_test.astype('int64')
    #     y_val = y_val.astype('int64')

        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_val = (y_val.values.reshape(-1) == 1).astype('int64')

        # Remove the ARFF file
        os.remove(arff_file_path)

        # Remove the data directory
        shutil.rmtree(data_dir)

        return dict(
            X_train=X_train.astype('float32'), y_train=y_train,
            X_valid=X_val.astype('float32'), y_valid=y_val,
            X_test=X_test.astype('float32'), y_test=y_test
        )


    def fetch_openml_california_data(data_dir="./OpenML_California"):
        print("---------------------OpenML_California DATASET--------------------------------------")
        # Create the data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_url = "https://api.openml.org/data/v1/download/22111914/california.arff"
        # Download the ARFF file
        arff_file_path = os.path.join(data_dir, "california.arff")
        urllib.request.urlretrieve(data_url, arff_file_path)

        # Load ARFF file into DataFrame
        data, meta = arff.loadarff(arff_file_path)
        df = pd.DataFrame(data)
        # Convert target variable to int
        last_column = df.columns[-1]
    #     print("df",df)

        df[last_column] = df[last_column].astype(int)

        # Split the data into train, validation, and test sets
        X = df.drop(last_column, axis=1)  # Assuming "SeriousDlqin2yrs" is the target variable
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df[last_column]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

    #     y_train = y_train.astype('int64')
    #     y_test = y_test.astype('int64')
    #     y_val = y_val.astype('int64')

        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_val = (y_val.values.reshape(-1) == 1).astype('int64')

        # Remove the ARFF file
        os.remove(arff_file_path)

        # Remove the data directory
        shutil.rmtree(data_dir)

        return dict(
            X_train=X_train.astype('float32'), y_train=y_train,
            X_valid=X_val.astype('float32'), y_valid=y_val,
            X_test=X_test.astype('float32'), y_test=y_test
        )


    def fetch_openml_heloc_data(data_dir="./OpenML_Heloc"):
        print("---------------------OpenML_Heloc DATASET--------------------------------------")
        # Create the data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_url = "https://api.openml.org/data/v1/download/22111912/heloc.arff"
        # Download the ARFF file
        arff_file_path = os.path.join(data_dir, "heloc.arff")
        urllib.request.urlretrieve(data_url, arff_file_path)

        # Load ARFF file into DataFrame
        data, meta = arff.loadarff(arff_file_path)
        df = pd.DataFrame(data)
        # Convert target variable to int
        last_column = df.columns[-1]
    #     print("df",df)

        df[last_column] = df[last_column].astype(int)

        # Split the data into train, validation, and test sets
        X = df.drop(last_column, axis=1)  # Assuming "SeriousDlqin2yrs" is the target variable
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df[last_column]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

    #     y_train = y_train.astype('int64')
    #     y_test = y_test.astype('int64')
    #     y_val = y_val.astype('int64')

        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')
        y_val = (y_val.values.reshape(-1) == 1).astype('int64')

        # Remove the ARFF file
        os.remove(arff_file_path)

        # Remove the data directory
        shutil.rmtree(data_dir)

        return dict(
            X_train=X_train.astype('float32'), y_train=y_train,
            X_valid=X_val.astype('float32'), y_valid=y_val,
            X_test=X_test.astype('float32'), y_test=y_test
        )
    
    REAL_DATASETS = {
        ####### 10 latest UCI datasets ########
        'ADULT': fetch_ADULT,
        'bank_marketing': fetch_bank_marketing,
        'credit_card_defaults': fetch_credit_card_defaults,
        'gamma_telescope': fetch_gamma_telescope,
        'rice_dataset': fetch_rice_dataset,
        'german_credit_data': fetch_german_credit_data,
        'spambase_dataset': fetch_spambase_dataset,
        'accelerometer_gyro_dataset': fetch_accelerometer_gyro_dataset,
        'swarm_behaviour': fetch_swarm_behaviour,
        ######## OpenML Tabular Datasets ##########
        'OpenML_Credit': fetch_openml_credit_data,
        'OpenML_Electricity': fetch_openml_electricity_data,
        'OpenML_Covertype': fetch_openml_covertype_data,
        'OpenML_Pol': fetch_openml_pol_data,
        'OpenML_House_16H': fetch_openml_house_16H_data,
        'OpenML_MiniBooNE': fetch_openml_MiniBooNE_data,
        'OpenML_Eye_movements': fetch_openml_eye_movements_data,
        'OpenML_Diabetes130US': fetch_openml_Diabetes130US_data,
        'OpenML_Jannis': fetch_openml_jannis_data,
        'OpenML_Bioresponse': fetch_openml_Bioresponse_data,
        'OpenML_California': fetch_openml_california_data,
        'OpenML_Heloc': fetch_openml_heloc_data
    }

    class Dataset:
        def __init__(self, dataset, data_path='./DATA', normalize=False, normalize_target=False, quantile_transform=False, quantile_noise=1e-3, in_features=None, out_features=None, flatten=False, **kwargs):
            """
            Dataset is a dataclass that contains all training and evaluation data required for an experiment
            :param dataset: a pre-defined dataset name (see DATASETS) or a custom dataset
                Your dataset should be at (or will be downloaded into) {data_path}/{dataset}
            :param data_path: a shared data folder path where the dataset is stored (or will be downloaded into)
            :param normalize: standardize features by removing the mean and scaling to unit variance
            :param quantile_transform: whether tranform the feature distributions into normals, using a quantile transform
            :param quantile_noise: magnitude of the quantile noise
            :param in_features: which features to use as inputs
            :param out_features: which features to reconstruct as output
            :param flatten: whether flattening instances to vectors
            :param kwargs: depending on the dataset, you may select train size, test size or other params
            """

            if dataset in REAL_DATASETS:
                data_dict = REAL_DATASETS[dataset](Path(data_path) / dataset, **kwargs)

                self.X_train = data_dict['X_train']
                self.y_train = data_dict['y_train']
                self.X_valid = data_dict['X_valid']
                self.y_valid = data_dict['y_valid']
                self.X_test = data_dict['X_test']
                self.y_test = data_dict['y_test']

                if flatten:
                    self.X_train, self.X_valid, self.X_test = self.X_train.reshape(len(self.X_train), -1), self.X_valid.reshape(len(self.X_valid), -1), self.X_test.reshape(len(self.X_test), -1)

                if normalize:

                    print("Normalize dataset")
                    axis = [0] + [i + 2 for i in range(self.X_train.ndim - 2)]
                    self.mean = np.mean(self.X_train, axis=tuple(axis), dtype=np.float32)
                    self.std = np.std(self.X_train, axis=tuple(axis), dtype=np.float32)

                    # if constants, set std to 1
                    self.std[self.std == 0.] = 1.

                    if dataset not in ['ALOI']:
                        self.X_train = (self.X_train - self.mean) / self.std
                        self.X_valid = (self.X_valid - self.mean) / self.std
                        self.X_test = (self.X_test - self.mean) / self.std

                if quantile_transform:
                    quantile_train = np.copy(self.X_train)
                    if quantile_noise:
                        stds = np.std(quantile_train, axis=0, keepdims=True)
                        noise_std = quantile_noise / np.maximum(stds, quantile_noise)
                        quantile_train += noise_std * np.random.randn(*quantile_train.shape)

                    qt = QuantileTransformer(output_distribution='normal').fit(quantile_train)
                    self.X_train = qt.transform(self.X_train)
                    self.X_valid = qt.transform(self.X_valid)
                    self.X_test = qt.transform(self.X_test)

                if normalize_target:

                    print("Normalize target value")
                    self.mean_y = np.mean(self.y_train, axis=0, dtype=np.float32)
                    self.std_y = np.std(self.y_train, axis=0, dtype=np.float32)

                    # if constants, set std to 1
                    if self.std_y == 0.:
                        self.std_y = 1.

                    self.y_train = (self.y_train - self.mean_y) / self.std_y
                    self.y_valid = (self.y_valid - self.mean_y) / self.std_y
                    self.y_test = (self.y_test - self.mean_y) / self.std_y

                if in_features is not None:
                    self.X_train_in, self.X_valid_in, self.X_test_in = self.X_train[:, in_features], self.X_valid[:, in_features], self.X_test[:, in_features]

                if out_features is not None:
                    self.X_train_out, self.X_valid_out, self.X_test_out = self.X_train[:, out_features], self.X_valid[:, out_features], self.X_test[:, out_features]

            elif dataset in TOY_DATASETS:
                data_dict = toy_dataset(distr=dataset, **kwargs)

                self.X = data_dict['X']
                self.Y = data_dict['Y']
                if 'labels' in data_dict:
                    self.labels = data_dict['labels']

            self.data_path = data_path
            self.dataset = dataset

    
        
    # DATA_NAME_UCI=["ADULT","bank_marketing","credit_card_defaults","gamma_telescope","rice_dataset","german_credit_data","spambase_dataset","accelerometer_gyro_dataset","swarm_behaviour"]#,"HIGGS"]
    # DATA_NAME_UCI=["OpenML_Credit","OpenML_Electricity","OpenML_Pol","OpenML_House_16H","OpenML_MiniBooNE","OpenML_Eye_movements","OpenML_Diabetes130US","OpenML_Jannis","OpenML_Bioresponse","OpenML_California","OpenML_Heloc"]#","OpenML_Covertype""]#,"bank_marketing","credit_card_defaults","gamma_telescope","rice_dataset","german_credit_data","spambase_dataset","accelerometer_gyro_dataset","swarm_behaviour"]#,"HIGGS"]

    DATA_NAME_UCI=["OpenML_Covertype"]
    # DATA_NAME_UCI=["ADULT"]
    for data_name in DATA_NAME_UCI:
        data = Dataset(data_name)
        print(data)
        eval_dataset(data)
