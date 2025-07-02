import os
from glob import glob
import warnings
import imageio.v2 as imageio

warnings.filterwarnings('ignore')

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from PIL import ImageFile, Image
import networkx as nx

Image.MAX_IMAGE_PIXELS = None
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import scprep as scp
import pyvips as pv
import cv2
import re
import scanpy as sc
import scipy.sparse as sp
import csv
import h5py
from utils import smooth_exp
import anndata as ad

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import hypernetx as hnx
from skimage.feature import graycomatrix, graycoprops

import torch.nn.functional as F
import json
from torch_geometric.nn import GCNConv

class BaselineDataset(torch.utils.data.Dataset):
    """Some Information about baselines"""

    def __init__(self):
        super(BaselineDataset, self).__init__()

        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.features_train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            torchvision.transforms.RandomApply([transforms.RandomRotation((0, 180))]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        self.features_test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

    def get_img(self, name: str):
        """Load whole slide image of a sample.

        Args:
            name (str): name of a sample

        Returns:
            PIL.Image: return whole slide image.
        """
        if self.data == 'STimage-1K4M':
            img_dir = self.data_dir + '/image'
        elif self.data == 'HEST-1k':
            img_dir = self.data_dir + '/wsis'
        else:
            img_dir = self.data_dir + '/ST-imgs'
        if self.data == 'her2st':
            pre = os.path.join(img_dir, name[0], name)
            fig_name = os.listdir(pre)
            jpg_files = [file for file in fig_name if file.endswith(".jpg")]
            path = pre + '/' + jpg_files[0]
            print(path)
        elif self.data == 'stnet' or '10x_breast' in self.data:
            print(name)
            path = glob(img_dir + '/*' + name + '.tif')[0]
        elif self.data == 'STimage-1K4M':
            print(name)
            path = glob(img_dir + '/*' + name + '*' + '.png')[0]
        elif self.data == 'HEST-1k':
            print("name", name)
            path = img_dir + '/' + name + '.tif'
            print(path)
        elif 'DRP' in self.data:
            path = glob(img_dir + '/*' + name + '.svs')[0]
        else:
            print(name)
            path = glob(img_dir + '/*' + name + '*' + '.jpg')[0]

        if self.use_pyvips:
            im = pv.Image.new_from_file(path, level=0)
        elif self.data == 'HEST-1k':
            im = Image.open(path)
        else:
            im = Image.open(path)
        return im

    def get_cnt(self, name: str):
        """Load gene expression data of a sample.

        Args:
            name (str): name of a sample

        Returns:
            pandas.DataFrame: return gene expression.
        """
        if self.data == 'STimage-1K4M':
            if self.datatype == 'Mouse':
                path = self.data_dir + '/gene_exp/' + name + '_count.csv'
                df = pd.read_csv(path)
                df['id'] = df.iloc[:, 0].apply(lambda x: re.search(r'_(\d+\.\d+x\d+\.\d+)$', x).group(1))
                # 将 id 列中的内容按 'x' 分割，四舍五入为整数，并重新组合为 '19x11' 格式
                df['id'] = df['id'].apply(
                    lambda x: f"{round(float(x.split('x')[0]))}x{round(float(x.split('x')[1]))}")
                # 设置 id 列为索引
                df = df.set_index('id')
                if 'Unnamed: 0' in df.columns:
                    df = df.drop(columns=['Unnamed: 0'])
            else:
                path = self.data_dir + '/gene_exp/' + name + '_count.csv'
                df = pd.read_csv(path)
                df['id'] = df.iloc[:, 0].apply(lambda x: re.split('x', x)[-2:])
                df['id'] = df['id'].apply(lambda y: 'x'.join(y))
                if 'Unnamed: 0' in df.columns:
                    df = df.drop(columns=['Unnamed: 0'])

        elif self.data == 'HEST-1k':
            path = self.data_dir + '/st/' + name + '.h5ad'
            h5ad = ad.read_h5ad(path)
            obs_names = h5ad.obs_names
            var_names = h5ad.var_names
            if sp.isspmatrix(h5ad.X):
                h5ad.X = h5ad.X.toarray()

            # 检查h5ad.X是否为二维数组
            if h5ad.X.ndim != 2:
                print("h5ad.X is not a 2D array")

            X_df = pd.DataFrame(h5ad.X, index=obs_names, columns=var_names)
            return X_df
        else:
            # 保留原有数据集的处理逻辑
            path = self.data_dir + '/ST-cnts/' + name + '_sub.parquet'
            df = pd.read_parquet(path)

        return df

    def get_pos(self, name: str):
        """Load position information of a sample.
        The 'id' column is for matching against the gene expression table.

        Args:
            name (str): name of a sample

        Returns:
            pandas.DataFrame: return DataFrame with position information.
        """
        if self.data == 'STimage-1K4M':
            path = self.data_dir + '/coord/' + name + '_coord.csv'
            df = pd.read_csv(path)
            if self.datatype == 'Mouse':
                df.iloc[:, 0] = df.iloc[:, 0].astype(str)
                df['id'] = df.iloc[:, 0].apply(lambda x: re.search(r'_(\d+\.\d+x\d+\.\d+)$', x).group(1))
                df['id'] = df['id'].apply(
                    lambda x: f"{round(float(x.split('x')[0]))}x{round(float(x.split('x')[1]))}")
                df = df.set_index('id')
                df['x'] = df.iloc[:, 0].apply(lambda x: round(float(re.search(r'(\d+\.\d+)', x).group(1))))
                df['y'] = df.iloc[:, 0].apply(lambda x: round(float(re.search(r'x(\d+\.\d+)', x).group(1))))
            else:
                df['id'] = df.iloc[:, 0].apply(lambda x: re.split('x', x)[-2:])
                df['id'] = df['id'].apply(lambda y: 'x'.join(y))
                df['x'] = df['id'].apply(lambda x: int(re.split('x', x)[0]))
                df['y'] = df['id'].apply(lambda x: int(re.split('x', x)[1]))
        elif self.data == 'HEST-1k':
            # 打开h5文件
            path = self.data_dir + '/st/' + name + '.h5ad'
            adata = sc.read(path)
            df = pd.DataFrame({
                'barcode': adata.obs_names,
                'x': adata.obs['array_col'],
                'y': adata.obs['array_row'],
                'pixel_x': adata.obs['pxl_row_in_fullres'],
                'pixel_y': adata.obs['pxl_col_in_fullres']

            })
            df.set_index('barcode', inplace=True)
        else:
            path = self.data_dir + '/ST-spotfiles/' + name + '_selection.tsv'
            df = pd.read_csv(path, sep='\t')
            x = df['x'].values
            y = df['y'].values
            x = np.around(x).astype(int)
            y = np.around(y).astype(int)
            id = []
            for i in range(len(x)):
                id.append(str(x[i]) + 'x' + str(y[i]))
            df['id'] = id

        return df

    def get_meta(self, name: str):
        """Load both gene expression and postion data and merge them.

        Args:
            name (str): name of a sample

        Returns:
            pandas.DataFrame: return merged table (gene exp + position)
        """

        pos = self.get_pos(name)

        if self.data == 'HEST-1k':
            cnt = self.get_cnt(name)
            meta = cnt.join(pos, how='inner')
            meta.reset_index(inplace=True)
            meta.rename(columns={'index': 'barcode'}, inplace=True)
            meta['index'] = meta.apply(lambda row: f"{int(row['x'])}x{int(row['y'])}", axis=1)
            meta.set_index('index', inplace=True)
        elif 'DRP' not in self.data:
            cnt = self.get_cnt(name)
            if self.data == 'STimage-1K4M':
                if self.datatype == 'Mouse':
                    pass
                else:
                    cnt = cnt.set_index('id')
            if self.data == 'STimage-1K4M' and self.datatype == 'Mouse':
                pass
            else:
                pos = pos.set_index('id')
                pos.index = pos.index.astype(str)
            meta = cnt.join(pos, how='inner')
        else:
            meta = pos

        if self.mode == "external_test":
            meta = meta.sort_values(['x', 'y'])
        else:
            meta = meta.sort_values(['y', 'x'])

        return meta


class STDataset(BaselineDataset):
    """Dataset to load ST data for TRIPLEX
    """

    def __init__(self,
                 mode: str,
                 fold: int = 0,
                 extract_mode: str = None,
                 test_data=None,
                 **kwargs):
        """
        Args:
            mode (str): 'train', 'test', 'external_test', 'extraction', 'inference'.
            fold (int): Number of fold for cross validation.
            test_data (str, optional): Test data name. Defaults to None.
        """
        super().__init__()
    

        # Set primary attribute
        self.gt_dir = kwargs['t_global_dir']
        self.num_neighbors = kwargs['num_neighbors']
        self.neighbor_dir = f"{kwargs['neighbor_dir']}_{self.num_neighbors}_224"

        self.use_pyvips = kwargs['use_pyvips']

        self.r = kwargs['radius'] // 2
        self.extract_mode = False

        self.mode = mode

        if mode in ["external_test", "inference"]:
            self.data = test_data
            self.data_dir = f"{kwargs['data_dir']}/test/{self.data}"
        elif mode == "extraction":
            self.extract_mode = extract_mode
            self.data = test_data
            self.data_dir = f"{kwargs['data_dir']}/{self.data}"
            self.datatype = f"{kwargs['datatype']}"

        else:
            self.data = kwargs['type']
            if self.data == 'STimage-1K4M':
                self.data_dir = f"{kwargs['data_dir']}/{self.data}/{kwargs['subtype']}"
                self.datatype = f"{kwargs['datatype']}"
            elif self.data == 'HEST-1k':
                self.data_dir = f"{kwargs['data_dir']}/{self.data}"
                self.datatype = f"{kwargs['datatype']}"
            else:
                self.data_dir = f"{kwargs['data_dir']}/{self.data}"

        if self.data == 'STimage-1K4M':
            names = [i for i in os.listdir(self.data_dir + '/coord') if f"{self.datatype}" in i]
            print("names", names)
            names.sort()
            names = [i.split('_coord.csv')[0] for i in names]
        elif self.data == 'HEST-1k':
            if self.datatype == 'ZEN':
                names = [i for i in os.listdir(self.data_dir + '/st') if "ZEN4" in i]
            else:
                names = [i for i in os.listdir(self.data_dir + '/st') if f"{self.datatype}" in i]
            print("names", names)
            names.sort()
            names = [i.split('.h5ad')[0] for i in names]
        else:
            names = os.listdir(self.data_dir + '/ST-spotfiles')
            names.sort()
            names = [i.split('_selection.tsv')[0] for i in names]

        if mode in ["external_test", "inference"]:
            self.names = names

        else:
            if self.data == 'stnet':
                if self.data == 'stnet':
                    kf = KFold(8, shuffle=True, random_state=2021)
                    patients = ['BC23209', 'BC23270', 'BC23803', 'BC24105', 'BC24220', 'BC23268', 'BC23269', 'BC23272',
                                'BC23277', 'BC23287', 'BC23288', 'BC23377', 'BC23450', 'BC23506', 'BC23508', 'BC23567',
                                'BC23810', 'BC23895', 'BC23901', 'BC23903', 'BC23944', 'BC24044', 'BC24223']
                    patients = np.array(patients)

                _, ind_val = [i for i in kf.split(patients)][fold]
                paients_val = patients[ind_val]

                te_names = []
                for pp in paients_val:
                    te_names += [i for i in names if pp in i]

            elif self.data == 'her2st':
                patients = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                te_names = [i for i in names if patients[fold] in i]
            elif self.data == 'skin':
                patients = ['P2', 'P5', 'P9', 'P10']
                te_names = [i for i in names if patients[fold] in i]
            elif self.data == 'STimage-1K4M':
                if self.datatype == 'PCW':
                    patients = ['4.5-5PCW', '6.5PCW', '9PCW']
                    te_names = [i for i in names if patients[fold] in i]
                elif self.datatype == 'Mouse':
                    num_patients = 4
                    kf = KFold(n_splits=num_patients, shuffle=True, random_state=42)
                    names_array = np.array(names)
                    _, ind_val = [i for i in kf.split(names_array)][fold]
                    te_names = names_array[ind_val]
            elif self.data == 'HEST-1k':
                if self.datatype == 'ZEN':
                    num_patients = 3
                    kf = KFold(n_splits=num_patients, shuffle=True, random_state=42)
                    names_array = np.array(names)
                    _, ind_val = [i for i in kf.split(names_array)][fold]
                    te_names = names_array[ind_val]
                else:
                    num_patients = 8
                    kf = KFold(n_splits=num_patients, shuffle=True, random_state=42)
                    names_array = np.array(names)
                    _, ind_val = [i for i in kf.split(names_array)][fold]
                    te_names = names_array[ind_val]

            tr_names = list(set(names) - set(te_names))

            if self.mode == 'train':
                self.names = tr_names
            else:
                self.names = te_names

        if self.use_pyvips:
            with open(f"{self.data_dir}/slide_shape.pickle", "rb") as f:
                self.img_shape_dict = pickle.load(f)
        else:
            self.img_dict = {i: np.array(self.get_img(i)) for i in self.names}
        self.meta_dict = {i: self.get_meta(i) for i in self.names}

        if mode not in ["extraction", "inference"]:
            if self.data in ('STimage-1K4M', 'HEST-1k'):
                gene_list = list(np.load(self.data_dir + f'/genes_{self.datatype}.npy', allow_pickle=True))
                self.exp_dict = {}

                for i, m in self.meta_dict.items():
                    filtered_m = m[gene_list]
                    normalized_m = scp.normalize.library_size_normalize(filtered_m)
                    logged_m = scp.transform.log(normalized_m)
                    self.exp_dict[i] = logged_m
            else:
                gene_list = list(np.load(self.data_dir + f'/genes_{self.data}.npy', allow_pickle=True))
                self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[gene_list])) for i, m in
                                 self.meta_dict.items()}

            # Smoothing data
            self.exp_dict = {i: smooth_exp(m).values for i, m in self.exp_dict.items()}

        if mode == "external_test":
            self.center_dict = {i: np.floor(m[['pixel_y', 'pixel_x']].values).astype(int) for i, m in
                                self.meta_dict.items()}
            self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}
        else:
            if self.data == 'STimage-1K4M':
                self.center_dict = {i: np.floor(m[['xaxis', 'yaxis']].values).astype(int) for i, m in
                                    self.meta_dict.items()}
            else:
                self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                                    self.meta_dict.items()}
            self.loc_dict = {i: m[['y', 'x']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))


    def relabel_nodes(self, G):
        mapping = {node: (new_idx,) + node[1:] for new_idx, node in enumerate(G.nodes())}
        relabelled_G = nx.relabel_nodes(G, mapping)
        return relabelled_G

    def create_hypergraph(self, name, target_x, target_y):
        centers_array = np.array(self.center_dict[name])

        if centers_array.ndim != 2 or centers_array.shape[1] != 2:
            raise ValueError("the shape of centers must be (n, 2)")

        distances = np.linalg.norm(centers_array - [target_x, target_y], axis=1)
        nearest_indices = np.argsort(distances)[:9]
        nodes = []
        node_features_list = []
        exp_features_list = []

        for idx in nearest_indices:
            center_x, center_y = centers_array[idx]
            features, exp_features = self.get_patch_features((name, idx))
            node_features_list.append(features)
            exp_features_list.append(exp_features)
            node_id = f"{name}_{idx}"
            nodes.append(node_id)

        x = torch.stack(node_features_list, dim=0).squeeze()
        exp_x = torch.stack(exp_features_list, dim=0).squeeze()

        return x, exp_x

    def is_neighbor(self, x1, y1, x2, y2, threshold=500):
        if ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 == 0:
            return 0
        elif ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 < threshold:
            return 1
        else:
            return 2

    def get_patch_features(self, node):
        name, idx = node
        im = self.img_dict[name]
        center = self.center_dict[name][idx]
        exp_level = self.exp_dict[name][idx]
        x, y = center
        if self.use_pyvips:
            patches = im.extract_area(x, y, self.r, self.r).numpy()[:, :, :3]
        else:
            patches = im[y - self.r:y + self.r, x - self.r:x + self.r, :]

        if self.mode == "external_test" or "test":
            patches = self.test_transforms(patches)
        else:
            patches = self.train_transforms(patches)

        if patches.dim() != 4:
            patches = patches.unsqueeze(0)
        exp_level = torch.tensor(exp_level)
        return patches, exp_level


    def __getitem__(self, index):
        """Return one piece of data for training, and all data within a patient for testing.

        Returns:
            tuple:
                patches (torch.Tensor): Target spot images
                edge_patches (torch.Tensor): Target spot edge images
                seg_patches (torch.Tensor): Target spot seg images
                neighbor_edge_patches (torch.Tensor): Neighbor edge images
                neighbor_seg_patches (torch.Tensor): Neighbor seg images
                exps (torch.Tensor): Gene expression of the target spot.
                pid (torch.LongTensor): patient index
                sid (torch.LongTensor): spot index
                wsi (torch.Tensor): Features extracted from all spots for the patient
                position (torch.LongTensor): Relative position of spots
                neighbors (torch.Tensor): Features extracted from neighbor regions of the target spot.
                maks_tb (torch.Tensor): Masking table for neighbor features
        """
        if self.mode in ['train', 'test']:
            i = 0
            while index >= self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i - 1]
            name = self.id2name[i]

            if self.data in ['STimage-1K4M', 'HEST-1k']:
                subdirs = [d for d in os.listdir(self.data_dir + '/' + self.datatype) if
                           os.path.isdir(os.path.join(self.data_dir + '/' + self.datatype, d))]
                name_subdirs = [d for d in subdirs if name in d]
                self.per_patch_csv_path = os.path.join(self.data_dir, self.datatype, name_subdirs[0], 'per_patch.csv')
            else:
                subdirs = [d for d in os.listdir(self.data_dir) if
                           os.path.isdir(os.path.join(self.data_dir, d))]
                name_subdirs = [d for d in subdirs if name in d]
                self.per_patch_csv_path = os.path.join(self.data_dir, name_subdirs[0], 'per_patch.csv')

            self.per_patch_csv = pd.read_csv(self.per_patch_csv_path)
            if self.mode == "external_test":
                self.per_patch_csv = self.per_patch_csv.sort_values(['x', 'y'])
            else:
                self.per_patch_csv = self.per_patch_csv.sort_values(['y', 'x'])

            path = self.per_patch_csv.iloc[idx]['path']
            path = os.path.join(self.data_dir, path)
            im_t = Image.open(path)
            im = np.array(im_t)
            im_gray = im_t.convert('L')
            im_gray_array = np.array(im_gray)
            im_t.close()

            center = self.center_dict[name][idx]
            x, y = center

            self.hypergraph_x, self.hypergraph_x_exp = self.create_hypergraph(name, x, y)
            hypergraph_x = self.hypergraph_x.detach()
            hypergraph_x_exp = self.hypergraph_x_exp.detach()


            patches = []
            if self.mode == 'train':
                patch = self.train_transforms(im)
            else:
                patch = self.test_transforms(im)
            patches.append(patch)
            patches = torch.stack(patches, dim=0)

            if self.mode in ["train", "test"]:
                exps = self.exp_dict[name][idx]
                exps = torch.Tensor(exps)
                pos = self.loc_dict[name][idx]
                position = torch.Tensor(pos)
            else:
                exps = self.exp_dict[name]
                exps = torch.Tensor(exps)
                pos = self.loc_dict[name]
                position = torch.Tensor(pos)
            sid = torch.LongTensor([idx])

        wsi = torch.load(self.data_dir + f"/{self.gt_dir}/{name}.pt")

        pid = torch.LongTensor([i])

        if self.mode not in ["external_test", "inference"]:
            name += f"+{self.data}"

        return patches, exps, pid, sid, wsi, position, name, hypergraph_x, hypergraph_x_exp

    def __len__(self):
        if self.mode in ['train', 'test']:
            return self.cumlen[-1]
        else:
            if '10x_breast' in self.names[0]:
                return len(self.meta_dict[self.names[0]])
            return len(self.meta_dict)

    def make_masking_table(self, x: int, y: int, img_shape: tuple):
        """Generate masking table for neighbor encoder.

        Args:
            x (int): x coordinate of target spot
            y (int): y coordinate of target spot
            img_shape (tuple): Shape of whole slide image

        Raises:
            Exception: if self.num_neighbors is bigger than 5, raise error.

        Returns:
            torch.Tensor: masking table
        """

        # Make masking table for neighbor encoding module
        mask_tb = torch.ones(self.num_neighbors ** 2)

        def create_mask(ind, mask_tb, window):
            if y - self.r * window < 0:
                mask_tb[self.num_neighbors * ind:self.num_neighbors * ind + self.num_neighbors] = 0
            if y + self.r * window > img_shape[0]:
                mask_tb[(self.num_neighbors ** 2 - self.num_neighbors * (ind + 1)):(
                        self.num_neighbors ** 2 - self.num_neighbors * ind)] = 0
            if x - self.r * window < 0:
                mask = [i + ind for i in range(self.num_neighbors ** 2) if i % self.num_neighbors == 0]
                mask_tb[mask] = 0
            if x + self.r * window > img_shape[1]:
                mask = [i - ind for i in range(self.num_neighbors ** 2) if
                        i % self.num_neighbors == (self.num_neighbors - 1)]
                mask_tb[mask] = 0

            return mask_tb

        ind = 0
        window = self.num_neighbors
        while window >= 3:
            mask_tb = create_mask(ind, mask_tb, window)
            ind += 1
            window -= 2

        return mask_tb

    def extract_patch(self, im, x, y, r, padding_color=255):
        height, width, _ = im.shape

        x_lt = max(0, x - r)
        y_lt = max(0, y - r)
        x_rb = min(width, x + r)
        y_rb = min(height, y + r)

        patch = im[y_lt:y_rb, x_lt:x_rb, :]

        pad_left = max(0, r - (x - x_lt))
        pad_right = max(0, (x + r) - width)
        pad_top = max(0, r - (y - y_lt))
        pad_bottom = max(0, (y + r) - height)

        padded_patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant',
                              constant_values=padding_color)

        if padded_patch.shape[0] != 2 * r or padded_patch.shape[1] != 2 * r:
            final_patch = np.full((2 * r, 2 * r, 3), padding_color, dtype=im.dtype)
            start_x = r - (x - x_lt)
            start_y = r - (y - y_lt)
            end_x = start_x + patch.shape[1]
            end_y = start_y + patch.shape[0]
            final_patch[start_y:end_y, start_x:end_x, :] = patch
            padded_patch = final_patch

        self.ensure_patch_size(padded_patch, (224, 224))

        return padded_patch


    def ensure_patch_size(self, image, target_size=(224, 224)):
        """Ensure the image patch has the correct size."""
        if image.shape[0] == 0 or image.shape[1] == 0:
            # If the image patch has zero height or width, create a full padding image
            image = np.full((3, target_size[1], target_size[0]), 255, dtype='uint8')
        elif image.shape[1] != target_size[0] or image.shape[0] != target_size[1]:
            # Resize the image patch to the target size
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        return image
