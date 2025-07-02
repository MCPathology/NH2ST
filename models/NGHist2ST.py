import os
import inspect
import importlib
import wget
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import HypergraphConv
import torch.distributed as dist
from models.module import (
    HGNN,
    EXPNN,
    TWOFusionEncoder,
    Decoder
)


def load_model_weights(path: str):
    resnet = torchvision.models.__dict__["resnet18"](weights=None)
    ckpt_dir = "./weights"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = f"{ckpt_dir}/tenpercent_resnet18.ckpt"

    if not os.path.exists(ckpt_path):
        ckpt_url = "https://github.com/ozanciga/self-supervised-histopathology/releases/download/tenpercent/tenpercent_resnet18.ckpt"
        wget.download(ckpt_url, out=ckpt_dir)

    state = torch.load(path)
    state_dict = state["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "").replace("resnet.", "")] = state_dict.pop(key)

    model_dict = resnet.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    if state_dict == {}:
        print("No weight could be loaded..")
    model_dict.update(state_dict)
    resnet.load_state_dict(model_dict)
    resnet.fc = nn.Identity()
    return resnet


class NGHist2ST(pl.LightningModule):
    def __init__(self, num_genes=250, emb_dim=512, depth1=2, num_heads1=8, mlp_ratio1=2.0, dropout1=0.1, res_neighbor=(5, 5), learning_rate=0.0001, temperature1=0.05, temperature2=0.05, loss_ratio1=1.0, loss_ratio2=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.best_loss = np.inf
        self.best_cor = -1
        self.num_genes = num_genes
        self.alpha = 0.3
        self.num_n = res_neighbor[0]
        self.ratio1 = loss_ratio1
        self.ratio2 = loss_ratio2
        self.temperature1 = temperature1
        self.temperature2 = temperature2


        resnet18 = load_model_weights("weights/tenpercent_resnet18.ckpt")
        module = list(resnet18.children())[:-2]
        self.target_encoder = nn.Sequential(*module)
        self.fc_target = nn.Linear(emb_dim, num_genes)

        self.exp_encoder = nn.Sequential(
            nn.Linear(num_genes, emb_dim),
            nn.Linear(emb_dim, emb_dim)
        )

        self.neighbor_encoder = HGNN(25088, 1024, 512)
        self.neighbor_exp_encoder = EXPNN(512, 1024, 512)
        self.fc_neighbor = nn.Linear(emb_dim, num_genes)
        self.fc_nuclei = nn.Linear(emb_dim, num_genes)

        self.fc_global = nn.Linear(emb_dim, num_genes)
        self.cross_encoder = TWOFusionEncoder(emb_dim, depth1, num_heads1, int(emb_dim * mlp_ratio1), dropout1)
        self.fc = nn.Linear(25088, emb_dim)
        self.decoder = Decoder(input_dim=emb_dim, output_dim=num_genes)

    def contrastive_loss(self, features1, features2, temperature, negative_weight=0.1):
        if features1.dim() == 1:
            features1 = features1.unsqueeze(0)
        if features2.dim() == 1:
            features2 = features2.unsqueeze(0)
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        similarity_matrix = torch.mm(features1, features2.t()) / temperature
        batch_size = features1.size(0)
        mask = torch.eye(batch_size, device=features1.device)
        similarity_matrix = similarity_matrix * mask + similarity_matrix * (1 - mask) * negative_weight
        labels = torch.arange(batch_size, device=features1.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

    def build_graph(self, neighbor_nodes, neighbor_exp, k=3):
        num_nodes = neighbor_nodes.size(0)
        x = self.target_encoder(neighbor_nodes)
        x_exp = self.exp_encoder(neighbor_exp)
        x = x.view(-1, 25088)
        x_norm = F.normalize(x, p=2, dim=1)
        sim_matrix = torch.mm(x_norm, x_norm.T)
        mask = torch.eye(num_nodes, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))
        _, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
        edge_index = []
        for i in range(num_nodes):
            for j in topk_indices[i]:
                if j >= 0:
                    edge_index.append([i, j.item()])
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)
        return x, x_exp, edge_index

    def build_hypergraph(self, neighbor_nodes, neighbor_exp, k=3):
        num_nodes = neighbor_nodes.size(0)
        x = self.target_encoder(neighbor_nodes)
        x_exp = self.exp_encoder(neighbor_exp)
        x = x.view(-1, 25088)
        x_norm = F.normalize(x, p=2, dim=1)
        sim_matrix = torch.mm(x_norm, x_norm.T)
        mask = torch.eye(num_nodes, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1)
        _, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
        topk_indices = topk_indices.long()
        hyperedge_indices = []
        for i in range(num_nodes):
            hyperedge_indices.extend([(node_idx, i) for node_idx in topk_indices[i]])
            hyperedge_indices.append((i, i))
        if hyperedge_indices:
            rows, cols = zip(*hyperedge_indices)
            hyperedge_index = torch.tensor([rows, cols], dtype=torch.long)
        else:
            hyperedge_index = torch.tensor([[[], []]], dtype=torch.long)
        hyperedge_index = hyperedge_index.long()
        return x, x_exp, hyperedge_index

    def forward(self, x, exp, x_neighbor, x_neighbor_exp):
        x = x.squeeze()
        if x.dim() != 4:
            x = x.unsqueeze(0)
        x = self.target_encoder(x)
        _, dim, w, h = x.shape
        x = rearrange(x, "b d h w -> b (h w) d", d=dim, w=w, h=h)
        x = self.fc(x.reshape(x.shape[0], -1, 25088))

        x_neighbor = x_neighbor.to(torch.float32)
        x_neighbor_exp = x_neighbor_exp.to(torch.float32)

        if x_neighbor.dim() == 4:
            neighbor, neighbor_exp, hyperedge = self.build_hypergraph(x_neighbor, x_neighbor_exp)
        elif x_neighbor.dim() == 5:
            batch_size, num_patches, channels, height, width = x_neighbor.size()
            neighbor = []
            neighbor_exp = []
            hyperedge = []
            for i in range(batch_size):
                n, n_exp, h = self.build_hypergraph(x_neighbor[i].squeeze(0), x_neighbor_exp[i].squeeze(0))
                neighbor.append(n)
                neighbor_exp.append(n_exp)
                hyperedge.append(h)
            neighbor = torch.stack(neighbor).view(batch_size, -1, 25088).to(x_neighbor.device)
            neighbor_exp = torch.stack(neighbor_exp).view(batch_size, -1, 512).to(x_neighbor.device)
            hyperedge = torch.stack(hyperedge).to(x_neighbor.device)

        neighbor = neighbor.view(x.shape[0], -1, 25088).to(x.device)
        neighbor_exp = neighbor_exp.view(x.shape[0], -1, 512).to(x.device)
        if hyperedge.dim() == 2:
            hyperedge = hyperedge.unsqueeze(0)
        hyperedge = hyperedge.to(x.device)

        all_neighbors = []
        all_neighbor_exps = []
        for i in range(x.shape[0]):
            neighbor_i = neighbor[i]
            neighbor_exp_i = neighbor_exp[i]
            hyperedge_i = hyperedge[i]
            neighbors = self.neighbor_encoder(neighbor_i, hyperedge_i).view(1, -1).to(x.device)
            neighbor_exps = self.neighbor_exp_encoder(neighbor_exp_i, hyperedge_i).view(1, -1).to(x.device)
            all_neighbors.append(neighbors)
            all_neighbor_exps.append(neighbor_exps)
        neighbors = torch.stack(all_neighbors, dim=0)
        neighbor_exps = torch.stack(all_neighbor_exps, dim=0)

        patch_fusion = x.reshape(x.shape[0], -1).to(x.device)
        patch_exp = self.exp_encoder(exp).reshape(x.shape[0], -1).to(x.device)
        neighbors = neighbors.reshape(x.shape[0], -1).to(x.device)
        neighbor_exps = neighbor_exps.reshape(x.shape[0], -1).to(x.device)
        pred_exp = self.decoder(patch_fusion)
        decoded_exp = self.decoder(patch_fusion)

        patch_fusion = self.cross_encoder(patch_exp, patch_fusion)
        patch_exp = self.cross_encoder(patch_fusion, patch_exp)
        neighbors = self.cross_encoder(neighbor_exps, neighbors)
        neighbor_exps = self.cross_encoder(neighbors, neighbor_exps)

        return patch_fusion, patch_exp, neighbors, neighbor_exps, decoded_exp, pred_exp

    def training_step(self, batch, batch_idx):
        patch, exp, pid, sid, wsi, position, name, neighbor, neighbor_exp = batch
        outputs = self(patch, exp, neighbor, neighbor_exp)
        loss_patch = self.contrastive_loss(outputs[0].squeeze(), outputs[1].squeeze(), self.temperature1)
        loss_neighbor = self.contrastive_loss(outputs[2].squeeze(), outputs[3].squeeze(), self.temperature2)
        reconstruction_loss = F.mse_loss(outputs[5], exp)
        loss = self.ratio1 * loss_patch + self.ratio2 * loss_neighbor + reconstruction_loss
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, exp, pid, sid, wsi, position, name, neighbor, neighbor_exp = batch
        patch, exp, neighbor, neighbor_exp = patch.squeeze(), exp.squeeze(), neighbor.squeeze(), neighbor_exp.squeeze()
        outputs = self(patch, exp, neighbor, neighbor_exp)
        pred = outputs[5].cpu().detach().numpy().T
        exp = exp.cpu().detach().numpy().T
        self.get_meta(name)
        return {"pred": pred.reshape(1, -1), "exp": exp.reshape(1, -1)}

    def validation_epoch_end(self, outputs):
        all_preds = torch.tensor(np.concatenate([x["pred"] for x in outputs], axis=0))
        all_exps = torch.tensor(np.concatenate([x["exp"] for x in outputs], axis=0))
        all_preds = all_preds.squeeze().cpu().detach().numpy().T
        all_exps = all_exps.squeeze().cpu().detach().numpy().T

        mse_losses = []
        mae_losses = []
        feature_corrs = []

        for g in range(all_exps.shape[0]):
            mse = F.mse_loss(torch.tensor(all_preds[g]), torch.tensor(all_exps[g]))
            mae = F.l1_loss(torch.tensor(all_preds[g]), torch.tensor(all_exps[g]))
            mse_losses.append(mse.item())
            mae_losses.append(mae.item())
            corr = pearsonr(all_preds[g], all_exps[g])[0]
            feature_corrs.append(corr)
        feature_corrs = sorted(feature_corrs, reverse=True)[:50]
        avg_loss = torch.tensor(np.mean(mse_losses))
        avg_corr = torch.tensor(np.nanmean(feature_corrs))
        os.makedirs(f"results/{self.__class__.__name__}/{self.data}", exist_ok=True)
        if self.best_cor < avg_corr:
            torch.save(avg_corr.cpu(), f"results/{self.__class__.__name__}/{self.data}/R_{self.patient}")
            torch.save(avg_loss.cpu(), f"results/{self.__class__.__name__}/{self.data}/loss_{self.patient}")
            self.best_cor = avg_corr
            self.best_loss = avg_loss

        self.log("valid_loss", avg_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("R", avg_corr, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        patch, exp, pid, sid, wsi, position, name, neighbor, neighbor_exp = batch
        patch, exp, neighbor, neighbor_exp = patch.squeeze(), exp.squeeze(), neighbor.squeeze(), neighbor_exp.squeeze()
        outputs = self(patch, exp, neighbor, neighbor_exp)
        pred = outputs[5].cpu().detach().numpy().T
        exp = exp.cpu().detach().numpy().T
        self.get_meta(name)
        return {"pred": pred.reshape(1, -1), "exp": exp.reshape(1, -1)}

    def test_epoch_end(self, outputs):
        all_preds = torch.tensor(np.concatenate([x["pred"] for x in outputs], axis=0))
        all_exps = torch.tensor(np.concatenate([x["exp"] for x in outputs], axis=0))
        all_preds = all_preds.squeeze().cpu().detach().numpy()
        all_exps = all_exps.squeeze().cpu().detach().numpy()

        mse_losses = []
        mae_losses = []
        feature_corrs = []

        for g in range(all_exps.shape[0]):
            mse = F.mse_loss(torch.tensor(all_preds[g]), torch.tensor(all_exps[g]))
            mae = F.l1_loss(torch.tensor(all_preds[g]), torch.tensor(all_exps[g]))
            mse_losses.append(mse.item())
            mae_losses.append(mae.item())
            corr = pearsonr(all_preds[g], all_exps[g])[0]
            feature_corrs.append(corr)
        feature_corrs = sorted(feature_corrs, reverse=True)[:50]
        avg_mse = np.mean(mse_losses)
        avg_mae = np.mean(mae_losses)
        avg_feature_corr = np.nanmean(feature_corrs)

        print("avg_mse", avg_mse)
        print("avg_mae", avg_mae)
        print("avg_feature_corr", avg_feature_corr)

        os.makedirs(f"final/{self.__class__.__name__}_{self.num_n}/{self.data}/{self.patient}", exist_ok=True)
        torch.save(avg_mse, f"final/{self.__class__.__name__}_{self.num_n}/{self.data}/{self.patient}/MSE")
        torch.save(avg_mae, f"final/{self.__class__.__name__}_{self.num_n}/{self.data}/{self.patient}/MAE")
        torch.save(avg_feature_corr, f"final/{self.__class__.__name__}_{self.num_n}/{self.data}/{self.patient}/feature_corr")

        best_corr_index = np.argmax(feature_corrs)
        print(f"Best feature_corr index: {best_corr_index}")
        print(f"Best feature_corr value: {feature_corr[best_corr_index]}")

        genes_path = "/path/to/genes_XXX.npy"
        genes = np.load(genes_path, allow_pickle=True)
        best_gene_name = genes[best_corr_index]
        print(f"Best gene name: {best_gene_name}")

        best_gene_expression = all_exps[:, best_corr_index]
        best_gene_pred_expression = all_preds[:, best_corr_index]

        for name, exp, pred_exp, pos in zip(all_names, best_gene_expression, best_gene_pred_expression, all_positions):
            save_dir = f"final/{name}"
            os.makedirs(save_dir, exist_ok=True)
            np.savez(os.path.join(save_dir, f"{name}_{pos}_best_gene_{best_gene_name}.npz"),
                     gene_expression=exp,
                     predicted_expression=pred_exp,
                     position=pos)

        print(
            f"Saved best gene expression, predicted expression, and positions for each sample to their respective directories.")


    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        StepLR = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.9)
        optim_dict = {"optimizer": optim, "lr_scheduler": StepLR}
        return optim_dict

    def get_meta(self, name):
        if "10x_breast" in name[0]:
            self.patient = name[0]
            self.data = "test"
        else:
            name = name[0]
            self.data = name.split("+")[1]
            self.patient = name.split("+")[0]
            if self.data == "her2st":
                self.patient = self.patient[0]
            elif self.data == "stnet":
                self.data = "stnet"
                patient = self.patient.split("_")[0]
                if patient in ["BC23277", "BC23287", "BC23508"]:
                    self.patient = "BC1"
                elif patient in ["BC24105", "BC24220", "BC24223"]:
                    self.patient = "BC2"
                elif patient in ["BC23803", "BC23377", "BC23895"]:
                    self.patient = "BC3"
                elif patient in ["BC23272", "BC23288", "BC23903"]:
                    self.patient = "BC4"
                elif patient in ["BC23270", "BC23268", "BC23567"]:
                    self.patient = "BC5"
                elif patient in ["BC23269", "BC23810", "BC23901"]:
                    self.patient = "BC6"
                elif patient in ["BC23209", "BC23450", "BC23506"]:
                    self.patient = "BC7"
                elif patient in ["BC23944", "BC24044"]:
                    self.patient = "BC8"
            elif self.data == "skin":
                self.patient = self.patient.split("_")[0]

    def load_model(self):
        name = self.hparams.MODEL.name
        if "_" in name:
            camel_name = "".join([i.capitalize() for i in name.split("_")])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(f"models.{name}"), camel_name)
        except:
            raise ValueError("Invalid Module File Name or Invalid Class Name!")
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.MODEL.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.MODEL, arg)
        args1.update(other_args)
        return Model(**args1)


class CustomWriter(BasePredictionWriter):
    def __init__(self, pred_dir, write_interval, emb_dir=None, names=None):
        super().__init__(write_interval)
        self.pred_dir = pred_dir
        self.emb_dir = emb_dir
        self.names = names

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        rank = dist.get_rank()
        for i, batch in enumerate(batch_indices[rank]):
            pred_path = os.path.join(self.pred_dir, f"{self.names[i]}_rank{rank}.pt")
            emb_path = os.path.join(self.emb_dir, f"{self.names[i]}_rank{rank}.pt")
            torch.save(predictions[rank][i][0].detach(), pred_path)
            if self.emb_dir:
                torch.save(predictions[rank][i][1].detach(), emb_path)