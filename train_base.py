# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""

import argparse
import os
import time
import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from sklearn.metrics import log_loss, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.avazu.AvazuDataLoader import get_avazu_dataloader_811
from data.criteo.CriteoDataLoader import get_criteo_dataloader_811
from data.frappe.FrappeDataLoader import get_frappe_dataloader_811
from data.malware.MalwareDataLoader import get_malware_dataloader_811
from data.mltag.MLtagDataLoader import get_mltag_dataloader811
from models import *
from utils.earlystoping import EarlyStopping
from utils.utils import count_params

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("=======================", DEVICE)
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name())

DataLoaders = {
    "criteo": lambda datapath, batch_size: get_criteo_dataloader_811(train_path=datapath, batch_size=batch_size),
    "avazu": lambda datapath, batch_size: get_avazu_dataloader_811(train_path=datapath, batch_size=batch_size),
    "malware": lambda datapath, batch_size: get_malware_dataloader_811(train_path=datapath, batch_size=batch_size),
    "frappe": lambda datapath, batch_size: get_frappe_dataloader_811(path=datapath, batch_size=batch_size),
    "mltag": lambda datapath, batch_size: get_mltag_dataloader811(path=datapath, batch_size=batch_size)
}


def get_model(
        name,
        field_dims,
        embed_dim=16,
        att_size=16,
        mlp_layers=(400, 400, 400)):
    if name == "fm":
        return FactorizationMachineModel(field_dims, embed_dim)
    elif name == "deepfm":
        return DeepFM(field_dims, embed_dim, mlp_layers=mlp_layers)
    elif name == "dcnv2p":
        return DCNV2P(field_dims, embed_dim, cn_layers=3, mlp_layers=mlp_layers)
    elif name == "dcnv2s":
        return DCNV2S(field_dims, embed_dim, cn_layers=3, mlp_layers=mlp_layers)
    elif name == "gdcn_s":
        return GDCNS(field_dims, embed_dim, cn_layers=3, mlp_layers=mlp_layers)
    elif name == "gdcn_p":
        return GDCNP(field_dims, embed_dim, cn_layers=3, mlp_layers=mlp_layers)

    else:
        raise ValueError("No valid model name.")


def train(model,
          optimizer,
          data_loader,
          criterion,
          device="cuda:0",
          log_interval=50000, ):
    model.train()
    pred = list()
    target = list()
    total_loss = 0
    for i, (user_item, label) in enumerate(tqdm.tqdm(data_loader)):
        # print(user_item)
        label = label.float()
        user_item = user_item.long()
        user_item = user_item.cpu()
        label = label.cpu()
        # label = label.cuda()

        model.zero_grad()
        pred_y = torch.sigmoid(model(user_item).squeeze(1))
        loss = criterion(pred_y, label)
        loss.backward()
        optimizer.step()

        pred.extend(pred_y.tolist())
        target.extend(label.tolist())
        total_loss += loss.item()

    loss2 = total_loss / (i + 1)
    return loss2


def test_roc(model, data_loader):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(
                data_loader, smoothing=0, mininterval=1.0):
            fields = fields.long()
            target = target.float()
            fields, target = fields.cpu(), target.cpu()
            # fields, target = fields.cuda(), target.cuda()
            y = torch.sigmoid(model(fields).squeeze(1))

            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts), log_loss(targets, predicts)


def main(dataset_name,
         data_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         save_dir,
         repeat=1,
         emb_dim=20,
         hint=""):
    # Load data
    field_dims, trainLoader, validLoader, testLoader = DataLoaders[dataset_name](data_path, batch_size)
    time_fix = time.strftime("%m%d%H%M%S", time.localtime())
    for K in [emb_dim]:
        paths = os.path.join(save_dir, str(K), model_name)
        if not os.path.exists(paths):
            os.makedirs(paths)

        with open(
                paths + f"/{model_name}logs2_{K}_{batch_size}_{learning_rate}_{weight_decay}_{time_fix}.p",
                "a+") as fout:
            fout.write("hint:{}\n".format(hint))
            fout.write("feature_nums:{}\n".format(sum(field_dims)))
            fout.write("model_name:{}\t Batch_size:{}\tlearning_rate:{}\t"
                       "StartTime:{}\tweight_decay:{}\n"
                       .format(model_name, batch_size, learning_rate,
                               time.strftime("%d%H%M%S", time.localtime()), weight_decay))
            print("Start train -- K : {}".format(K))
            criterion = torch.nn.BCELoss()
            model = get_model(name=model_name,field_dims=field_dims,embed_dim=K).cpu()
            params = count_params(model)
            fout.write("count_params:{}\n".format(params))
            optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate,
                                         weight_decay=weight_decay, )

            early_stopping = EarlyStopping(patience=4, verbose=True)

            val_auc_best = 0
            auc_index_record = ""
            val_loss_best = 1000
            loss_index_record = ""

            scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=5)
            for epoch_i in range(epoch):
                print(__file__, model_name, batch_size, repeat, K, learning_rate, weight_decay,
                      epoch_i, "/", epoch)

                start = time.time()
                train_loss = train(model, optimizer, trainLoader, criterion)
                val_auc, val_loss = test_roc(model, validLoader)
                test_auc, test_loss = test_roc(model, testLoader)

                scheduler.step(val_auc)
                end = time.time()
                if val_auc > val_auc_best:
                    torch.save(model, paths + f"/{model_name}_best_auc_{K}_{time_fix}.pkl")

                if val_auc > val_auc_best:
                    val_auc_best = val_auc
                    auc_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i, test_auc, test_loss)

                if val_loss < val_loss_best:
                    val_loss_best = val_loss
                    loss_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i, test_auc, test_loss)

                print(
                    "Train  K:{}\tEpoch:{}\ttrain_loss:{:.6f}\tval_loss:{:.6f}\tval_auc:{:.6f}\ttime:{:.6f}\ttest_auc and test_loss: {:.6f}\t{:.6f}\n"
                    .format(K, epoch_i, train_loss, val_loss, val_auc, end - start, test_auc, test_loss))

                fout.write(
                    "Train  K:{}\tEpoch:{}\ttrain_loss:{:.6f}\tval_loss:{:.6f}\tval_auc:{:.6f}\ttime:{:.6f}\ttest_auc and test_loss: {:.6f}\t{:.6f}\n"
                    .format(K, epoch_i, train_loss, val_loss, val_auc, end - start, test_auc, test_loss))

                early_stopping(val_auc)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            print("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                  .format(K, val_auc, val_auc_best, val_loss, val_loss_best, test_loss, test_auc))

            fout.write("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                       .format(K, val_auc, val_auc_best, val_loss, val_loss_best, test_loss, test_auc))

            fout.write("auc_best:\t{}\nloss_best:\t{}".format(auc_index_record, loss_index_record))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='frappe')
    parser.add_argument('--save_dir', default='../chkpts/gdcn/')
    parser.add_argument('--data_path', default="data/", help="")
    parser.add_argument('--model_name', default='fm')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--emb_dim', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0', help="cuda:0")
    parser.add_argument('--choice', default=0, type=int)
    parser.add_argument('--repeats', default=10, type=int)
    parser.add_argument('--hint', default="v1.0")
    args = parser.parse_args()

    model_names = []
    if args.choice == 0:
        model_names = ["fm", "deepfm", "gdcn_s", "gdcn_p"]
    elif args.choice == 1:
        model_names = ["fm"]

    print(model_names)

    for i in range(args.repeats):
        for name in model_names:
            main(dataset_name=args.dataset_name,
                 data_path=args.data_path,
                 model_name=name,
                 epoch=args.epoch,
                 learning_rate=args.learning_rate,
                 batch_size=args.batch_size,
                 weight_decay=args.weight_decay,
                 save_dir=args.save_dir,
                 repeat=i + 1,
                 emb_dim=args.emb_dim,
                 hint=args.hint,
                 )
