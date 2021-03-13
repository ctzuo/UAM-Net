import torch
from torch.autograd import Variable
import torch.nn.functional as F
from data.build import make_data_loader
import torch.optim as optim
import argparse
from config import cfg
import os
from evaluation import label_accuracy_score
from model import UAM_NET


# ------- 1. define loss function --------

def cross_entropy(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target.long(), weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss


def trans_f_to_label(f7, f6, f5, f4, f3, f2):
    label_preds_f7 = f7.max(dim=1)[1]
    label_preds_f6 = f6.max(dim=1)[1]
    label_preds_f5 = f5.max(dim=1)[1]
    label_preds_f4 = f4.max(dim=1)[1]
    label_preds_f3 = f3.max(dim=1)[1]
    label_preds_f2 = f2.max(dim=1)[1]

    return label_preds_f7, label_preds_f6, label_preds_f5, label_preds_f4, label_preds_f3, label_preds_f2


def tam(f7, f6, f5, f4, f3, f2, f1, labels_v):
    label_preds_f7, label_preds_f6, label_preds_f5, label_preds_f4, label_preds_f3, label_preds_f2 = trans_f_to_label(
        f7, f6, f5, f4, f3, f2)
    lossg_7 = cross_entropy(f7, labels_v)
    loss7_6 = cross_entropy(f6, label_preds_f7)
    loss6_5 = cross_entropy(f5, label_preds_f6)
    loss5_4 = cross_entropy(f4, label_preds_f5)
    loss4_3 = cross_entropy(f3, label_preds_f4)
    loss3_2 = cross_entropy(f2, label_preds_f3)
    loss2_1 = cross_entropy(f1, label_preds_f2)
    lossg_6 = cross_entropy(f6, labels_v)
    lossg_5 = cross_entropy(f5, labels_v)
    lossg_4 = cross_entropy(f4, labels_v)
    lossg_3 = cross_entropy(f3, labels_v)
    lossg_2 = cross_entropy(f2, labels_v)
    lossg_1 = cross_entropy(f1, labels_v)

    loss = lossg_7 + loss7_6 + loss6_5 + loss5_4 + loss4_3 + loss3_2 + loss2_1 + 0.1 * (
                lossg_6 + lossg_5 + lossg_4 + lossg_3 + lossg_2 + lossg_1)

    return loss


# ------- 2. set the directory of training dataset --------

model_name = 'uam_net'

model_dir = "/home/uam/saved_models"  # path to save the model
epoch_num = 1  # 300
validation = True
parser = argparse.ArgumentParser(description="UAM_Net Training")
parser.add_argument(
    "--config_file", default="", help="path to config file", type=str
)
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

if args.config_file != "":
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

salobj_dataloader = make_data_loader(cfg, True)
salobjval_dataloader = make_data_loader(cfg, False)

# ------- 3. define model --------
# define the net
if (model_name == 'uam_net'):
    net = UAM_NET(3, 23)

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005)
# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
ite_num4val = 0
best_miou_train = 0
best_miou_test = 0
for epoch in range(0, epoch_num):
    net.train()
    avg_acc, avg_acc_cls, avg_mean_iu, avg_fwavacc = 0.0, 0.0, 0.0, 0.0
    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1
        inputs, labels = data
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
        optimizer.zero_grad()
        f7, f6, f5, f4, f3, f2, f1 = net(inputs_v)
        loss = tam(f7, f6, f5, f4, f3, f2, f1, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.item()
        predict = f7
        label = labels
        evals = label_accuracy_score(label, predict, 23)  # 18
        avg_acc += evals[0]
        avg_acc_cls += evals[1]
        avg_mean_iu += evals[2]
        avg_fwavacc += evals[3]
        del f7, f6, f5, f4, f3, f2, f1, loss
    avg_acc = avg_acc / (i + 1)
    avg_acc_cls = avg_acc_cls / (i + 1)
    avg_mean_iu = avg_mean_iu / (i + 1)
    avg_fwavacc = avg_fwavacc / (i + 1)
    print('Tra [epoch:%3d loss:%.3f avg_acc:%.3f avg_acc_cls:%.3f miou:%.3f fwavacc:%.3f]' % (
        epoch + 1, running_loss / (i + 1), avg_acc, avg_acc_cls, avg_mean_iu, avg_fwavacc))
    running_loss = 0.0
    if avg_mean_iu > best_miou_train:
        last_name = 'best_train_' + str(best_miou_train) + '.pth'
        best_miou_train = avg_mean_iu
        new_name = 'best_train_' + str(best_miou_train) + '.pth'
        torch.save(net.state_dict(), model_dir + '/' + new_name)
        if last_name != 'best_train_0.pth':
            os.remove(model_dir + '/' + last_name)

    if validation == True:
        net.eval()
        avg_acc, avg_acc_cls, avg_mean_iu, avg_fwavacc = 0.0, 0.0, 0.0, 0.0
        for i, data in enumerate(salobjval_dataloader):
            inputs, labels = data
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            optimizer.zero_grad()

            f7, f6, f5, f4, f3, f2, f1 = net(inputs_v)
            loss2, loss = tam(f7, f6, f5, f4, f3, f2, f1, labels_v)
            running_loss += loss.item()
            predict = f7
            label = labels
            evals = label_accuracy_score(label, predict, 23)  # 23   56
            avg_acc += evals[0]
            avg_acc_cls += evals[1]
            avg_mean_iu += evals[2]
            avg_fwavacc += evals[3]
            del f7, f6, f5, f4, f3, f2, f1, loss
        avg_acc = avg_acc / (i + 1)
        avg_acc_cls = avg_acc_cls / (i + 1)
        avg_mean_iu = avg_mean_iu / (i + 1)
        avg_fwavacc = avg_fwavacc / (i + 1)
        print('Val [epoch:%3d loss:%.3f avg_acc:%.3f avg_acc_cls:%.3f miou:%.3f fwavacc:%.3f]' % (
            epoch + 1, running_loss / (i + 1), avg_acc, avg_acc_cls, avg_mean_iu, avg_fwavacc))
        running_loss = 0.0
        if avg_mean_iu > best_miou_test:
            last_name1 = 'best_test_' + str(best_miou_test) + '.pth'
            best_miou_test = avg_mean_iu
            new_name1 = 'best_test_' + str(best_miou_test) + '.pth'
            torch.save(net.state_dict(), model_dir + '/' + new_name1)
            if last_name1 != 'best_test_0.pth':
                os.remove(model_dir + '/' + last_name1)
