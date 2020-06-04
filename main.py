# -*- coding: utf-8 -*-
"""
Description :   控制训练、验证
     Author :   Yang
       Date :   2020/6/4
"""
import os
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch

from data_loader.data_loader import load_data
from utils.load_model_optimizer_lr import load_model_op_lr
from utils.get_result import get_result
from utils.utils import get_config_from_json

config_file = 'config1.json'
config = get_config_from_json(config_file)

for k, v in config.items():
    print(k, ': ', v)

# 1.加载数据
print('loading data.')
if config['only_train']:
    train_loader = load_data(config)
else:
    train_loader, valid_loader, test_loader = load_data(config)
print('loaded data.')

# 2.加载模型、优化器、损失函数、学习率计划器
sentiment_model, optimizer, criterion, scheduler = load_model_op_lr(config, len(train_loader))


def validation(sentiment_model):
    y_pred = []
    y_true = []
    total_eval_loss = 0

    sentiment_model.eval()  # eval模式下dropout不会工作
    for i, (inp, tar) in enumerate(valid_loader):
        with torch.no_grad():
            pred = sentiment_model(inp)
            loss = criterion(pred, tar)

            total_eval_loss += loss.item()

            pred_list = torch.argmax(pred.cpu().detach(), dim=-1).tolist()
            tar_list = tar.cpu().detach().numpy().tolist()

            y_pred.extend(pred_list)
            y_true.extend(tar_list)

    avg_val_loss = total_eval_loss / len(valid_loader)
    acc = accuracy_score(tar_list, pred_list)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return avg_val_loss, acc, macro_f1


def test(sentiment_model):
    y_pred = []
    y_true = []
    total_test_loss = 0

    sentiment_model.eval()  # eval模式下dropout不会工作
    for i, (inp, tar) in enumerate(test_loader):
        with torch.no_grad():
            pred = sentiment_model(inp)
            loss = criterion(pred, tar)

            total_test_loss += loss.item()

            pred_list = torch.argmax(pred.cpu().detach(), dim=-1).tolist()
            tar_list = tar.cpu().detach().numpy().tolist()

            y_pred.extend(pred_list)
            y_true.extend(tar_list)

    avg_test_loss = total_test_loss / len(test_loader)
    acc = accuracy_score(tar_list, pred_list)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return avg_test_loss, acc, macro_f1


def train(sentiment_model):
    only_train = config['only_train']
    training_stats = []
    for epoch in range(config['epoch']):
        sentiment_model.train()

        # whole_epoch_logits = []  # 放在for循环内部确保取到的是最后一个epoch的logits
        # whole_epoch_labels = []

        total_train_loss = 0
        for batch, (inp, tar) in enumerate(train_loader):
            sentiment_model.zero_grad()

            pred = sentiment_model(inp)  # pred: logit

            # batch_op = OptimizedF1()
            # batch_op.fit(pred.cpu().detach().numpy(), tar.cpu().numpy())
            # pred = torch.tensor(batch_op.coefficients()) * pred.cpu()
            # loss = criterion(pred.cuda(), tar)  # 通过改变logit大小来改变loss，从而影响训练，希望参数更优

            loss = criterion(pred, tar)  # old pred, 没有阈值搜索

            # whole_epoch_logits.extend(pred)  # 也可以学得整体的参数，然后再将参数coef_用于评估

            total_train_loss += loss.item()

            loss.backward()  # 反向传播计算梯度
            torch.nn.utils.clip_grad_norm_(sentiment_model.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
            optimizer.step()  # 参数更新
            scheduler.step()  # 更新学习率

            pred_list = torch.argmax(pred.detach(), dim=-1).tolist()
            tar_list = tar.cpu().detach().numpy().tolist()

            macro_f1 = f1_score(tar_list, pred_list, average='macro')
            acc = accuracy_score(tar_list, pred_list)

            if batch % 50 == 0:
                print('epoch: {}, batch: {}, loss: {:.4f}, acc: {:.4f}, macro_f1: {:.4f}'.format(epoch,
                                                                                                 batch,
                                                                                                 loss,
                                                                                                 acc,
                                                                                                 macro_f1))
        avg_train_loss = total_train_loss / len(train_loader)
        print('train set avg loss:{:.4f},\n'.format(avg_train_loss))

        if not only_train:
            val_loss, val_acc, val_f1 = validation(sentiment_model)
            test_loss, test_acc, test_f1 = test(sentiment_model)  # 测试集上测试模型效果
            print('valid set:\n\t val loss: {:.4f}, val acc: {:.4f}, val f1: {:.4f}'.format(val_loss,
                                                                                            val_acc,
                                                                                            val_f1))
            print('test set:\n\t test loss: {:.4f}, test acc: {:.4f}, test f1: {:.4f}'.format(test_loss,
                                                                                              test_acc,
                                                                                              test_f1))

            training_stats.append(
                {
                    'epoch': epoch + 1,
                    'Train. Loss': avg_train_loss,
                    'Valid. Loss': val_loss,
                    'Valid. F1.': val_f1,
                    'Test. Loss': test_loss,
                    'Test. F1': test_f1
                }
            )

        torch.save(sentiment_model.state_dict(), os.path.join(config['finetune_dir'], "pytorch_model.bin"))
    # TODO 这里进行阈值搜索，测试时是否需要搜索策略还不得知
    # op.fit(whole_epoch_logits, whole_epoch_labels)

    return training_stats


def main():
    training_stats = train(sentiment_model)
    get_result(sentiment_model, config_file.split('.')[0] + '.csv')  # 生成可以提交的结果

    pd.set_option('precision', 4)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    print(df_stats)
    # df_stats.to_csv('train_vs_valid.csv')


if __name__ == "__main__":
    main()
    # nohup python -u main.py > config16.log 2>&1 &
