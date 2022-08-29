import time
import torch
import numpy as np
import sys
sys.path.append('../global_module/')
from global_module import d2lzh_pytorch as d2l

def evaluate_accuracy(data_iter, net, loss, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X1, X2, y in data_iter:
            test_l_sum, test_num = 0, 0
            X1 = X1.to(device)
            X2 = X2.to(device)
            y = y.to(device)
            net.eval() # 评估模式, 这会关闭dropout
            y_hat = net(X1,X2)
            l = loss(y_hat, y.long())
            # l=l1
            # l2 = loss2(out1, out2, y.long()-1)
            # l = 0.8 * l1+ 0.2 * l2
            acc_sum += (y_hat.argmax(dim=1).long() == y.long().to(device)).sum().cpu().item()
            test_l_sum += l
            test_num += 1
            net.train() # 改回训练模式
            n += y.shape[0]
    return [acc_sum / n, test_l_sum] # / test_num]

def train(net, train_iter, valida_iter, loss,  optimizer, device, epochs=10):
    loss_list = [100]
    early_epoch = 0
    best = 0
    net = net.to(device)
    print("training on ", device)
    start = time.time()
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    newnet = []
    for epoch in range(epochs):
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
        for X1, X2, y in train_iter:
            batch_count, train_l_sum = 0, 0
            X1 = X1.to(device)
            X2 = X2.to(device)
            y = y.to(device)
            y_hat = net(X1, X2)
            l= loss(y_hat, y.long())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1).long() == y.long()).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        lr_adjust.step(epoch)
        valida_acc, valida_loss = evaluate_accuracy(valida_iter, net, loss, device)
        loss_list.append(valida_loss)
        if valida_acc >= best:
            best = valida_acc
            torch.save(net,'best.pth')

        print(best)
        # 绘图部分
        train_loss_list.append(train_l_sum) # / batch_count)
        train_acc_list.append(train_acc_sum / n)
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(valida_acc)

        print('epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, time %.1f sec'
                % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, valida_loss, valida_acc, time.time() - time_epoch))

        PATH = "./net_DBA.pt"
        # if loss_list[-1] <= 0.01 and valida_acc >= 0.95:
        #     torch.save(net.state_dict(), PATH)
        #     break



    d2l.set_figsize()
    d2l.plt.figure(figsize=(8, 8.5))
    train_accuracy = d2l.plt.subplot(221)
    train_accuracy.set_title('train_accuracy')
    d2l.plt.plot(np.linspace(1, epoch, len(train_acc_list)), train_acc_list, color='green')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('train_accuracy')
    # train_acc_plot = np.array(train_acc_plot)
    # for x, y in zip(num_epochs, train_acc_plot):
    #    d2l.plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=11)

    test_accuracy = d2l.plt.subplot(222)
    test_accuracy.set_title('valida_accuracy')
    d2l.plt.plot(np.linspace(1, epoch, len(valida_acc_list)), valida_acc_list, color='deepskyblue')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('test_accuracy')
    # test_acc_plot = np.array(test_acc_plot)
    # for x, y in zip(num_epochs, test_acc_plot):
    #   d2l.plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=11)

    loss_sum = d2l.plt.subplot(223)
    loss_sum.set_title('train_loss')
    d2l.plt.plot(np.linspace(1, epoch, len(valida_acc_list)), valida_acc_list, color='red')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('train loss')
    # ls_plot = np.array(ls_plot)

    test_loss = d2l.plt.subplot(224)
    test_loss.set_title('valida_loss')
    #d2l.plt.plot(np.linspace(1, epoch, len(valida_loss_list)), valida_loss_list, color='gold')
    #d2l.plt.xlabel('epoch')
    #d2l.plt.ylabel('valida loss')
    # ls_plot = np.array(ls_plot)

    #d2l.plt.show()
    print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))

    return newnet


