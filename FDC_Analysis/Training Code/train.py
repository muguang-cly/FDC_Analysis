import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
import time
device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device1)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


def z_score(signal):
    mean_signal = np.mean(signal)
    std_signal = max(signal) - min(signal)
    if std_signal == 0:
        return signal - mean_signal
    signal_re = (signal - mean_signal) / std_signal
    return signal_re

def qiepian_data_label(list_line, list_line_label, qiepian_window_long=150, initial_site=0):
    window_data_line = []
    for i in range(int(len(list_line) / qiepian_window_long) + 1):
        start_index = initial_site + i * qiepian_window_long
        end_index = initial_site + (i + 1) * qiepian_window_long
        pian = list_line[max(start_index, 0):min(end_index, len(list_line))]
        window_data_line.append(pian)
        if len(window_data_line[0]) != qiepian_window_long:
            window_data_line[0] = [list_line[0]] * (qiepian_window_long - len(window_data_line[0])) + list(
                window_data_line[0])
        if len(window_data_line[-1]) != qiepian_window_long:
            window_data_line[-1] = list(window_data_line[-1]) + [list_line[-1]] * (
                        qiepian_window_long - len(window_data_line[-1]))
    window_data_line_label = []
    for i in range(int(len(list_line_label) / qiepian_window_long) + 1):
        start_index = initial_site + i * qiepian_window_long
        end_index = initial_site + (i + 1) * qiepian_window_long
        pian = list_line_label[max(start_index, 0):min(end_index, len(list_line))]
        window_data_line_label.append(pian)
        if len(window_data_line_label[0]) != qiepian_window_long:
            window_data_line_label[0] = [list_line[0]] * (qiepian_window_long - len(window_data_line_label[0])) + list(
                window_data_line_label[0])
        if len(window_data_line_label[-1]) != qiepian_window_long:
            window_data_line_label[-1] = list(window_data_line_label[-1]) + [list_line[-1]] * (
                        qiepian_window_long - len(window_data_line_label[-1]))
    return window_data_line, window_data_line_label


def get_formatted_time(seconds):
    hours = round(seconds // 3600)
    minutes = round((seconds % 3600) // 60)
    seconds = round(seconds % 60)

    return f'{hours:02}:{minutes:02}:{seconds:02}'


def data_proceeding(data_3, sample_long=2000, downsample_rate=1):
    data = np.array(data_3, dtype='float')[::downsample_rate]
    force_noise = data[:, 0:1 * sample_long]
    distance_noise = data[:, 1 * sample_long:2 * sample_long]
    force = data[:, 2 * sample_long:3 * sample_long]
    distance = data[:, 3 * sample_long:4 * sample_long]
    label_4000_4045 = data[:, 4 * sample_long:4 * sample_long + 45]
    fold_num = label_4000_4045[:, 0]
    force1 = label_4000_4045[:, 1:7]
    force2 = label_4000_4045[:, 7:13]
    distance1 = label_4000_4045[:, 13:19]
    distance2 = label_4000_4045[:, 19:25]
    fold_size_list = label_4000_4045[:, 25:31]
    P_list = label_4000_4045[:, 31:38]
    L_list = label_4000_4045[:, 38:45]

    re_force_noise = []
    re_distance_noise = []
    re_force = []
    re_distance = []
    qiepian_window_long = 300
    for i in range(len(force_noise)):
        # if i % 1000 == 0:
        #     print(i)
        initial_site = np.random.randint(-int(qiepian_window_long / 4), 0)
        sample_data_f, sample_label_f = qiepian_data_label(list_line=force_noise[i], list_line_label=force[i],
                                                           qiepian_window_long=qiepian_window_long,
                                                           initial_site=initial_site)
        sample_data_d, sample_label_d = qiepian_data_label(list_line=distance_noise[i], list_line_label=distance[i],
                                                           qiepian_window_long=qiepian_window_long,
                                                           initial_site=initial_site)

        re_force_noise_block = []
        re_force_block = []
        for signal1, signal2 in zip(sample_data_f, sample_label_f):
            mean_signal = np.mean(signal1)
            std_signal = max(signal1) - min(signal1)
            if std_signal == 0:
                data_f_re1 = signal1 - mean_signal
                label_f_re2 = signal2 - mean_signal
                re_force_noise_block.append(data_f_re1)
                re_force_block.append(label_f_re2)
                continue
            data_f_re1 = (signal1 - mean_signal) / std_signal
            label_f_re2 = (signal2 - mean_signal) / std_signal
            re_force_noise_block.append(data_f_re1)
            re_force_block.append(label_f_re2)

        re_distance_noise_block = []
        re_distance_block = []
        for signal1, signal2 in zip(sample_data_d, sample_label_d):
            mean_signal = np.mean(signal1)
            std_signal = max(signal1) - min(signal1)
            if std_signal == 0:
                data_f_re1 = signal1 - mean_signal
                label_f_re2 = signal2 - mean_signal
                re_distance_noise_block.append(data_f_re1)
                re_distance_block.append(label_f_re2)
                continue
            data_f_re1 = (signal1 - mean_signal) / std_signal
            label_f_re2 = (signal2 - mean_signal) / std_signal
            re_distance_noise_block.append(data_f_re1)
            re_distance_block.append(label_f_re2)


        re_force_noise.append(re_force_noise_block)
        re_distance_noise.append(re_distance_noise_block)
        re_force.append(re_force_block)
        re_distance.append(re_distance_block)
    re_force_noise = np.array([item for sublist in re_force_noise for item in sublist])
    re_distance_noise = np.array([item for sublist in re_distance_noise for item in sublist])
    re_force = np.array([item for sublist in re_force for item in sublist])
    re_distance = np.array([item for sublist in re_distance for item in sublist])



    re_force_noise = re_force_noise.reshape(re_force_noise.shape[0], 1, re_force_noise.shape[1])
    re_distance_noise = re_distance_noise.reshape(re_distance_noise.shape[0], 1, re_distance_noise.shape[1])
    data_4000_normalization_re = np.concatenate((
        re_force_noise,
        re_distance_noise,
    ), axis=1)

    re_force = re_force.reshape(re_force.shape[0], 1, re_force.shape[1])
    re_distance = re_distance.reshape(re_distance.shape[0], 1, re_distance.shape[1])

    label_4000_normalization_re = np.concatenate((
        re_force,
        re_distance
    ), axis=1)



    data = np.array(data_3, dtype='float')[::downsample_rate]
    force_noise = data[:, 0:1 * sample_long][:, ::-1]
    distance_noise = data[:, 1 * sample_long:2 * sample_long][:, ::-1]
    force = data[:, 2 * sample_long:3 * sample_long][:, ::-1]
    distance = data[:, 3 * sample_long:4 * sample_long][:, ::-1]
    label_4000_4045 = data[:, 4 * sample_long:4 * sample_long + 45]
    fold_num = label_4000_4045[:, 0]
    force1 = label_4000_4045[:, 1:7]
    force2 = label_4000_4045[:, 7:13]
    distance1 = label_4000_4045[:, 13:19]
    distance2 = label_4000_4045[:, 19:25]
    fold_size_list = label_4000_4045[:, 25:31]
    P_list = label_4000_4045[:, 31:38]
    L_list = label_4000_4045[:, 38:45]

    re_force_noise = []
    re_distance_noise = []
    re_force = []
    re_distance = []
    for i in range(len(force_noise)):
        # if i % 1000 == 0:
        #     print(i)
        initial_site = np.random.randint(-int(qiepian_window_long / 4), 0)
        sample_data_f, sample_label_f = qiepian_data_label(list_line=force_noise[i], list_line_label=force[i],
                                                           qiepian_window_long=qiepian_window_long,
                                                           initial_site=initial_site)
        sample_data_d, sample_label_d = qiepian_data_label(list_line=distance_noise[i], list_line_label=distance[i],
                                                           qiepian_window_long=qiepian_window_long,
                                                           initial_site=initial_site)

        re_force_noise_block = []
        re_force_block = []
        for signal1, signal2 in zip(sample_data_f, sample_label_f):
            mean_signal = np.mean(signal1)
            std_signal = max(signal1) - min(signal1)
            if std_signal == 0:
                data_f_re1 = signal1 - mean_signal
                label_f_re2 = signal2 - mean_signal
                re_force_noise_block.append(data_f_re1)
                re_force_block.append(label_f_re2)
                continue
            data_f_re1 = (signal1 - mean_signal) / std_signal
            label_f_re2 = (signal2 - mean_signal) / std_signal
            re_force_noise_block.append(data_f_re1)
            re_force_block.append(label_f_re2)

        re_distance_noise_block = []
        re_distance_block = []
        for signal1, signal2 in zip(sample_data_d, sample_label_d):
            mean_signal = np.mean(signal1)
            std_signal = max(signal1) - min(signal1)
            if std_signal == 0:
                data_f_re1 = signal1 - mean_signal
                label_f_re2 = signal2 - mean_signal
                re_distance_noise_block.append(data_f_re1)
                re_distance_block.append(label_f_re2)
                continue
            data_f_re1 = (signal1 - mean_signal) / std_signal
            label_f_re2 = (signal2 - mean_signal) / std_signal
            re_distance_noise_block.append(data_f_re1)
            re_distance_block.append(label_f_re2)

        re_force_noise.append(re_force_noise_block)
        re_distance_noise.append(re_distance_noise_block)
        re_force.append(re_force_block)
        re_distance.append(re_distance_block)
    re_force_noise = np.array([item for sublist in re_force_noise for item in sublist])
    re_distance_noise = np.array([item for sublist in re_distance_noise for item in sublist])
    re_force = np.array([item for sublist in re_force for item in sublist])
    re_distance = np.array([item for sublist in re_distance for item in sublist])



    re_force_noise = re_force_noise.reshape(re_force_noise.shape[0], 1, re_force_noise.shape[1])
    re_distance_noise = re_distance_noise.reshape(re_distance_noise.shape[0], 1, re_distance_noise.shape[1])
    data_4000_normalization_augment = np.concatenate((
        re_force_noise,
        re_distance_noise,
    ), axis=1)

    re_force = re_force.reshape(re_force.shape[0], 1, re_force.shape[1])
    re_distance = re_distance.reshape(re_distance.shape[0], 1, re_distance.shape[1])

    label_4000_normalization_augment = np.concatenate((
        re_force,
        re_distance
    ), axis=1)



    data_4000_normalization_augment_all = np.concatenate((
        data_4000_normalization_re,
        data_4000_normalization_augment,
    ), axis=0)
    label_4000_normalization_augment_all = np.concatenate((
        label_4000_normalization_re,
        label_4000_normalization_augment,
    ), axis=0)


    return data_4000_normalization_augment_all, label_4000_normalization_augment_all

# denoising model
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, T = x.size()
        query = self.query(x).view(batch_size, -1, T).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, T)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        value = self.value(x).view(batch_size, -1, T)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, T)
        out = self.gamma * out + x
        return out


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, hid_channels, kernel_size=[59, 29, 15, 7]):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Conv1d(in_channels, hid_channels, kernel_size=1, stride=1)
        self.branch2 = nn.Conv1d(hid_channels, hid_channels, kernel_size=kernel_size[0], stride=1,
                                 padding=kernel_size[0] // 2)
        self.branch3 = nn.Conv1d(hid_channels, hid_channels, kernel_size=kernel_size[1], stride=1,
                                 padding=kernel_size[1] // 2)
        self.branch4 = nn.Conv1d(hid_channels, hid_channels, kernel_size=kernel_size[2], stride=1,
                                 padding=kernel_size[2] // 2)
        self.branch5 = nn.Conv1d(hid_channels, hid_channels, kernel_size=kernel_size[3], stride=1,
                                 padding=kernel_size[3] // 2)
        self.maxpool = nn.MaxPool1d(kernel_size=9, stride=1, padding=4)
        self.branch6 = nn.Conv1d(in_channels, hid_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.branch1(x))
        branch1 = self.relu(self.branch2(x1))
        branch2 = self.relu(self.branch3(x1))
        branch3 = self.relu(self.branch4(x1))
        branch4 = self.relu(self.branch5(x1))
        branch5 = self.branch6(self.maxpool(x))

        outputs = [branch1, branch2, branch3, branch4, branch5]
        return torch.cat(outputs, dim=1)


class Inception(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_channels * 5, kernel_size=1)
        self.relu = nn.ReLU()

        self.inception_block1 = InceptionBlock(in_channels=hidden_channels * 5, hid_channels=hidden_channels,
                                               kernel_size=[59, 29, 15, 7])
        self.inception_block2 = InceptionBlock(in_channels=hidden_channels * 5, hid_channels=hidden_channels,
                                               kernel_size=[59, 29, 15, 7])
        self.inception_block3 = InceptionBlock(in_channels=hidden_channels * 5, hid_channels=hidden_channels,
                                               kernel_size=[59, 29, 15, 7])
        self.self_attention1 = SelfAttention(hidden_channels * 5)

        self.inception_block4 = InceptionBlock(in_channels=hidden_channels * 5, hid_channels=hidden_channels,
                                               kernel_size=[59, 29, 15, 7])
        self.inception_block5 = InceptionBlock(in_channels=hidden_channels * 5, hid_channels=hidden_channels,
                                               kernel_size=[59, 29, 15, 7])
        self.inception_block6 = InceptionBlock(in_channels=hidden_channels * 5, hid_channels=hidden_channels,
                                               kernel_size=[59, 29, 15, 7])
        self.self_attention2 = SelfAttention(hidden_channels * 5)

        self.conv_final = nn.Conv1d(hidden_channels * 5, output_channels, kernel_size=1)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        residual = x

        x = self.inception_block1(x)
        x = self.inception_block2(x)
        x = self.inception_block3(x)
        x = self.self_attention1(x)
        x += residual
        residual = x
        x = self.relu(x)
        x = self.inception_block4(x)
        x = self.inception_block5(x)
        x = self.inception_block6(x)
        x = self.self_attention2(x)
        x += residual
        x = self.relu(x)
        x = self.conv_final(x)
        return x

if __name__ == "__main__":

    filepath=r'fold count2.csv' #generate by FDC Simulation Model
    data_fold2= np.array(pd.read_csv(filepath, header=None))
    print(data_fold2.shape)
    filepath=r'fold count3.csv' #generate by FDC Simulation Model
    data_fold3= np.array(pd.read_csv(filepath, header=None))
    print(data_fold3.shape)
    tran_data_fold2,label_data_fold2=data_proceeding(data_fold2, sample_long=2000, downsample_rate=1)
    tran_data_fold3,label_data_fold3=data_proceeding(data_fold3, sample_long=2000, downsample_rate=1)
    data_normalization_augment_all = np.concatenate((
        tran_data_fold2,
        tran_data_fold3,
    ), axis=0)
    label_normalization_augment_all = np.concatenate((
        label_data_fold2,
        label_data_fold3,
    ), axis=0)




    train_data = torch.tensor(data_normalization_augment_all, dtype=torch.float32)
    target_data= torch.tensor(label_normalization_augment_all, dtype=torch.float32)

    print(train_data.shape)
    print(target_data.shape)

    dataset = torch.utils.data.TensorDataset(train_data, target_data)

    train_size = int(0.9 * len(dataset))
    val_size = int(0.05 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])


    batch_size = 50
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=100)
    print('success')





    input_channels  = 2
    hidden_channels = 16
    output_channels = 2

    model = Inception(input_channels, hidden_channels,output_channels)

    # model = bestmodel
    device = device1
    model.train()
    model.to(device)
    start_time = time.time()
    num_epochs = 200

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    best_val_loss = float('inf')
    bestmodel = None
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):

        train_loss = 0.0
        for i, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        if epoch == 0:
            end_time2 = time.time()
            total_time = round(end_time2 - start_time)
            formatted_time = get_formatted_time(total_time)
            print('——' * 20)
            print(f'time of one epoch: {total_time}(s),  {formatted_time}')
            print(
                f'total time of {num_epochs} epochs: {total_time * num_epochs}(s),  {get_formatted_time(total_time * num_epochs)}')
            print('——' * 20)
            print('bestmodol_loss:', best_val_loss)


        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for j, (data, label) in enumerate(val_loader):
                data = data.to(device)
                label = label.to(device)

                output = model(data)

                loss = criterion(output, label)

                val_loss += loss.item()


            avg_val_loss = val_loss / len(val_loader)
            val_loss_history.append(avg_val_loss)


            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                bestmodel = model
                print('Epoch [{}/{}], Train Loss: {:.8f}, Val Loss: {:.8f} ↓'
                      .format(epoch + 1, num_epochs, avg_train_loss, avg_val_loss))


                # linewidth = 0.6
                # tushu = 3
                # k = np.random.randint(0, len(data))
                # data = data.cpu()
                # output = output.cpu()
                # force_origin = data[k, 0]
                # distance_origin = data[k, 1]
                # force_p = output[k, 0]
                # distance_p = output[k, 1]
                # plt.figure(figsize=(15, 5), dpi=300)
                #
                # plt.subplot(1, tushu, 1)
                # plt.plot(distance_origin, force_origin, linewidth=0.6)
                # plt.plot(distance_p, force_p, color='black', linewidth=0.6)
                #
                # plt.subplot(1, tushu, 2)
                # plt.plot(distance_origin, linewidth=linewidth)
                # plt.plot(distance_p, color='black', linewidth=linewidth)
                #
                # plt.subplot(1, tushu, 3)
                # plt.plot(force_origin, linewidth=linewidth)
                # plt.plot(force_p, color='black', linewidth=linewidth)
                # plt.show()


            else:
                print('Epoch [{}/{}], Train Loss: {:.8f}, Val Loss: {:.8f}'
                      .format(epoch + 1, num_epochs, avg_train_loss, avg_val_loss))


    end_time = time.time()
    total_time = round(end_time - start_time)
    formatted_time = get_formatted_time(total_time)
    print(f'总时间: {total_time}（秒）, 总时间: {formatted_time}')
    print(f'最佳验证损失: {best_val_loss}')

    # torch.save(bestmodel.state_dict(), "bestmodel.pth")

