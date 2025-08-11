import torch
import sys
import scipy.integrate as integrate
from scipy.optimize import curve_fit
from scipy.optimize import brentq
import numpy as np
from pathlib import Path
def resource_path(rel_path: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / rel_path
def re_zscore(mean_data, std_data, data):
    return data * std_data + mean_data


def extract_and_resample(data, target_length=1000):
    index = 0
    for i in range(1, len(data)):
        if data[-i] != 0:
            index = len(data) - i
            break
    valid_data = data[:index]
    return valid_data
    interp_func = interp1d(np.linspace(0, 1, len(valid_data)), valid_data, kind='linear')

    return interp_func(np.linspace(0, 1, target_length))

def qiepian_real_data_max_min(list_f=[], list_d=[], qiepian_window_long=300, chongdie_rate=0.5):
    data_f = []
    data_d = []
    index_start = []
    index_end = []
    mean_f_list = []
    mean_d_list = []
    std_f_list = []
    std_d_list = []
    end_index = int(chongdie_rate * qiepian_window_long)
    for i in range(int(len(list_f) / (1 - chongdie_rate) / qiepian_window_long)):
        start_index = int(end_index - chongdie_rate * qiepian_window_long)
        end_index = int(start_index + qiepian_window_long)
        index_start.append(start_index)
        index_end.append(end_index)
        pian_f = list_f[start_index:end_index]
        pian_d = list_d[start_index:end_index]
        data_f.append(pian_f)
        data_d.append(pian_d)
        if len(data_f[-1]) != qiepian_window_long:
            data_f[-1] = list(data_f[-1]) + [data_f[-1][-1]] * (qiepian_window_long - len(data_f[-1]))
            data_d[-1] = list(data_d[-1]) + [data_d[-1][-1]] * (qiepian_window_long - len(data_d[-1]))
    for i, (signalf, signald) in enumerate(zip(data_f, data_d)):
        mean_f = np.mean(signalf)
        mean_d = np.mean(signald)
        std_f = max(signalf) - min(signalf)
        std_d = max(signald) - min(signald)
        if std_f == 0 or std_d == 0:
            data_f[i] = (signalf - mean_f)
            data_d[i] = (signald - mean_d)
            mean_f_list.append(mean_f)
            mean_d_list.append(mean_d)
            std_f_list.append(std_f)
            std_d_list.append(std_d)
            continue
        data_f[i] = (signalf - mean_f) / std_f
        data_d[i] = (signald - mean_d) / std_d
        mean_f_list.append(mean_f)
        mean_d_list.append(mean_d)
        std_f_list.append(std_f)
        std_d_list.append(std_d)
    return data_f, data_d, index_start, index_end, mean_f_list, mean_d_list, std_f_list, std_d_list

def merge_lists_part_average(lists, index_start, index_end):
    max_len = max(index_end)
    merged = [0] * max_len
    count = [0] * max_len
    for lst, start, end in zip(lists, index_start, index_end):
        for i in range(start, end):
            merged[i] += lst[i - start]
            count[i] += 1
    for i in range(max_len):
        if count[i] > 0:
            merged[i] /= count[i]

    return merged

def locate_mult_fold(force_p, distance_p, yuzhi_peak=-0.8, chuangkou=10):
    data = force_p
    data2 = distance_p
    lst = data
    daoshu = [lst[i + chuangkou] - lst[i] for i in range(len(lst) - chuangkou)]
    daoshu2 = [daoshu[i + chuangkou] - daoshu[i] for i in range(len(daoshu) - chuangkou)]

    index_down = 0
    index_up = 0
    point_num = 0
    point_range = []
    point_list_index = []
    point_4X_list = []
    for i in range(len(daoshu2)):
        if daoshu2[i] < yuzhi_peak and index_down == 0:
            index_down = i
        if daoshu2[i] > abs(yuzhi_peak) and index_down != 0 and index_up == 0:
            index_up = i
        if daoshu2[i] < abs(yuzhi_peak) / 2 and index_down != 0 and index_up != 0:
            index_up2 = i
            point_range.append([index_down + chuangkou, index_up2 + chuangkou])
            point_num = point_num + 1
            index_down = 0
            index_up = 0
    for pointrange in point_range:
        point_list = np.arange(pointrange[0], pointrange[-1])
        value_list = [data[i] for i in point_list]
        max_value = max(value_list)
        point_index_max = point_list[value_list.index(max_value)]
        min_value = min(value_list)
        point_index_min = point_list[value_list.index(min_value)]
        if point_index_max > point_index_min:
            middle_index = int((point_index_max + point_index_min) / 2)
            point_list1 = np.arange(point_index_min, middle_index + 1)
            point_list2 = np.arange(middle_index, point_index_max)

            value_list = [data[i] for i in point_list1]
            max_value = max(value_list)
            point_index_max = point_list1[value_list.index(max_value)]

            value_list = [data[i] for i in point_list2]
            min_value = min(value_list)
            point_index_min = point_list2[value_list.index(min_value)]

        point_4X_list.append(
            [data[point_index_max], data[point_index_min], data2[point_index_max], data2[point_index_min]])
        point_list_index.append([point_index_max, point_index_min])
    return point_4X_list, point_list_index



def wlc_model(x, P, L, C):
    k_B = 1.38e-23
    k_B = k_B * 1e21
    T = 298
    F = (k_B * T / P) * ((1 / (4 * (1 - x / L) ** 2)) - 1 / 4 + x / L) + C
    return F

def solve_d_brentq(F, popt):
    def equation(x):
        return wlc_model(x, *popt) - F

    return float(brentq(equation, 0, popt[1] - 1e-9))


def WLC_autofit(data_force, data_distance, point_index):

    L = max(data_distance)
    P = 60
    C = -5
    limit_P = 0.01
    limit_L = 0.1
    structure_size = []
    structure_size_direct = []
    S_fold_list = []
    P_L_C_list = []
    if len(point_index) != 0:
        for i in range(len(point_index) + 1):
            if i == len(point_index):
                index_1 = point_index[i - 1][1] + 2
                index_2 = len(data_distance)
            if i != 0 and i != len(point_index):
                index_1 = point_index[i - 1][1] + 2
                index_2 = point_index[i][0] - 2
            if i == 0:
                index_1 = 0
                index_2 = point_index[i][1] - 2

            x_data = data_distance[index_1:index_2]
            y_data = data_force[index_1:index_2]

            if i == 0:
                popt, pcov = curve_fit(wlc_model, x_data, y_data, p0=[P, max(x_data) + 2, C],
                                       bounds=([10, max(x_data) + 1, -10], [80, 2000, 10]),
                                       method='trf', maxfev=10000)
            if i != 0:
                popt, pcov = curve_fit(wlc_model, x_data, y_data, p0=[P, max(x_data) + 1.001, C],
                                       bounds=([P - abs(P * limit_P), max(x_data) + 1, C - abs(C * limit_L)],
                                               [P + abs(P * limit_P), 2000, C + abs(C * limit_L)]),
                                       method='trf', maxfev=10000)
            P = popt[0]
            L = popt[1]
            C = popt[2]
            P_L_C_list.append([popt[0], popt[1], popt[2]])

            if i != len(point_index):
                force1 = data_force[point_index[i][0]]
                force2 = data_force[point_index[i][1]]
                distance1 = data_distance[point_index[i][0]]
                distance2 = data_distance[point_index[i][1]]
                distance_recover = solve_d_brentq(force2, popt)
                structure_size.append(distance2 - distance_recover)
                structure_size_direct.append(distance2 - distance1)

        S_fold_list = []
        for i in range(len(structure_size)):
            def f(x):
                return wlc_model(x, P=P_L_C_list[i][0], L=P_L_C_list[i][1], C=P_L_C_list[i][2])

            C = P_L_C_list[i][2]
            distance1 = data_distance[point_index[i][0]]
            distance2 = data_distance[point_index[i][1]]
            force1 = f(distance1)
            a = 100
            b = distance1
            S_triangle_up, error = integrate.quad(f, a, b)
            S_triangle_up = S_triangle_up - C * (b - a)

            def f(x):
                return wlc_model(x, P=P_L_C_list[i + 1][0], L=P_L_C_list[i + 1][1], C=P_L_C_list[i + 1][2])

            b = distance2
            force2 = f(distance2)
            S_trapezoid = (force1 + force2 - 2 * C) * (distance2 - distance1) * 0.5

            S_triangle_down, error = integrate.quad(f, a, b)
            S_triangle_down = S_triangle_down - C * (b - a)
            S_cha = S_triangle_up + S_trapezoid - S_triangle_down
            S_fold_list.append(S_cha)
        S_fold_list = np.array(S_fold_list) * 0.6022
    else:
        x_data = data_distance
        y_data = data_force
        popt, pcov = curve_fit(wlc_model, x_data, y_data, p0=[P, L + 2, C],
                               bounds=([10, max(x_data) + 1, -10], [80, 2000, 10]),
                               method='trf', maxfev=10000)

        P = popt[0]
        L = popt[1]
        C = popt[2]

        P_L_C_list.append([popt[0], popt[1], popt[2]])
    return P_L_C_list,structure_size,structure_size_direct,S_fold_list


def model_out(force,distance):
    model_path = resource_path("InSAF_Model.pt")
    model = torch.jit.load(str(model_path), map_location=torch.device("cpu"))
    # model = torch.jit.load(r"InSAF_Model.pt", map_location=torch.device('cpu'))



    fold_num_all = []

    re_distance=distance
    re_force=force

    qiepian_window_long = 300

    data_f, data_d, index_start, index_end, mean_f_list, mean_d_list, std_f_list, std_d_list = qiepian_real_data_max_min(
        list_f=re_force,
        list_d=re_distance,
        qiepian_window_long=qiepian_window_long,
        chongdie_rate=0.98)

    data_f = np.array(data_f)
    data_d = np.array(data_d)


    data_f = data_f.reshape(data_f.shape[0], 1, data_f.shape[1])
    data_d = data_d.reshape(data_d.shape[0], 1, data_d.shape[1])
    realdata_normalization = np.concatenate((
        data_f,
        data_d,
    ), axis=1)
    realdata_normalization = torch.tensor(realdata_normalization).float()

    data = realdata_normalization

    batch_process_size = 100
    num_samples = realdata_normalization.shape[0]

    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_process_size):
            batch_data = realdata_normalization[i:i + batch_process_size]
            batch_output = model(batch_data)
            outputs.append(batch_output)
    output = torch.cat(outputs, dim=0)



    output = output.cpu().detach().numpy()
    force_origin = np.array([re_zscore(mean_f_list[i], std_f_list[i], data[i, 0]) for i in range(len(data))])
    distance_origin = np.array([re_zscore(mean_d_list[i], std_d_list[i], data[i, 1]) for i in range(len(data))])
    force_p = np.array([re_zscore(mean_f_list[i], std_f_list[i], output[i, 0]) for i in range(len(output))])
    distance_p = np.array([re_zscore(mean_d_list[i], std_d_list[i], output[i, 1]) for i in range(len(output))])


    cutoffset_long = 2
    force_origin = merge_lists_part_average(force_origin, index_start, index_end)[:len(re_distance) - cutoffset_long]
    distance_origin = merge_lists_part_average(distance_origin, index_start, index_end)[
                      :len(re_distance) - cutoffset_long]
    force_p = merge_lists_part_average(force_p, index_start, index_end)[:len(re_distance) - cutoffset_long]
    distance_p = merge_lists_part_average(distance_p, index_start, index_end)[:len(re_distance) - cutoffset_long]

    point_list, point_index = locate_mult_fold(force_p=force_p, distance_p=distance_p, yuzhi_peak=-0.4, chuangkou=10)
    fold_num_all.append(len(point_list))

    P_L_C_list, structure_size, structure_size_direct, S_fold_list=WLC_autofit(data_force=force_p,
                      data_distance=distance_p,
                      point_index=point_index)
    data_out= {
        'origin_force': force_origin,
        'origin_distance': distance_origin,
        'denoise_force': force_p,
        'denoise_distance': distance_p,
        'fold_num': len(point_list),
        'PLC': P_L_C_list,
        'actual_structure_size': structure_size,
        'structure_size_direct': structure_size_direct,
        'free_energy': S_fold_list,
        'fold_site': point_list,
        'point_index': point_index
    }
    return data_out
    # return force_origin,distance_origin,force_p,distance_p,len(point_list),P_L_C_list,structure_size,structure_size_direct,S_fold_list,point_list
    #
    # print(f'origin_force：{force_origin}')
    # print(f'origin_distance：{distance_origin}')
    # print(f'denoise_force：{force_p}')
    # print(f'denoise_distance：{distance_p}')
    # print(f'fold_num：{len(point_list)}')
    # print(f'PLC：{P_L_C_list}')
    # print(f'actual_structure_size：{structure_size} nm')
    # print(f'structure_size_direct：{structure_size_direct} nm')
    # print(f'free_energy：{S_fold_list} kJ/mol')
    # print(f'fold_site：{point_list} kJ/mol')
