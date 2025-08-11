import random
from sklearn.utils import resample
from tqdm import tqdm
from scipy.signal import resample
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor
from sympy import symbols, solve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.optimize import brentq

font = FontProperties(fname="C:\Windows\Fonts\SimHei.ttf", size=12)


#WLC model
k_B = 1.38e-23  # Boltzmann constant, unit: J/K
k_B = k_B * 1e21   # Convert to pN·nm/K
T = 298  # Temperature, unit: K
P_d=[ 3.10400592e-11  ,2.14008502e-08 ,-1.73386648e-04 , 2.73277605e-01,
  6.77253265e+02]
p_std=[ 1.85626418e-08, -9.71028428e-07,  3.24060378e-06 , 6.20246120e-04,
 -1.25704257e-02,  1.74555969e-01]
def padding_0(list_label,long_target):
    list_label=list(list_label)
    if len(list_label)<long_target:
        list_label=list_label+[0]*(long_target-len(list_label))
    return list_label

def wlc_model(x, P, L):
    F = (k_B * T / P) * ((1 / (4 * (1 - x / L) ** 2)) - 1 / 4 + x / L)
    return F

def force_noise_generate(force_nonoise):
    force_maoci_noise=[]
    for li in force_nonoise:
        if li <= 20:
            force_std = np.polyval(p_std, li)
            force_noise_std = np.random.normal(0, force_std, 1)
            force_maoci_noise.append(force_noise_std)
        else:
            force_std = 0.1
            # force_std = np.random.uniform(0.9,0.13)
            force_maoci_noise.append(np.random.normal(0, force_std, 1))
    force_maoci_noise=[item for sublist in force_maoci_noise for item in sublist]
    force_maoci_noise=force_nonoise+np.array(force_maoci_noise)*np.random.uniform(0.9,1.2)
    return force_maoci_noise


def extract_and_resample(data, target_length=1030):
    index = 0
    for i in range(1, len(data)):
        if data[-i] != 0:
            index = len(data) - i
            break
    valid_data = data[:index]
    normalized_data = resample(valid_data, target_length)
    return normalized_data[15:target_length - 15]
def delete_0(data):
    index = 0
    for i in range(1, len(data)+1):
        if data[-i] != 0:
            index = len(data) - i
            break
    valid_data = data[:index+1]
    return valid_data
def offset_solve_jie(x0, v_trap, t, k,P,L):
    x_actual_solution_list=[]
    for time in  t:
        def equation_wlcmodel(x_actual):
            return x_actual - x0 - v_trap * time + (wlc_model(x_actual,P,L) / k)
        x_actual_solution_list.append(brentq(equation_wlcmodel, 0, L - 1e-9))
    return x_actual_solution_list

def solve_d(F, popt):
    x = symbols('x')
    equation = wlc_model(x, *popt) - F
    solution = solve(equation, x)
    real_solutions = [sol.evalf().as_real_imag()[0] for sol in solution]
    x_solve1 = [round(x,8) for x in real_solutions if x < popt[1] and x > 0]
    return float(x_solve1[0])

def solve_d_brentq(F,popt):
    def equation(x):
        return wlc_model(x, *popt) - F
    return float(brentq(equation, 0, popt[1] - 1e-9))

def plot_1x3(distance_list,force_list):
    plt.figure(figsize=(9,4),dpi=100)
    plt.subplot(1,3,1)
    # plt.scatter(np.arange(len(distance_list)),distance_list,s=1)
    plt.plot(np.arange(len(distance_list)),distance_list,linewidth=1)

    plt.ylabel('Distance (nm)')
    plt.subplot(1, 3, 2)
    # plt.scatter(np.arange(len(force_list)),force_list,s=1)
    plt.plot(np.arange(len(force_list)),force_list,linewidth=1)
    plt.ylabel('Force (pN)')
    plt.title(len(force_list))
    plt.subplot(1,3,3)
    plt.plot(distance_list, force_list)
    plt.scatter(distance_list, force_list,s=2,color='red')
    # plt.scatter(distance_sequence_up_and_down_merged[1:], force_sequence_up_and_down_merged[1:],s=2,color='black')
    # plt.scatter(distance_list_t_start, force_list_t_start,s=2,color='black')
    # plt.scatter(distance_list_t_end, force_list_t_end,s=2,color='black')
    plt.xlabel('Distance (nm)')
    plt.ylabel('Force (pN)')
    plt.tight_layout()
    plt.show()
def plot_2x3_scatter(distance_list,force_list,distance_list_noise,force_list_noise,force_list_t_start,force_list_t_end,distance_list_t_start,distance_list_t_end):
    plt.figure(figsize=(9,6),dpi=100)
    plt.subplot(2,3,1)
    # plt.scatter(np.arange(len(distance_list)),distance_list,s=1)
    plt.plot(np.arange(len(distance_list)),distance_list,linewidth=1)

    plt.ylabel('Distance (nm)')
    plt.subplot(2, 3, 2)
    # plt.scatter(np.arange(len(force_list)),force_list,s=1)
    plt.plot(np.arange(len(force_list)),force_list,linewidth=1)
    plt.ylabel('Force (pN)')
    plt.title(len(force_list))
    plt.subplot(2,3,3)
    plt.plot(distance_list, force_list)
    plt.scatter(distance_list, force_list,s=2,color='red')
    # plt.scatter(distance_sequence_up_and_down_merged[1:], force_sequence_up_and_down_merged[1:],s=2,color='black')
    # plt.scatter(distance_list_t_start, force_list_t_start,s=2,color='black')
    # plt.scatter(distance_list_t_end, force_list_t_end,s=2,color='black')
    plt.xlabel('Distance (nm)')
    plt.ylabel('Force (pN)')
    plt.subplot(2,3,4)
    # plt.scatter(np.arange(len(distance_list)),distance_list,s=1)
    plt.plot(np.arange(len(distance_list_noise)),distance_list_noise,linewidth=0.5)

    plt.ylabel('Distance (nm)')
    plt.subplot(2, 3, 5)
    # plt.scatter(np.arange(len(force_list)),force_list,s=1)
    plt.plot(np.arange(len(force_list_noise)),force_list_noise,linewidth=0.5)
    plt.ylabel('Force (pN)')
    plt.title(len(force_list_noise))
    plt.subplot(2,3,6)
    plt.plot(distance_list_noise, force_list_noise,linewidth=0.5)
    plt.scatter(distance_list_t_start,force_list_t_start,s=2,color='red', zorder=10)
    plt.scatter(distance_list_t_end,force_list_t_end,s=2,color='red', zorder=10)
    # plt.scatter(distance_list_noise, force_list_noise,s=1,color='red')
    # plt.scatter(distance_sequence_up_and_down_merged[1:], force_sequence_up_and_down_merged[1:],s=2,color='black')
    # plt.scatter(distance_list_t_start, force_list_t_start,s=2,color='black')
    # plt.scatter(distance_list_t_end, force_list_t_end,s=2,color='black')
    plt.xlabel('Distance (nm)')
    plt.ylabel('Force (pN)')

    plt.tight_layout()
    plt.show()

def plot_2x3(distance_list,force_list,distance_list_noise,force_list_noise):
    plt.figure(figsize=(9,6),dpi=100)
    plt.subplot(2,3,1)
    # plt.scatter(np.arange(len(distance_list)),distance_list,s=1)
    plt.plot(np.arange(len(distance_list)),distance_list,linewidth=1)

    plt.ylabel('Distance (nm)')
    plt.subplot(2, 3, 2)
    # plt.scatter(np.arange(len(force_list)),force_list,s=1)
    plt.plot(np.arange(len(force_list)),force_list,linewidth=1)
    plt.ylabel('Force (pN)')
    plt.title(len(force_list))
    plt.subplot(2,3,3)
    plt.plot(distance_list, force_list)
    plt.scatter(distance_list, force_list,s=2,color='red')
    # plt.scatter(distance_sequence_up_and_down_merged[1:], force_sequence_up_and_down_merged[1:],s=2,color='black')
    # plt.scatter(distance_list_t_start, force_list_t_start,s=2,color='black')
    # plt.scatter(distance_list_t_end, force_list_t_end,s=2,color='black')
    plt.xlabel('Distance (nm)')
    plt.ylabel('Force (pN)')
    plt.subplot(2,3,4)
    # plt.scatter(np.arange(len(distance_list)),distance_list,s=1)
    plt.plot(np.arange(len(distance_list_noise)),distance_list_noise,linewidth=0.5)

    plt.ylabel('Distance (nm)')
    plt.subplot(2, 3, 5)
    # plt.scatter(np.arange(len(force_list)),force_list,s=1)
    plt.plot(np.arange(len(force_list_noise)),force_list_noise,linewidth=0.5)
    plt.ylabel('Force (pN)')
    plt.title(len(force_list_noise))
    plt.subplot(2,3,6)
    plt.plot(distance_list_noise, force_list_noise,linewidth=0.5)
    # plt.scatter(distance_list_t_start,force_list_t_start,s=2,color='red')
    # plt.scatter(distance_list_t_end,force_list_t_end,s=2,color='red')
    # plt.scatter(distance_list_noise, force_list_noise,s=1,color='red')
    # plt.scatter(distance_sequence_up_and_down_merged[1:], force_sequence_up_and_down_merged[1:],s=2,color='black')
    # plt.scatter(distance_list_t_start, force_list_t_start,s=2,color='black')
    # plt.scatter(distance_list_t_end, force_list_t_end,s=2,color='black')
    plt.xlabel('Distance (nm)')
    plt.ylabel('Force (pN)')

    plt.tight_layout()
    plt.show()
def generate_sorted_random_list(start = -20,end = 20):

    total_numbers = int((end - start + 1)/2)
    count = random.randint(0, total_numbers)
    random_list = random.sample(range(start, end + 1), count)
    sorted_list = sorted(random_list)
    return sorted_list
def FD_generate(_):
    data_num=0
    while data_num==0:
        offset=np.random.uniform(-2,2) #Offset
        max_fold_num=6 #Maximum number of folds allowed by the program
        fold_num=2 #Fold Count
        P=np.random.uniform(30,60)#persistence
        L=np.random.uniform(500,1000)#Maximum length
        sample_long=2000
        min_force=3#Minimum force for fold
        max_force=40#Maximum force for fold
        max_force2_down=2 #Maximum descending force
        min_structure_d=5# Minimum size of folded structure
        max_structure_d=30#  Maximum size of folded structure
        distance_valve=5#Minimum structure extension threshold
        force_valve=0.5#Minimum structure force threshold
        maoci_force_yuzhi=np.random.uniform(1.5,2)  #Upper limit of force for generating fuzzy signals
        fold_size_list=[np.random.uniform(min_structure_d,max_structure_d) for i in range(fold_num)]
        L_list=[L+sum(fold_size_list[:i]) for i in range(fold_num+1)]
        P_list=[P for i in range(fold_num+1)]
        f1_fold_point_list = []#force1
        distance_f1_list = []#distance1
        f2_fold_point_list = []#force2
        distance_f2_list = []#distance2

        if fold_num !=0:
            f1_fold_point_list.append(np.random.uniform(min_force,max_force))
        for i in range(fold_num):
            distance_f1_list.append(solve_d_brentq(f1_fold_point_list[i], [P_list[i], L_list[i]]))
            limit_f1_min = wlc_model(distance_f1_list[i], P_list[i+1], L_list[i+1])
            f1_fold_point=np.random.uniform(min_force,max_force)
            while solve_d_brentq(f1_fold_point, [P_list[i + 1], L_list[i + 1]])<distance_f1_list[i]+0.01:
                f1_fold_point = np.random.uniform(min_force, max_force)
            f1_fold_point_list.append(f1_fold_point)
            if i==fold_num-1:
                f1_fold_point_list.remove(f1_fold_point_list[-1])
                break
            f2_fold_point_list.append(np.random.uniform(limit_f1_min, min(f1_fold_point_list[i], f1_fold_point_list[i + 1]-1)))
            distance_f2_list.append(solve_d_brentq(f2_fold_point_list[i], [P_list[i + 1], L_list[i + 1]]))
        limit_f1_min = wlc_model(distance_f1_list[-1], P_list[-1], L_list[-1])
        f2_fold_point_list.append(np.random.uniform(max(limit_f1_min,f1_fold_point_list[-1]-max_force2_down),  f1_fold_point_list[-1]))


        distance_f2_list.append(solve_d_brentq(f2_fold_point_list[-1], [P_list[-1], L_list[-1]]))
        force_sequence_up_and_down_merged = [elem for pair in zip(f1_fold_point_list, f2_fold_point_list) for elem in pair]
        distance_sequence_up_and_down_merged = [elem for pair in zip(distance_f1_list, distance_f2_list) for elem in pair]

        distance_sequence_up_and_down_merged.insert(0, np.random.uniform(0.75*L,0.8*L))
        force_sequence_up_and_down_merged.insert(0, wlc_model(distance_sequence_up_and_down_merged[0], P_list[0], L_list[0]))

        force_sequence_up_and_down_merged.append(np.random.uniform(max(force_sequence_up_and_down_merged)+2
                                                     , max(force_sequence_up_and_down_merged[-1]+20,max_force)))
        distance_sequence_up_and_down_merged.append(solve_d_brentq(force_sequence_up_and_down_merged[-1], [P_list[-1], L_list[-1]]))



        distance_all_L=[]
        distance_list=[]
        force_list=[]

        v_trap=np.random.uniform(10, 80)#Stretching speed
        x0 = L_list[0] * np.random.uniform(0.73, 0.78)#Initial length
        time_range=np.linspace(0,(max(L_list))/v_trap,2000)#time series

        for i in range(fold_num+1):
            distance_all_L.append(offset_solve_jie(x0=x0, v_trap=v_trap,
                             t=time_range,
                             k=0.3, P=P_list[i], L=L_list[i]))
        for i in range(fold_num+1):
            if i ==0:
                distance_list.append([x for x in distance_all_L[i] if x < distance_sequence_up_and_down_merged[1]])
            else:
                distance_list.append([x for x in distance_all_L[i]
                                      if x > distance_sequence_up_and_down_merged[2*i] and x < distance_sequence_up_and_down_merged[2*i+1]])
        for i in range(fold_num+1):
            force_list.append([wlc_model(x, P_list[i], L_list[i]) for x in distance_list[i]])


        re_force_list=[]
        re_distance_list=[]
        len_list=[len(i) for i in force_list]
        force_list_num=[round(i/sum(len_list)*sample_long) for i in len_list ]
        if sum(force_list_num)!=sample_long:
            force_list_num[-1]=force_list_num[-1]-sum(force_list_num)+sample_long
        for i in range(len(force_list_num)):
            if len(force_list[i]) > 0:
                interp_func = interp1d(np.linspace(0, 1, len(force_list[i])), force_list[i], kind='linear')
            else:
                print(f"Warning: force_list is empty, skipping interpolation.")
                continue
            try:
                interp_func = interp1d(np.linspace(0, 1, len(force_list[i])), force_list[i], kind='linear')
                re_force_list.append( interp_func(np.linspace(0, 1, force_list_num[i])))
                interp_func = interp1d(np.linspace(0, 1, len(distance_list[i])), distance_list[i], kind='linear')
                re_distance_list.append( interp_func(np.linspace(0, 1, force_list_num[i])))
            except Exception:
                continue

        distance_list=re_distance_list.copy()
        force_list=re_force_list.copy()
        len_re_list=[len(i) for i in distance_list]

        try:
            distance_list_t_start = [x[-1] for x in distance_list][:-1]
            distance_list_t_end = [x[0] for x in distance_list][1:]
            force_list_t_start = [x[-1] for x in force_list][:-1]
            force_list_t_end = [x[0] for x in force_list][1:]
        except Exception:
            print("-1 index error")
            continue


        distance_list=[item for sublist in distance_list for item in sublist]
        force_list=[item for sublist in force_list for item in sublist]




        differences = [abs(a - b) < distance_valve for a, b in zip(distance_list_t_start, distance_list_t_end)]
        if any(differences):
            continue
        differences = [abs(a - b) < force_valve for a, b in zip(force_list_t_start, force_list_t_end)]
        if any(differences):
            continue




        index_force_xiao_list=[]
        for i in range(len(force_list_t_start)):
            if (force_list_t_start[i]-force_list_t_end[i])<maoci_force_yuzhi:
            # if abs(distance_list_t_start[i] - distance_list_t_end[i]) < maoci_distance_yuzhi:
                index_force_xiao=sum(len_re_list[:i+1])
                index_force_xiao_list.append(index_force_xiao)
        # if len(index_force_xiao_list)!=0:
        #     print(index_force_xiao_list)
        distance_list_mohu=distance_list.copy()
        force_list_mohu=force_list.copy()
        distance_list_clear=distance_list.copy()
        force_list_clear=force_list.copy()

        for i in index_force_xiao_list:
            force_down_sharp=abs(force_list_mohu[i-1]-force_list_mohu[i])
            distance_down_sharp=abs(distance_list_mohu[i-1]-distance_list_mohu[i])
            index_mohu_list=generate_sorted_random_list(start=-15,end=15)
            index_mohu_finish=np.array(index_mohu_list)+i
            for xuhao,k in enumerate(index_mohu_finish):
                if k<i:
                    distance_list_mohu[k]=distance_list_mohu[k]+distance_down_sharp*np.random.uniform(0,1)
                    force_list_mohu[k]=force_list_mohu[k]-force_down_sharp*np.random.uniform(0,1)
                elif k > i:
                    distance_list_mohu[k] = distance_list_mohu[k] - distance_down_sharp*np.random.uniform(0,1)
                    force_list_mohu[k]=force_list_mohu[k]+force_down_sharp*np.random.uniform(0,1)
        distance_list=distance_list_mohu.copy()
        force_list=force_list_mohu.copy()

        #add noise
        # d_noise = np.random.normal(0, 3.63, len(distance_list))
        d_noise = np.random.normal(0, np.random.uniform(3,4), len(distance_list))
        distance_list_noise=np.array(distance_list)+d_noise
        force_list_noise = force_noise_generate(force_list)

        long_input=3000
        force_list_noise=np.array(force_list_noise)+offset
        distance_list_noise

        #noise-free data
        force_list=np.array(force_list_clear)+offset
        distance_list_clear
        # fold count
        fold_num
        # Force1
        force_list_t_start=np.array(force_list_t_start)+offset
        force_list_t_start=padding_0(force_list_t_start,max_fold_num)
        # Force2
        force_list_t_end=np.array(force_list_t_end)+offset
        force_list_t_end=padding_0(force_list_t_end,max_fold_num)
        # Distance1
        distance_list_t_start
        distance_list_t_start=padding_0(distance_list_t_start,max_fold_num)
        # Distance2
        distance_list_t_end
        distance_list_t_end=padding_0(distance_list_t_end,max_fold_num)
        # fold size
        fold_size_list
        fold_size_list=padding_0(fold_size_list,max_fold_num)
        #P
        P_list
        P_list=padding_0([P_list[0]],max_fold_num+1)
        #L
        L_list
        L_list=padding_0(L_list,max_fold_num+1)

        data_all=[]
        label_all=[]
        data_label_all=[]
        data_all.extend(force_list_noise) #0-1000
        data_all.extend(distance_list_noise)#1000-2000
        data_all.extend(force_list)#2000-3000
        data_all.extend(distance_list_clear)#3000-4000
        label_all.append(fold_num)#0
        label_all.extend(force_list_t_start)#1-7
        label_all.extend(force_list_t_end)#7-13
        label_all.extend(distance_list_t_start)#13-19
        label_all.extend(distance_list_t_end)#19-25
        label_all.extend(fold_size_list)#25-31
        label_all.extend(P_list)#31-38
        label_all.extend(L_list)#38-45
        data_all.extend(label_all)#data0-4000,label4000-4045
        if len(data_all)!=(4*sample_long+45):
            print('length error'+str(len(data_all)))
            continue
        data_num=1
    # print(len(data_all))
    # print(label_all)
    # print(len(label_all))
    return data_all

if __name__ == "__main__":
    '''
    num_row: generate quantity.
    The folding count , force extremum and distance extremum are set in the FD_generate() function.
    '''
    num_row=100
    with ProcessPoolExecutor(max_workers=20) as executor:
        # results = list(executor.map(FD_generate, range(num_row)))
        results = list(tqdm(executor.map(FD_generate, range(num_row)), total=num_row))
    data = np.array(results)
    print(data.shape)
    path=f'save_path.csv'

    # np.savetxt(path, data, delimiter=',')

    #check
    for k in range(20):
        sample_long=2000
        force_list_noise=data[k,0:sample_long]
        distance_list_noise=data[k,sample_long:2*sample_long]
        force_list=data[k,2*sample_long:3*sample_long]
        distance_list=data[k,3*sample_long:4*sample_long]
        plot_2x3_scatter(distance_list,force_list,distance_list_noise,force_list_noise,
                         delete_0(data[k,4*sample_long+1:4*sample_long+7]),delete_0(data[k,4*sample_long+7:4*sample_long+13]),delete_0(data[k,4*sample_long+13:4*sample_long+19]),delete_0(data[k,4*sample_long+19:4*sample_long+25]))


