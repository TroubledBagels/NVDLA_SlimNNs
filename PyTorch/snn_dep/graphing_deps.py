import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
import time
import torch

def get_grad_colours(all_thresholds):
    thresh_total = len(all_thresholds)
    wav_min = 440
    wav_max = 750
    wav_diff = wav_max - wav_min

    col_list = []

    gamma = 0.5
    for t in all_thresholds:
        ind = all_thresholds.index(t)
        interpolation = ind / thresh_total
        ind_wav = wav_min + (interpolation * wav_diff)
        if 380 <= ind_wav <= 440:
            attenuation = 0.3 + 0.7 * (ind_wav - 380) / (440-380)
            R = ((-(ind_wav - 440) / (440 - 380)) * attenuation) ** gamma
            G = 0.0
            B = (1.0 * attenuation) ** gamma
        elif 440 <= ind_wav <= 490:
            R = 0.0
            G = ((ind_wav - 440) / (490 - 440)) ** gamma
            B = 1.0
        elif 490 <= ind_wav <= 510:
            R = 0.0
            G = 1.0
            B = (-(ind_wav - 510) / (510 - 490)) ** gamma
        elif 510 <= ind_wav <= 580:
            R = ((ind_wav - 510) / (580 - 510)) ** gamma
            G = 1.0
            B = 0.0
        elif 580 <= ind_wav <= 645:
            R = 1.0
            G = (-(ind_wav - 645) / (645 - 580)) ** gamma
            B = 0.0
        elif 645 <= ind_wav <= 750:
            attenuation = 0.3 + 0.7 * (750 - ind_wav) / (750 - 645)
            R = (1.0 * attenuation) ** gamma
            G = 0.0
            B = 0.0
        else:
            R = G = B = 0.0
        col_list.append((R, G, B, 1.0))
    return col_list

def export_graphs(model_name, plots):
    plots[0][0].savefig("./model_images/" + model_name + "_acc.png")
    plots[1][0].savefig("./model_images/" + model_name + "_width_freq.png")
    plots[2][0].savefig("./model_images/" + model_name + "_avg_times.png")
    plots[3][0].savefig("./model_images/" + model_name + "_conf.png")

def run_thresholding_tests(model, dataloader, device, verbosity, threshold_list):
    all_correct_lists = []
    all_confidence_lists = []
    all_times = []
    all_widths = []

    for t in threshold_list:
        if verbosity > 0:
            print("[INFO] Testing threshold: " + str(t))
        cur_cor_list, cur_conf_list, cur_times, cur_widths = run_tests(model, dataloader, device, verbosity, t)
        all_correct_lists.append(cur_cor_list)
        all_confidence_lists.append(cur_conf_list)
        all_times.append(cur_times)
        all_widths.append(cur_widths)

    if verbosity > 0: print("[INFO] All tests complete.")

    return all_correct_lists, all_confidence_lists, all_times, all_widths

def create_graphs(all_cor, all_conf, all_times, all_widths, all_thresholds, wml):
    thresholds_str = [str(t) for t in all_thresholds]

    # Calculate accuracies for each threshold
    all_accs = [np.array(all_cor[i]).sum() / len(all_cor[i]) for i in range(len(all_cor))]

    # Calculate average confidence per thresholds
    mean_conf = [np.array(all_conf[i]).sum() / len(all_conf[i]) for i in range(len(all_conf))]

    # Calculate median confidence per thresholds
    median_conf = [np.median(all_conf[i]) for i in range(len(all_conf))]

    max_conf = [np.max(all_conf[i]) for i in range(len(all_conf))]
    min_conf = [np.min(all_conf[i]) for i in range(len(all_conf))]

    # Quartile confidence per thresholds
    q1_conf = [np.percentile(all_conf[i], 25) for i in range(len(all_conf))]
    q3_conf = [np.percentile(all_conf[i], 75) for i in range(len(all_conf))]

    # Create line graph for accuracies and avg confidence
    fig1, ax1 = plt.subplots()
    ax1.plot(thresholds_str, all_accs, label="Accuracy")
    ax1.set_xlabel('Threshold')
    ax1.set_title('Accuracy Results')

    # Calculate width frequency for each threshold
    temp_arr = [sorted(Counter(all_widths[i]).items()) for i in range(len(all_widths))]
    width_labels = [str(k) for k in wml]
    width_freqs = [[v for _, v in temp_arr[i]] for i in range(len(temp_arr))]

    # Create 3D bar chart for width frequencies
    col_type_list = get_grad_colours(all_thresholds)
    col_list = []
    for i in range(len(col_type_list)):
        for j in range(len(width_labels)):
            col_list.append(col_type_list[i])

    fig2 = plt.figure(figsize=(10, 7))
    ax2 = fig2.add_subplot(111, projection='3d')
    xpos, ypos = np.meshgrid(np.arange(len(width_labels)), np.arange(len(thresholds_str)))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.9

    print(width_labels)
    for i in range(len(width_freqs)):
        while len(width_freqs[i]) < len(width_labels):
            width_freqs[i].append(0)
        print(width_freqs[i])

    dz = np.array(width_freqs).flatten()

    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color=col_list, zsort='average')
    ax2.set_yticks(np.arange(len(thresholds_str)))
    ax2.set_yticklabels(thresholds_str)
    ax2.set_ylabel('Threshold')
    ax2.set_xticks(np.arange(len(width_labels)))
    ax2.set_xticklabels(width_labels)
    ax2.set_xlabel('Width')
    ax2.set_zlabel('Frequency')
    ax2.set_title('Width Frequency per Threshold')

    # Calculate average times
    avg_time = [1000 * (np.array(all_times[i]).sum() / len(all_times[i])) for i in range(len(all_times))]

    # Create line graph for average time
    fig3, ax3 = plt.subplots()
    ax3.plot(thresholds_str, avg_time)
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel("Average Time (ms)")
    ax3.set_title('Average Time per Threshold')


    fig4, ax4 = plt.subplots()
    ax4.plot(thresholds_str, mean_conf, label="Mean Confidence")
    ax4.plot(thresholds_str, median_conf, label="Median Confidence")
    ax4.plot(thresholds_str, q1_conf, label="Q1 Confidence")
    ax4.plot(thresholds_str, q3_conf, label="Q3 Confidence")
    ax4.plot(thresholds_str, max_conf, label="Max Confidence")
    ax4.plot(thresholds_str, min_conf, label="Min Confidence")
    ax4.set_xlabel('Threshold')
    ax4.set_title('Confidence Results')
    ax4.legend()

    plt.ion()
    plt.pause(0.001)
    return [(fig1, ax1), (fig2, ax2), (fig3, ax3), (fig4, ax4)]

def run_tests(model, dataloader, device, verbosity, threshold):
    size = len(dataloader)

    verb_mod = random.randint(30, 80)

    if verbosity > 0:
        print("[INFO] Testing model on", size, "samples")

    wml = model.width_mult_list

    correct_list = []
    confidence_list = []
    times = []
    width_list = []

    model.change_width_mult(wml[0])

    model.eval()

    i = 0

    with torch.no_grad():
        for X, y in dataloader:
            i += 1
            wm = wml[0]
            model.change_width_mult(wm)
            start_time = time.time()
            X, y = X.to(device), y.to(device)
            pred = None
            done = False

            while not done:
                pred = model.forward_train(X)
                # norm_pred = (pred / pred.sum())
                # norm_pred = torch.norm(pred, p=1, dim=0)
                # print(pred)
                # print(norm_pred)
                # time.sleep(10)
                confidence = (torch.max(pred, 1)[0] - torch.topk(pred, 2)[0][:, 1]).item()
                # print(pred)
                # print(confidence)
                # time.sleep(1)
                if confidence < threshold:
                    if wml.index(wm) == len(wml) - 1:
                        done = True
                    else:
                        wm = wml[wml.index(wm) + 1]
                        model.change_width_mult(wm)
                else:
                    done = True

            correct_list.append((pred.argmax(1) == y).type(torch.float).sum().item())
            confidence_list.append(confidence)
            times.append(time.time() - start_time)
            width_list.append(wm)

            if verbosity > 1 and (i == size or i % verb_mod == 0):
                bar_len = 100
                percent = f"{(i/size)*100:2.1f}"
                filled_len = int(float(percent) // 1)
                bar = "█" * filled_len + '-' * (bar_len - filled_len)
                if i == size:
                    print(f"\r[INFO] Completed {i:5}/{size} | {bar} | {percent}%")
                elif i % verb_mod == 0:
                    print(f"\r[INFO] Completed {i:5}/{size} | {bar} | {percent}%", end="")

    if verbosity > 1: print("[INFO] Done!")
    print(f"[INFO] Accuracy: {np.array(correct_list).sum() / len(correct_list)}")
    return correct_list, confidence_list, times, width_list
