import json
import argparse
from matplotlib import pyplot as plt

## Find peaks in prediction
def find_peaks_output(content_output):
  total_peaks = []
  for i in range(len(content_output)):
    cur_peak = []
    cur_pred = content_output[i][1]
    for j in range(5, len(cur_pred)-5):
      cur_window = cur_pred[max(0, j-5):min(j+5, len(cur_pred))]
      ## Found a valid peak
      if (max(cur_window) > 0.75) and (max(cur_window) == cur_pred[j]):
        cur_time = 0.02*(j+1)
        cur_peak.append((cur_time, '4'))
    total_peaks.append(cur_peak)
  return total_peaks

def find_peaks_eval(content_eval):
  total_peaks = []
  for i in range(len(content_eval)):
    cur_peak = []
    cur_pred = content_eval[i]["label"]
    for j in range(5, len(cur_pred)-5):
      cur_window = cur_pred[max(0, j-5):min(j+5, len(cur_pred))]
      ## Found a valid peak
      if (max(cur_window) > 0.75) and (max(cur_window) == cur_pred[j]):
        cur_time = 0.02*(j+1)
        cur_peak.append((cur_time, '4'))
    total_peaks.append(cur_peak)
  return total_peaks

def compare_tup_bi(act,pred,threshold = 0.1):
    pred_ind = 0
    act_ind = 0
    correct_c = 0
    while act_ind != len(act):
      if (pred_ind == len(pred)):
        break
      ts = act[act_ind][0]
      if ((pred[pred_ind][0] >= ts-threshold) and (pred[pred_ind][0] <= ts+threshold)):
        correct_c += 1
        pred_ind += 1
        act_ind +=1
      elif (pred[pred_ind][0] < ts-threshold):
        pred_ind += 1
      elif (pred[pred_ind][0] > ts+threshold):
        act_ind += 1
      else:
        pred_ind += 1
        act_ind +=1

    recall = correct_c/len(act)
    precision = correct_c/len(pred)
    if precision == 0 and recall == 0:
      f1 = 0
    else:
      f1 = 2*(precision*recall)/(precision+recall)
    return precision, recall, f1

## Find flats in prediction
def find_flat_output(content_output):
  total_flat = []
  for i in range(len(content_output)):
    prev_pivot = 0
    cur_pivot = 1
    cur_pred = content_output[i][1]
    cur_flat = []
    while prev_pivot < (len(cur_pred)-1):
      if (cur_pivot <= (len(cur_pred)-1)) and (cur_pred[cur_pivot] - cur_pred[prev_pivot]) >= 0.2:
        cur_pivot += 1
        continue
      elif abs(prev_pivot - cur_pivot) >= 10:
        mid_time = 0.02*(prev_pivot + cur_pivot)/2
        max_val = max(cur_pred[prev_pivot:cur_pivot])
        if max_val > 0.75:
          cur_flat.append((mid_time, '4'))
        prev_pivot = cur_pivot
        cur_pivot += 1
      else:
        prev_pivot += 1
        cur_pivot += 1
    total_flat.append(cur_flat)
  return total_flat

def find_flat_eval(content_eval):
  total_flat = []
  for i in range(len(content_eval)):
    prev_pivot = 0
    cur_pivot = 1
    cur_pred = content_eval[i]["label"]
    cur_flat = []
    while prev_pivot < (len(cur_pred)-1):
      if (cur_pivot <= (len(cur_pred)-1)) and (cur_pred[cur_pivot] - cur_pred[prev_pivot]) >= 0.2:
        cur_pivot += 1
        continue
      elif abs(prev_pivot - cur_pivot) >= 10:
        mid_time = 0.02*(prev_pivot + cur_pivot)/2
        max_val = max(cur_pred[prev_pivot:cur_pivot])
        if max_val > 0.75:
          cur_flat.append((mid_time, '4'))
        prev_pivot = cur_pivot
        cur_pivot += 1
      else:
        prev_pivot += 1
        cur_pivot += 1
    total_flat.append(cur_flat)
  return total_flat

def evaluation_step(total_peaks, ground_peaks, tolerance):
    total_prec = 0
    total_rec = 0
    total_f1 = 0
    tolerance = tolerance
    total_len = 0
    assert len(total_peaks) == len(ground_peaks)
    for i in range(len(total_peaks)):
        if (len(ground_peaks[i]) == 0):
            continue
        if (len(total_peaks[i]) == 0):
            total_len += 1
            continue
        #Compare function
        prec, rec, f1 = compare_tup_bi(ground_peaks[i],total_peaks[i],tolerance)
        total_prec += prec
        total_rec += rec
        total_f1 += f1
        total_len += 1
    print('------------------------------')
    print('Average Precision when tolerance =', tolerance ,'s :', "{:.2f}".format(total_prec/total_len*100))
    print('Average Recall when tolerance =', tolerance ,'s :   ', "{:.2f}".format(total_rec/total_len*100))
    print('Average F-1 when tolerance =', tolerance ,'s :      ', "{:.2f}".format(total_f1/total_len*100))

def eval_peaks(total_peaks, ground_peaks):
    # Evaluation on the four tolerance
    evaluation_step(total_peaks, ground_peaks, 0.00)
    evaluation_step(total_peaks, ground_peaks, 0.04)
    evaluation_step(total_peaks, ground_peaks, 0.08)
    evaluation_step(total_peaks, ground_peaks, 0.1)

def eval_flats(total_flat, ground_flat):
    # Evaluation on the four tolerance
    evaluation_step(total_flat, ground_flat, 0.00)
    evaluation_step(total_flat, ground_flat, 0.04)
    evaluation_step(total_flat, ground_flat, 0.08)
    evaluation_step(total_flat, ground_flat, 0.1)


def load_test(file_test):
   # Load predicted labels
    with open(file_test) as f_test:
        lines = f_test.readlines()

    content_test = [x.split(',') for x in lines]
    for i in range(len(content_test)):
        content_test[i][1] = content_test[i][1].strip("[]\n").split()
        content_test[i][1] = [float(content_test[i][1][j]) for j in range(len(content_test[i][1]))]
    return content_test
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating test performance')
    parser.add_argument('--file_eval', type=str, default='/home/ubuntu/Wav2ToBI/data/output_json/test_tone.json', help='Path to ground truth labels')
    parser.add_argument('--file_test', type=str, default='/home/ubuntu/Wav2ToBI/data/output_json/test_output_tone.txt', help='Path to predicted labels')
    parser.add_argument('--file_ind', type=int, default=1, help='Index of file to plot')
    parser.add_argument('--plot_out', type=str, default='img/example_eval_plot', help='Name of output plot')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--peak', action='store_true')
    group.add_argument('--flat', action='store_true')
    args = parser.parse_args()

    # Load ground truth labels
    f_eval = open(args.file_eval)
    content_eval = json.load(f_eval)

    content_test = load_test(args.file_test)
    
    # Example Plotting
    plt.rcParams["figure.figsize"] = (20,3)
    plt.plot(content_test[args.file_ind][1],'r--', label = "prediction")
    plt.plot(content_eval[args.file_ind]["label"],label = "ground_truth", color = "blue")
    plt.xlim(0, 1000)
    plt.ylim(0, 1.2)
    plt.legend(loc = 'upper right', fontsize = 'large')
    plt.savefig(args.plot_out + '.png', dpi=300, bbox_inches='tight')

    if args.peak:
        # Get peaks in prediction and ground truth
        total_peaks = find_peaks_output(content_test)
        ground_peaks = find_peaks_eval(content_eval)
        eval_peaks(total_peaks, ground_peaks)
        with open('data/output_json/eval_results.json', 'w') as f:
            json.dump({'total_peaks': total_peaks, 'ground_peaks': ground_peaks}, f)
    elif args.flat:
        # Get flats in prediction and ground truth
        total_flat = find_flat_output(content_test)
        ground_flat = find_flat_eval(content_eval)
        eval_flats(total_flat, ground_flat)
        with open('data/output_json/eval_results.json', 'w') as f:
            json.dump({'total_flats': total_flat, 'ground_flats': ground_flat}, f)

