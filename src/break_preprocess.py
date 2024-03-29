import json
import os
import soundfile
import argparse
from utils import SplitWavAudioMubin, is_float, SUFFIX_LEN

DEBUG = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splitting .wav files')
    parser.add_argument('--bfilepath', type=str, default='/home/ubuntu/Wav2ToBI/data/break_files/', help='Folder of .break file')
    parser.add_argument('--wfilepath', type=str, default='/home/ubuntu/Wav2ToBI/data/wav_files/', help='Folder of .wav file')
    parser.add_argument('--sec_per_split', type=int, default=20, help='Duration of each split')
    parser.add_argument('--window_size', type=int, default=10, help='Window size')
    parser.add_argument('--output_path', type=str, default='/home/ubuntu/Wav2ToBI/data/output_json/test_break.json', help='Output path')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--peak', action='store_true')
    group.add_argument('--flat', action='store_true')
    args = parser.parse_args()

    total_tones = {}
    bfiles = os.listdir(args.bfilepath)
    wfiles = os.listdir(args.wfilepath)

    total_doc = []
    for i in range(len(bfiles)):
        bfilen = bfiles[i]
        wfile = bfilen[:-SUFFIX_LEN] + 'wav'
        if (wfile not in wfiles):
            continue
        wavfile = SplitWavAudioMubin(args.wfilepath, wfile).multiple_split(args.sec_per_split, args.window_size)
        data = open(args.bfilepath + bfilen,mode = 'rb')
        doc_count = 0
        line_count = 0
        total_b = []
        for line in data:
            line_count += 1
            line = line.strip()
            parts = line.split()
            if (len(parts) > 3):
                parts = parts[:3]
            elif (len(parts) < 3):
                continue
            break_time = parts[0].decode("utf-8")
            if (is_float(break_time) and break_time != "NaN"):
                parts[0] = float(break_time)
            else:
                continue
            parts[2] = parts[2].decode("utf-8")
            if (parts[2][0] not in ['1','2','3','4']):
                parts[2] = "0"
            else:
                parts[2] = parts[2][0]
            bfilen_mod = bfilen[:-SUFFIX_LEN] + 'wav'
            if (bfilen_mod in wfiles):
                parts.append(bfilen_mod)
            else:
                continue
            total_b.append(parts)
        total_doc.append(total_b)

    json_file = []
    doc_count = 0
    total_count = 0
    # NOTE: modify the wave path to the correct wave path
    wavPath = args.wfilepath + 'sliding/'+ str(doc_count) + '_' + total_doc[0][0][3]
    total_break = [wavPath]

    for num in range(len(total_doc)):
        doc_count = 0
        total_list = total_doc[num]
        if (total_list == []):
            continue
        wavPath = args.wfilepath + 'sliding/' + str(doc_count) + '_' + total_doc[num][0][3]
        if os.path.exists(wavPath):
            totalPath = args.wfilepath + total_doc[num][0][3]
            total_break = []
            total_time = int(soundfile.info(totalPath).frames/320)
            print("Total Time:", total_time)
            doc_name = total_doc[num][0][3]
            print("Document name:", wavPath)
            ## Time step start from 1, indicating that it is the end of the time segment
            # Current timestep
            step = 1
            # Current start time of sliding window
            cur_start = 1
            # Start break index of the next sliding window
            next_start_ind = 0
            # Current index of the break label
            cur_ind = 0
            while step < total_time:
                if (cur_ind < len(total_list)) and (abs(int(total_list[cur_ind][0]/0.02) - step) <= 10):
                    if (total_list[cur_ind][2][0] == '4'):
                        if args.peak:
                            total_break.append(1 - abs(int(total_list[cur_ind][0]/0.02) - step)/10)
                        elif args.flat:
                            total_break.append(1)
                    elif (total_list[cur_ind][2][0] == '3'):
                        if (abs(int(total_list[cur_ind][0]/0.02) - step) <= 5):
                            if args.peak:
                                total_break.append(0.5 - abs(int(total_list[cur_ind][0]/0.02) - step)/10)
                            elif args.flat:
                                total_break.append(0.5)
                        elif (step - int(total_list[cur_ind][0]/0.02) > 5):
                            cur_ind += 1
                            total_break.append(0)
                        else:
                            total_break.append(0)
                    else:
                        cur_ind += 1
                        total_break.append(0)
                    
                elif (cur_ind < len(total_list)) and (step - int(total_list[cur_ind][0]/0.02) > 10):
                        cur_ind += 1
                        total_break.append(0)
                    
                else:
                        total_break.append(0)

                if step == total_time - 1:
                    doc_count += 1
                    json_file.append({'path': wavPath, "label": total_break})
                    total_count += 1
                    break
                
                if (step - cur_start == 998):
                    doc_count += 1
                    json_file.append({'path': wavPath, "label": total_break})
                    wavPath = args.wfilepath + 'sliding/' + str(doc_count) + '_' + total_doc[num][0][3]
                    total_break = []
                    total_count +=1
                    step = step - 498
                    print("Step:", step)
                    cur_start = step
                    cur_ind = next_start_ind
                    continue

                # See if the time has past the sliding window
                if (next_start_ind < len(total_list)) and (int(total_list[next_start_ind][0]/0.02) - cur_start) <= 499:
                    next_start_ind += 1
                elif (next_start_ind >= len(total_list)):
                    next_start_ind = len(total_list)
                step += 1

    with open(args.output_path, 'w') as f:
        json.dump(json_file, f)

    # Plot the ground truth for debugging
    if DEBUG:
        from matplotlib import pyplot as plt
        plt.rcParams["figure.figsize"] = (20,3)
        plt.plot(json_file[0]["label"],label = "ground_truth", color = "blue")
        plt.xlim(0, 1000)
        plt.ylim(0, 1.2)
        plt.legend()
        plt.savefig('img/output_break.png')