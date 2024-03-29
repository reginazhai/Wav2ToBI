import json
import os
import soundfile
import argparse
from utils import SplitWavAudioMubin, is_float, SUFFIX_LEN, PITCH_ACC, ORIG_PITCH_MAP

DEBUG = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splitting .wav files')
    parser.add_argument('--tfilepath', type=str, default='/home/ubuntu/Wav2ToBI/data/tone_files/', help='Folder of .tone file')
    parser.add_argument('--wfilepath', type=str, default='/home/ubuntu/Wav2ToBI/data/wav_files/', help='Folder of .wav file')
    parser.add_argument('--sec_per_split', type=int, default=20, help='Duration of each split')
    parser.add_argument('--window_size', type=int, default=10, help='Window size')
    parser.add_argument('--output_path', type=str, default='/home/ubuntu/Wav2ToBI/data/output_json/test_tone.json', help='Output path')
    parser.add_argument('--splitwav', action='store_true')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--peak', action='store_true')
    group.add_argument('--flat', action='store_true')
    args = parser.parse_args()

    total_tones = {}
    tfiles = os.listdir(args.tfilepath)
    wfiles = os.listdir(args.wfilepath)

    total_doc = []
    for tfilen in tfiles:
        wfile = tfilen[:-SUFFIX_LEN] + 'wav'
        if (wfile not in wfiles):
            continue
        if args.splitwav:
            wavfile = SplitWavAudioMubin(args.wfilepath, wfile).multiple_split(args.sec_per_split, args.window_size)
        data = open(args.tfilepath + tfilen,mode = 'rb')
        doc_count = 0
        line_count = 0
        total_t = []
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
            if (parts[2][-1] == ";") or (parts[2][-1] == "?"):
                parts[2] = parts[2][:len(parts[2])-1]

            if ('H' not in parts[2]) and ('L' not in parts[2]):
                parts[2] = "*"

            total_tones[parts[2]] = total_tones.get(parts[2], 0) + 1
            tfilen_mod = tfilen[:-SUFFIX_LEN] + 'wav'
            if (tfilen_mod in wfiles):
                parts.append(tfilen_mod)
            else:
                continue
            total_t.append(parts)
        total_doc.append(total_t)

    json_file = []
    doc_count = 0
    total_count = 0
    # NOTE: modify the wave path to the correct wave path
    wavPath = args.wfilepath + 'sliding/'+ str(doc_count) + '_' + total_doc[0][0][3]

    for num in range(len(total_doc)):
        doc_count = 0
        total_list = total_doc[num]
        if (total_list == []):
            continue
        wavPath = args.wfilepath + 'sliding/' + str(doc_count) + '_' + total_doc[num][0][3]
        if os.path.exists(wavPath):
            totalPath = args.wfilepath + total_doc[num][0][3]
            total_tone = []
            total_time = int(soundfile.info(totalPath).frames/320)
            print(total_time)
            doc_name = total_doc[num][0][3]
            cur_ind = 0
            print("Document name:", wavPath)
            ## Time step start from 1, indicating that it is the end of the time segment
            # Current timestep
            step = 1
            # Current start time of sliding window
            cur_start = 1
            # Start tone index of the next sliding window
            next_start_ind = 0
            # Current index of the tone label
            cur_ind = 0
            while step < total_time:
                if (cur_ind < len(total_list)) and (abs(int(total_list[cur_ind][0]/0.02) - step) <= 8):
                    cur_sym = total_list[cur_ind][2]
                    if (cur_sym in PITCH_ACC) or (cur_sym in ORIG_PITCH_MAP):
                        if args.peak:
                            total_tone.append(1 - abs(int(total_list[cur_ind][0]/0.02) - step)/8)
                        elif args.flat:
                            total_tone.append(1)
                        else:
                            raise ValueError("Please specify the preprocessing type")
                    else:
                        cur_ind += 1
                        total_tone.append(0)

                elif (cur_ind < len(total_list)) and (step - int(total_list[cur_ind][0]/0.02) > 8):
                        cur_ind += 1
                        total_tone.append(0)

                else:
                        total_tone.append(0)

                if step == total_time - 1:
                    doc_count += 1
                    json_file.append({'path': wavPath, "label": total_tone})
                    total_count += 1
                    break

                if (step - cur_start == 998):
                    doc_count += 1
                    json_file.append({'path': wavPath, "label": total_tone})
                    wavPath = args.wfilepath + 'sliding/' + str(doc_count) + '_' + total_doc[num][0][3]
                    total_tone = []
                    total_count +=1
                    total_count +=1
                    step = step - 498
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
        plt.savefig('img/output_tone.png')