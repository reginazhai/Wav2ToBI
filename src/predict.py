import argparse
import os

from eval import *
from model import *
from train import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prosody Prediction using Checkpoints')
    parser.add_argument('--model_checkpoint', type=str, default="facebook/wav2vec2-base", help='Path, url or short name of the model')
    parser.add_argument('--file_eval', type=str, default='/home/ubuntu/Wav2ToBI/data/output_json/test_break.json', help='Path to the evaluation (test) dataset (a JSON file)')
    parser.add_argument('--file_ind', type=int, default=0, help='Index of file to plot')
    parser.add_argument('--file_out', type=str, default='img/example_predict_plot', help='Name of output file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--max_duration', type=float, default=21.0, help='Maximum duration of audio files, default = 21s (must be >= duration of the longest file)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--peak', action='store_true')
    group.add_argument('--flat', action='store_true')
    args = parser.parse_args()
        
    dataset = load_json_dataset_forAudioFrameClassification(None,args.file_eval)
    model = Wav2Vec2ForAudioFrameClassification_custom.from_pretrained(args.model_checkpoint, num_labels=1)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base",
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    dataset = dataset.rename_column("path","audio")
    dataset = dataset.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

    def preprocess_function(examples):
        '''
        Preprocesses the dataset for the model.
        '''
        if examples is None:
            return None

        pitches = []
        audio_arrays = []
        for x in examples["audio"]:
          audio_arrays.append(x["array"])
          snd = parselmouth.Sound(values = x["array"], sampling_frequency = 16_000)
          pitch = snd.to_pitch()
          pitches.append(list(pitch.selected_array['frequency'])[::2])

        labels = examples["label"]

        sampling_rate = 16000
        labels_rate = 50 # labels per second

        num_padded_labels = round(args.max_duration * labels_rate)

        for label in labels:
            for _ in range(len(label), num_padded_labels):
                label.append(0)
            label.pop()

        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            padding='max_length', # pad to max_length, not just to the longest sequence
            max_length=int(sampling_rate * args.max_duration), 
            truncation=False,
        )

        inputs["label"] = labels
        inputs["pitch"] = pitches

        return inputs

    # Preprocessing the dataset
    print("Processing test data...")
    processed_dataset_test = dataset["eval"].map(preprocess_function, remove_columns=["audio","label"], batched=True)

    processed_dataset_test = processed_dataset_test.rename_column("label", "labels")
    processed_dataset_test.set_format("torch", columns=["input_values", "pitch", "labels"])
    eval_dataloader = DataLoader(processed_dataset_test, batch_size=1)

    print("Starting evaluation...")
    model.eval()
    predictions_all = []

    progress_bar = tqdm(range(len(eval_dataloader)))
    for _, batch in enumerate(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits

        if model.num_labels == 1:
            predictions = logits
        else:
            predictions = torch.argmax(logits, dim=-1)

        labels = batch["labels"].reshape(-1)
        predictions = predictions.reshape(-1)
        predictions_all.append(predictions.cpu().detach().numpy())
        progress_bar.update(1)
    progress_bar.close()

    with open('./tmp.txt', 'w') as file:
        for ii,prediction in enumerate(predictions_all):
            file.write(dataset["eval"][ii]["audio"]["path"])
            file.write(",[")
            prediction.tofile(file,sep=" ", format="%s")
            file.write("]\n")
    
    # Load ground truth labels
    f_eval = open(args.file_eval)
    content_eval = json.load(f_eval)

    content_test = load_test('./tmp.txt')

    # Example Plotting
    plt.rcParams["figure.figsize"] = (20,3)
    plt.plot(content_test[args.file_ind][1],'r--', label = "prediction")
    plt.plot(content_eval[args.file_ind]["label"],label = "ground_truth", color = "blue")
    plt.xlim(0, 1000)
    plt.ylim(0, 1.2)
    plt.legend(loc = 'upper right', fontsize = 'large')
    plt.savefig(args.file_out + '.png', dpi=300, bbox_inches='tight')

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
    os.remove('./tmp.txt')