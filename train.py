import os
import parselmouth

from datasets import load_metric

from transformers import AutoFeatureExtractor
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

import argparse
import warnings
from model import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model for audio frame classification')
    parser.add_argument('--model_checkpoint', type=str, default="facebook/wav2vec2-base", help='Path, url or short name of the model')
    parser.add_argument('--file_train', type=str, default='/content/drive/MyDrive/BUR/tone_json_files/train_json_sliding_pitch_detection.json', help='Path to the training dataset (a JSON file)')
    parser.add_argument('--file_valid', type=str, default='/content/drive/MyDrive/BUR/tone_json_files/valid_json_sliding_pitch_detection.json', help='Path to the validation dataset (a JSON file)')
    parser.add_argument('--file_eval', type=str, default='/content/drive/MyDrive/BUR/tone_json_files/valid_json_sliding_pitch_detection.json', help='Path to the evaluation (test) dataset (a JSON file)')
    parser.add_argument('--lstm_hidden_size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--file_output', type=str, default='/content/drive/MyDrive/BUR/tone_json_output_files/valid_1layer_10_sliding_256_pitch_detection.txt', help='Path for the output file (output.txt)')
    parser.add_argument('--model_save_dir', type=str, default='/content/drive/MyDrive/BUR/valid_tone_1layer_10_sliding_256_pitch_det_flat', help='Directory for saving the training log and the finetuned model')
    parser.add_argument('--max_duration', type=float, default=21.0, help='Maximum duration of audio files, default = 21s (must be >= duration of the longest file)')
    parser.add_argument('--mode', type=str, default="both", help='Mode: "train", "eval" or "both" (default is "both")')
    parser.add_argument('--epochs_between_checkpoints', type=int, default=1, help='Number of epochs between saved checkpoints during training. Default is 1 - saving every epoch.')
    parser.add_argument('--lr_init', type=float, default=5e-5, help='Initial learning rate')
    parser.add_argument('--lr_num_warmup_steps', type=int, default=0, help='Number of warmup steps for the learning rate scheduler')
    parser.add_argument('--remove_last_label', type=int, default=1, help='Remove the last value from ref. labels to match the number of predictions? 0 = no, 1 = yes (Default: yes)')

    args = parser.parse_args()

    if args.remove_last_label > 0:
        remove_extra_label = True # in a 20.0 s audio, there will be 1000 labels but only 999 logits -> remove the last label so the numbers match
    else: # if the labels are already fixed elsewhere
        remove_extra_label = False

    do_train = args.mode in ['train','both']
    do_eval = args.mode in ['eval','both']

    if args.epochs_between_checkpoints < 0:
        raise ValueError("''--epochs_between_checkpoints'' must be >= 0")

    if do_train:
        if args.file_train is None:
            raise ValueError("Training requires path to the training dataset (argument '--file_train <path>'). "
                             "To disable training and only run evaluation using the existing model, use '--mode 'eval''")
        if args.num_epochs is None:
            raise ValueError("For training the model, the number of epochs must be specified (argument '--num_epochs <number>'). "
                             "To disable training and only run evaluation using the existing model, use '--mode 'eval''")
        if args.model_save_dir is None:
            warnings.warn("argument ''--model_save_dir'' is not set -> the finetuned model will NOT be saved.")
            if args.epochs_between_checkpoints > 0:
                print("Checkpoints during training will also NOT be saved.")

        if args.file_valid is None:
            print("There is no validation set. Loss will be calculated only on the training set.")
            do_validation = False
        else:
            do_validation = True
    else: 
        do_validation = False

    if do_eval:
        if args.file_eval is None:
            raise ValueError("Evaluation requires path to the evaluation dataset (argument '--file_eval <path>'). "
                             "To disable evaluation and only perform training, use '--mode 'train''")

    if args.model_save_dir is None or args.epochs_between_checkpoints == 0:
        save_checkpoints = False
    else:
        save_checkpoints = True

    freeze_feature_encoder = True
    freeze_base_model = False

    metric = load_metric("mse")
    dataset = load_json_dataset_forAudioFrameClassification(args.file_train,args.file_eval,args.file_valid)
    model = Wav2Vec2ForAudioFrameClassification_custom.from_pretrained(args.model_checkpoint, num_labels=1, lstm_hidden_size = args.lstm_hidden_size)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base",
    )

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
            if remove_extra_label:
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

    if freeze_feature_encoder:
        model.freeze_feature_encoder()

    if freeze_base_model:
        model.freeze_base_model()


    # -------------
    # process the train/val/test data
    # -------------
    if do_train:
        print("Processing training data...")
        processed_dataset_train = dataset["train"].map(preprocess_function, remove_columns=["audio","label"], batched=True)

        processed_dataset_train = processed_dataset_train.rename_column("label", "labels")
        processed_dataset_train.set_format("torch", columns=["input_values", "pitch", "labels"])
        train_dataloader = DataLoader(processed_dataset_train, shuffle=True, batch_size=args.batch_size)

    else:
        processed_dataset_train = None

    if do_validation:
        print("Processing validation data...")
        processed_dataset_valid = dataset["valid"].map(preprocess_function, remove_columns=["audio","label"], batched=True)

        processed_dataset_valid = processed_dataset_valid.rename_column("label", "labels")
        processed_dataset_valid.set_format("torch", columns=["input_values", "pitch", "labels"])
        valid_dataloader = DataLoader(processed_dataset_valid, shuffle=False, batch_size=args.batch_size)
    else:
        processed_dataset_valid = None

    if do_eval:
        print("Processing test data...")
        processed_dataset_test = dataset["eval"].map(preprocess_function, remove_columns=["audio","label"], batched=True)

        processed_dataset_test = processed_dataset_test.rename_column("label", "labels")
        processed_dataset_test.set_format("torch", columns=["input_values", "pitch", "labels"])
        eval_dataloader = DataLoader(processed_dataset_test, batch_size=1)
    else:
        processed_dataset_test = None

    # ----------
    # Training
    # ----------
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    if do_train:

        print("Starting training...")

        optimizer = AdamW(model.parameters(), lr=args.lr_init)

        num_training_steps = args.num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=args.lr_num_warmup_steps, num_training_steps=num_training_steps
        )
        
        progress_bar = tqdm(range(num_training_steps))
        if args.model_save_dir is not None:
            if args.model_save_dir != "":
                os.makedirs(args.model_save_dir, exist_ok=True)

            logfile = open(os.path.join(args.model_save_dir,"log.csv"), "w")
        else:
            out_dir = os.path.dirname(args.file_output)
            if out_dir != "":
                os.makedirs(out_dir, exist_ok=True)
            logfile = open(args.file_output + ".log.csv", "w")

        logfile.write("epoch,train loss,val loss\n")
        
        for epoch in range(args.num_epochs):
            model.train()
            # save checkpoint every `epochs_between_checkpoints` epochs
            if save_checkpoints and epoch > 0 and (epoch % args.epochs_between_checkpoints == 0):
                epoch_dir = os.path.join(args.model_save_dir,"epoch%d"%epoch)
                os.makedirs(epoch_dir, exist_ok=True)
                model.save_pretrained(epoch_dir)

            train_loss = 0
            for _, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                train_loss = train_loss + loss.detach().item()

            if do_validation:
                model.eval()
                val_loss = 0

                for _, batch in enumerate(valid_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)
                    loss = outputs.loss
                    val_loss = val_loss + loss.detach().item()

                logfile.write("%d,%f,%f\n"%(epoch,train_loss,val_loss))
                logfile.flush()

            else:
                logfile.write("%d,%f,N/A\n"%(epoch,train_loss))
                logfile.flush()

        logfile.close()
        progress_bar.close()

        if args.model_save_dir is not None:
            model.save_pretrained(args.model_save_dir)

    if do_eval:
        print("Starting evaluation...")

        out_dir = os.path.dirname(args.file_output)
        if out_dir != "":
            os.makedirs(out_dir, exist_ok=True)

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
            metric.add_batch(predictions=predictions, references=labels)
            progress_bar.update(1)

        progress_bar.close()

        with open(args.file_output, 'w') as file:
            for ii,prediction in enumerate(predictions_all):
                file.write(dataset["eval"][ii]["audio"]["path"])
                file.write(",[")
                prediction.tofile(file,sep=" ", format="%s")
                file.write("]\n")

        