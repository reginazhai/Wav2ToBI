import datasets
import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import MSELoss
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import Wav2Vec2Model, Wav2Vec2ForAudioFrameClassification

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# TODO: Add hidden layers as inputs to config

class Wav2Vec2ForAudioFrameClassification_custom(Wav2Vec2ForAudioFrameClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.lstm_hidden_size = config.lstm_hidden_size
        self.wav2vec2 = Wav2Vec2Model(config)
        self.lstm = nn.LSTM(config.hidden_size + 1, self.lstm_hidden_size, batch_first=True, bidirectional=True, num_layers=1)
        self.classifier = nn.Linear(self.lstm_hidden_size*2, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_values,
        pitch = None, 
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if pitch != None:
          pitch = pitch[0]
          if (len(pitch) < hidden_states.shape[1]):
            pitch = F.pad(pitch, pad=(0, hidden_states.shape[1] - len(pitch)), mode='constant', value=0)
          elif (len(pitch) > hidden_states.shape[1]):
            pitch = pitch[:hidden_states.shape[1]]
          pitch = pitch.reshape(1, len(pitch),1)

        tup = (pitch.to(device, dtype=torch.float),hidden_states)
        x_ = torch.cat(tup,2)
        x_, (_, _) = self.lstm(x_)
        logits = self.classifier(x_)
        labels = labels.reshape(-1,1) # 1xN -> Nx1

        loss = None
        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_json_dataset_forAudioFrameClassification(file_train,file_eval,file_valid=None):

    data_files = {}
    if file_train is not None:
        data_files["train"] = file_train
    if file_eval is not None:
        data_files["eval"] = file_eval
    if file_valid is not None:
        data_files["valid"] = file_valid

    phrasing_features = datasets.Features({
       'path': datasets.features.Value('string'), 
       'label': datasets.features.Sequence(datasets.features.Value(dtype='float64'))
       })
    
    dataset = datasets.load_dataset("json", data_files=data_files, features=phrasing_features)
    return dataset

