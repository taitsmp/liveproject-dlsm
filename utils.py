from allennlp.models import Model

from allennlp.training.trainer import Trainer, GradientDescentTrainer

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding

from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import LSTM
from typing import Dict, Iterable, Union, Optional, List


@DatasetReader.register("char_lm_reader")
class CharDatasetReader(DatasetReader):
    def __init__(self) -> None:
        super().__init__(lazy=False)
        # todo: could become args
        self._token_indexers = {'tokens': SingleIdTokenIndexer()} #'tokens' is the namespace we're using
        self._tokenizer = CharacterTokenizer()
        
    def text_to_instance(self, sentence: str,) -> Instance:
        
        tokenized = self._tokenizer.tokenize(sentence)
        # TODO: do you want to add "source" and "target" here? 
        instance = Instance({"source": TextField(tokenized, self._token_indexers)})
        return instance
    
    def _read(self, csv_file: str) -> Iterable[Instance]:
        
        df = pd.read_csv(csv_file)
        df_titles = df[df.category == 'title']

        for row in df_titles.itertuples(index=False):
            instance = self.text_to_instance(row.text)
            yield instance


@Model.register('char_lm_model')
class CharLanguageModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,  # you pass in the model with layers here. LSTM, etc.
                 ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder

        num_labels = vocab.get_vocab_size("tokens") #get from the tokens namespace
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)

# how to get the correct vocab size? https://guide.allennlp.org/reading-data#3

    def forward(self,
                source: TextFieldTensors,
                **args
                ) -> Dict[str, torch.Tensor]:

        output: Dict[str, torch.Tensor] = {}
        mask = get_text_field_mask(source)  # mask of 0 or 1 for padding or token

        targets = self._target(source)

        # run NN layers forward
        embedded    = self.embedder(source)
        encoded     = self.encoder(embedded, mask)
        char_logits = self.classifier(encoded)

        # calculate the loss 
        output['loss'] = self._loss(char_logits, targets, mask)
        return output
   
    @staticmethod
    def _loss(char_logits: torch.Tensor,
              targets: torch.Tensor,
              mask: torch.BoolTensor
              ) -> torch.FloatTensor:
        print(char_logits.shape, targets.shape, mask.shape)
        loss = sequence_cross_entropy_with_logits(char_logits, targets, mask)
        return loss

    def _target(self,
                source: TextFieldTensors
                ) -> torch.Tensor:
         
        target = None
        token_id_dict = source.get("tokens")
        if token_id_dict is not None:
            token_ids = token_id_dict["tokens"]

            # Use token_ids to compute targets
            # last token id is set at zero?
            target = torch.zeros_like(token_ids)
            target[:, 0:-1] = token_ids[:, 1:]
        
        return target

# left off here. Review what forward looks like in sample AllenNLP language model.
# - do you need to use a mask?
#   + https://mlexplained.com/2019/01/30/an-in-depth-tutorial-to-allennlp-from-basics-to-elmo-and-bert/
#   + review get get_text_field_mask() function
# AllenNLP Language Model Implementation uses softmaxloss - https://docs.allennlp.org/master/api/modules/softmax_loss/
# this is where they get the targets from the tokens/sources - https://github.com/allenai/allennlp-models/blob/master/allennlp_models/lm/models/language_model.py#L265
