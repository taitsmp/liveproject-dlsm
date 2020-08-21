from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance

import pandas as pd


from torch.nn import LSTM

from typing import Dict, Iterable, Union, Optional, List

@DatasetReader.register("char_lm_reader")
class CharDatasetReader(DatasetReader):
    def __init__(self) -> None:
        super().__init__(lazy=False)
        #todo: could become args
        self._token_indexers = {'tokens': SingleIdTokenIndexer()}
        self._tokenizer = CharacterTokenizer()
        

    def text_to_instance(self, sentence: str,) -> Instance:
        
        tokenized = self._tokenizer.tokenize(sentence)
        #TODO: do you want to add "source" and "target" here? 
        instance = Instance({"source": TextField(tokenized, self._token_indexers)})
        return instance
    
    def _read(self, csv_file: str) -> Iterable[Instance]:
        
        df = pd.read_csv(csv_file)
        df_titles = df[df.category == 'title']

        for row in df_titles.itertuples(index=False):
            instance = self.text_to_instance(row.text)
            yield instance

@Model.register('char_lm_model')
class CharLanguageModel(Model)
  def __init__(self,
          vocab: Vocabulary,
          embedder: TextFieldEmbedder,
          encoder: Seq2SeqEncoder, #you pass in the model with layers here. LSTM, etc.
          )
    super().__init__(vocab)
    self.embedder = embedder
    self.encoder = encoder

    num_labels = vocab.get_vocab_size("source")
    self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)

# how to get the correct vocab size? https://guide.allennlp.org/reading-data#3


    def forward(self,
                tokens:   Dict[str, torch.Tensor],
                next_tokens: Optional[torch.Tensor] = None,
                **args) -> Dict[str, torch.Tensor]:
            pass

#left off here. Review what forward looks like in sample AllenNLP language model.
# AllenNLP Language Model Implementation uses softmaxloss - https://docs.allennlp.org/master/api/modules/softmax_loss/
# this is where they get the targets from the tokens/sources - https://github.com/allenai/allennlp-models/blob/master/allennlp_models/lm/models/language_model.py#L265
