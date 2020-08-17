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

@DatasetReader.register("char_lang_mod")
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
    
    def _read(self, df: pd.DataFrame) -> Iterable[Instance]:
        
        for row in df.itertuples(index=False):
            instance = self.text_to_instance(row.text)
            yield instance


