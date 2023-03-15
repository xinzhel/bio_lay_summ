from pathlib import Path
from typing import Dict, Optional, List
import logging
import os
import json


from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)


@DatasetReader.register("biosumm")
class BioSummDatasetReader(DatasetReader):

    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        source_max_tokens: Optional[int] = None,
        target_max_tokens: Optional[int] = None,
        source_prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._source_tokenizer = source_tokenizer or SpacyTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_prefix = source_prefix

    def _read(self, file_path: str):

        with open(file_path, 'r') as f:
            for line in f.readlines():
                text = json.loads(line)
                yield self.text_to_instance(text['article'], text['lay_summary'])
              

    def text_to_instance(
        self, source_sequence: str, target_sequence: str = None
    ) -> Instance:  # type: ignore
        if self._source_prefix is not None:
            tokenized_source = self._source_tokenizer.tokenize(
                self._source_prefix + source_sequence
            )
        else:
            tokenized_source = self._source_tokenizer.tokenize(source_sequence)
        if self._source_max_tokens is not None and len(tokenized_source) > self._source_max_tokens:
            tokenized_source = tokenized_source[: self._source_max_tokens]

        source_field = TextField(tokenized_source)
        if target_sequence is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_sequence)
            if (
                self._target_max_tokens is not None
                and len(tokenized_target) > self._target_max_tokens
            ):
                tokenized_target = tokenized_target[: self._target_max_tokens]
            target_field = TextField(tokenized_target)
            return Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:
            return Instance({"source_tokens": source_field})

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["source_tokens"]._token_indexers = self._source_token_indexers  # type: ignore
        if "target_tokens" in instance.fields:
            instance.fields["target_tokens"]._token_indexers = self._target_token_indexers  # type: ignore