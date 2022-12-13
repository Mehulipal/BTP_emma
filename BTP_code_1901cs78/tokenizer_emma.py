import os
import torch

from itertools import chain
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]", "[sad]", "[None]", "[Keywords]"]

class DatasetBase(Dataset):

    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data_files = list()
        self.data_files_offset = list()
        self.data_len = 0
        self._check_files()

    def _check_files(self):
        if self.data_path is None:
            raise RuntimeError("Data path cannot be \
                empty at same time.")

        if self.data_path:
            if not os.path.exists(self.data_path):
                raise RuntimeError("Training files does not exist at " + self.data_path)
            prepare_files_offset(self.data_path, self.data_files,
                                 self.data_files_offset)
            self.data_len = len(self.data_files_offset)

    def __len__(self):
        return self.data_len

    def _get_line(self, index):
        tup = self.data_files_offset[index]
        target_file = self.data_files[tup[0]]
        with open(target_file, "r", encoding="utf-8") as f:
            f.seek(tup[1])
            line = f.readline()
        return line

class EmmaDataset(DatasetBase):

    def __init__(self, tokenizer, max_history=50, batch_first=True, lm_labels=True, *inputs, **kwargs):
        super(EmmaDataset, self).__init__(*inputs, **kwargs)
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels

    # Reference: https://www.geeksforgeeks.org/__getitem__-in-python/
    def __getitem__(self, index):
        tokenizer = self.tokenizer
        dialog = self._get_line(index)
        #print("Index:",index) # why is index shuffled???
        #print("Old dialog:", dialog)
        dialog = dialog.replace('{', '')
        dialog = dialog.replace('}', '')
        dialog = dialog.replace(':', '')
        dialog = dialog.replace(',', '')
        dialog = dialog.replace('\"', '')
        dialog = dialog.strip().split()
        #print("New dialog:", dialog)

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
            
        query_text, query_label, query_keyword = dialog[2], dialog[4], dialog[8]
        if query_keyword == "[None]":
            hasKeyword = "[False]"
        else:
            hasKeyword = "[True]"
        query_text = query_text + " [SEP] " + query_label + " [SEP] " + hasKeyword + " [SEP] " + query_keyword + " [SEP]"
        query = tokenize(query_text.split("|")) # call tokenize() function
        response = tokenize(dialog[-1]) # call tokenize() function
        
        # [100, 1916, 1855, 100, 100, 100, 102, 6569, 102, 30524, 102, 100, 100, 102]
        '''
        print("100: ", tokenizer.convert_ids_to_tokens([100]))
        print("1916: ", tokenizer.convert_ids_to_tokens([1916]))
        print("1855: ", tokenizer.convert_ids_to_tokens([1855]))
        print("102: ", tokenizer.convert_ids_to_tokens([102]))
        print("6569: ", tokenizer.convert_ids_to_tokens([6569]))
        print("30524: ", tokenizer.convert_ids_to_tokens([30524]))
        '''
        
        '''
        print("query: ", query_text)
        print("query_ids:", query)
        print("response:", dialog[-1])
        print("response_ids:", response)
        print('\n')
        '''
        return self.process(query, response)

    def process(self, history, resposne, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:4])
        sequence = [[bos]] + history + [resposne + ([eos] if with_eos else [])]
        # print("seq1:", sequence)
        sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s
                                    for i, s in enumerate(sequence[1:])]
        # print("seq2:", sequence)
        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        # print("input_ids:", instance['input_ids'])
        instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=0.0) #padding_value=self.pad
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=0.0) #padding_value=self.pad
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, labels

def prepare_files_offset(path, files_list, offset_list):
    """Fill the file index and offsets of each line in files_list in offset_list
    Args:
        path: string of file path, support single file or file dir
        files_list: the list contains file names
        offset_list: the list contains the tuple of file name index and offset
    """
    if os.path.isdir(path):  # for multi-file, its input is a dir
        files_list.extend([os.path.join(path, f) for f in os.listdir(path)])
    elif os.path.isfile(path):  # for single file, its input is a file
        files_list.append(path)
    else:
        raise RuntimeError(path + " is not a normal file.")
    for i, f in enumerate(files_list):
        offset = 0
        with open(f, "r", encoding="utf-8") as single_file:
            for line in single_file:
                tup = (i, offset)
                offset_list.append(tup)
                offset += len(bytes(line, encoding='utf-8'))

