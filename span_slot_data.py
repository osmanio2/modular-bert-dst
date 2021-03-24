import os
import json
import random
import torch as T
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple


def extract_span_values(frames):
    values = []
    for frame in frames:
        service = frame['service']
        for slot_span in frame['slots']:
            slot = slot_span['slot']
            strt, end = slot_span['start'], slot_span['exclusive_end']
            values.append((f'{service}--{slot}', strt, end))
    return values


def extract_slot_values(frames):
    slot_values = {}
    for frame in frames:
        service = frame['service']
        for slot, values in frame['state']['slot_values'].items():
            slot_values[f'{service}--{slot}'] = set(values)
    return slot_values


def filter_span_states(span_slots, states, sys, usr):
    new_states = {}
    for slot, strt, end in span_slots:
        if slot in states:
            if usr[strt:end] in states[slot]:
                new_states[slot] = usr[strt:end]
            elif sys[strt:end] in states[slot]:
                new_states[slot] = sys[strt:end]
    return new_states


def extract_labels(dialog):
    results = []
    system = "Hello, how can I help?"
    system_values = []
    dialog_id = dialog['dialogue_id']
    for i, turn in enumerate(dialog['turns']):
        if turn['speaker'] == 'USER':
            result = {'id': f'{dialog_id}-{i}'}
            result['user'] = turn['utterance']
            result['system'] = system
            values = extract_span_values(turn['frames'])
            states = extract_slot_values(turn['frames'])
            span_slots = values + system_values
            result['labels'] = filter_span_states(span_slots, states, system, result['user'])
            results.append(result)
        elif turn['speaker'] == 'SYSTEM':
            system = turn['utterance']
            system_values = extract_span_values(turn['frames'])
    return results


def bert_span_encode(self, serv_desc, slot_desc, span, user, system,
                     none, max_length=80, return_tensors=False):
    question = 'Service description: ' + serv_desc + ' [SEP] Slot description: ' + slot_desc
    answer = 'System: ' + system + ' [SEP] User: ' + user
    enc = self.encode_plus(question, answer, max_length=max_length, add_special_tokens=True,
                            pad_to_max_length=True, truncation_strategy='only_first')
    if none:
        enc['start_positions'], enc['end_positions'] = 0, 0
    else:
        tokenized_txt = self.tokenize(answer)
        tokenized_span = self.tokenize(span)
        pre_strt = tokenized_txt.index(tokenized_span[0])
        pre_end = tokenized_txt.index(tokenized_span[-1])
        strt = enc['token_type_ids'].index(1) + pre_strt
        end = enc['token_type_ids'].index(1) + pre_end
        enc['start_positions'], enc['end_positions'] = strt, end
    if return_tensors:
        return {k: T.tensor(v) for k, v in enc.items()}
    return enc


class SlotDataset(Dataset):

    BATCH = namedtuple('Batch', ['x', 'id', 'slot'])
    
    def __init__(self, data_url, tokenizer, max_len=50, random_seed=42):
        random.seed(random_seed)
        with open(os.path.join(data_url, 'schema.json'), 'r') as f:
            self.schema = self._extract_non_categorical_schema(json.load(f))
        with open(os.path.join(data_url, 'dialogues.json'), 'r') as f:
            self.data = [result for dialog in json.load(f) for result in extract_labels(dialog)
                         if len(result['labels']) > 0]

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def _extract_non_categorical_schema(schema):
        services = {}
        for service in schema:
            service_name = service['service_name']
            service_desc = service['description']
            for slot in service['slots']:
                if not slot['is_categorical']:
                    slot_name = slot['name']
                    slot_desc = slot['description']    
                    services[f'{service_name}--{slot_name}'] = [service_desc, slot_desc]
        return services

    def _sample_non_categorical_slots(self, labels, test=False):
        labels = [(serv_slot, span) for serv_slot, span in labels.items()]
        if test:
            return labels
        random.shuffle(labels)
        return labels[0]
    
    def _convert_to_batch(self, example, serv_slot, span):
        serv_desc, slot_desc = self.schema[serv_slot]
        enc = self.tokenizer.span_encode(serv_desc, slot_desc, span, example['user'],
                                             example['system'], False, self.max_len, True) 

        slot_value = f'{serv_slot}--{span}'
        return self.BATCH(x=enc, id=example['id'], slot=slot_value)

    def __getitem__(self, idx):
        example = self.data[idx]
        serv_slot, span = self._sample_non_categorical_slots(example['labels'], test=False)
        return self._convert_to_batch(example, serv_slot, span)

    def load_valid_data(self, batch_size):
        batches = []
        batch = []
        keys = ['x', 'id', 'slot']
        for datum in self.data:
            for serv_slot, span in self._sample_non_categorical_slots(datum['labels'], test=True):
                batch.append(self._convert_to_batch(datum, serv_slot, span))
                if len(batch) == batch_size:
                    batch = T.utils.data._utils.collate.default_collate(batch)
                    batches.append(self.BATCH(**dict(zip(keys, batch))))
                    batch = []
        if len(batch) > 0:
            batch = T.utils.data._utils.collate.default_collate(batch)
            batches.append(self.BATCH(**dict(zip(keys, batch))))
        return batches

    @classmethod
    def map_to_cuda(cls, batch):
        if isinstance(batch, T.Tensor):
            return batch.cuda()
        elif isinstance(batch, tuple) and hasattr(batch, '_fields'):
            return type(batch)(*[cls.map_to_cuda(e) for e in batch])
        elif isinstance(batch, dict):
            return {k: cls.map_to_cuda(v) for k, v in batch.items()}
        return batch
    
    def dataLoader(self, batch_size=1, shuffle=False, pin_memory=False, loop=False, cuda=False):
        while True:
            loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                                pin_memory=pin_memory)
            for batch in loader:
                yield (self.map_to_cuda(batch) if cuda else batch)
            
            if not loop:
                break