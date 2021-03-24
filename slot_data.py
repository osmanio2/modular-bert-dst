import os
import json
import random
import torch as T
import numpy as np
import slot_class_data as D
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, namedtuple

def process_year(value):
    year = int(value)
    rel = f'{2019-year} years ago'
    if year == 2019:
        rel = f'this year'
    return f'{year} or {rel} or {year - 2000}'

def split_service_slot(label):
    service, slot, value = label.split('--')
    return (f'{service}--{slot}', value)

def extract_labels(dialog):
    results = D.extract_labels(dialog)
    for result in results:
        states = dict([split_service_slot(s) for s in result['states']])
        labels = set()
        for label in result['labels']:
            serv_slot, value = split_service_slot(label)
            if value == 'transferred':
                continue
            value = states[serv_slot]
            if serv_slot.split('--')[1] == 'year':
                value = process_year(value)
            labels.add((serv_slot, value))
        result['labels'] = labels
    return results


class SlotDataset(Dataset):

    BATCH = namedtuple('Batch', ['x', 'y', 'id', 'slot'])
    SEQUENCE = namedtuple('Sequence', ['input_ids', 'attention_mask', 'token_type_ids'])
    
    def __init__(self, data_url, tokenizer, max_len=50, random_seed=42):
        random.seed(random_seed)
        with open(os.path.join(data_url, 'schema.json'), 'r') as f:
            self.schema = self._extract_categorical_schema(json.load(f))
        with open(os.path.join(data_url, 'dialogues.json'), 'r') as f:
            self.data = [result for dialog in json.load(f) for result in extract_labels(dialog)
                        if any([r[0] in self.schema and r[1] != 'dontcare'
                                for r in result['labels']])]
        self.num_class = max([len(v[-1]) for v in self.schema.values()])
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def _extract_categorical_schema(schema):
        services = {}
        for service in schema:
            service_name = service['service_name']
            service_desc = service['description']
            for slot in service['slots']:
                if slot['is_categorical']:
                    slot_name = slot['name']
                    slot_desc = slot['description']
                    values = {}
                    for value in slot['possible_values']:
                        if slot_name == 'year':
                            value = process_year(value)
                        values[value] = value#value.replace('$', '')
                    services[f'{service_name}--{slot_name}'] = [service_desc, slot_desc, values]
        return services
    
    def _sample_categorical_slots(self, labels, test=False):
        labels = [label for label in labels if label[0] in self.schema and label[1] != 'dontcare']
        if test:
            return sorted(labels, key=lambda x: x[0])
        random.shuffle(labels)
        return labels[0]
    
    def _convert_to_example(self, example, service_slot, value):
        service_desc, slot_desc, values = self.schema[service_slot]
        service_desc = 'Service description: ' + service_desc
        slot_desc = 'Slot description: ' + slot_desc
        system = example['system']
        if system == '':
            system = 'Hello, how can I help?'
        context = 'System: ' + system + ' [SEP] User: ' + example['user'] + ' [SEP] ' \
                                        + service_desc + ' [SEP] ' + slot_desc
        id2value = list(values.keys())
        value2id = {k: i for i, k in enumerate(id2value)}
        choices = [values[k] for k in id2value]
        label = value2id[value]
        return {'context': context, 'choices': choices, 'label': label, 'id2value': id2value}
    
    def _convert_to_feature(self, context, choices):
        encoded_sequence = [self.tokenizer.encode_plus(context, choice, add_special_tokens=True,
                                                max_length=self.max_len, pad_to_max_length=True)
                            for choice in choices]
        keys = list(encoded_sequence[0].keys())
        feature = dict()
        for key in keys:
            feature[key] = np.pad([seq[key] for seq in encoded_sequence],
                                ((0, self.num_class - len(encoded_sequence)), (0, 0)), 'constant')
        return self.SEQUENCE(**feature)

    def _convert_to_batch(self, datum, service_slot, value):
        example = self._convert_to_example(datum, service_slot, value)
        feature = self._convert_to_feature(example['context'], example['choices'])
        value = '--'.join(example['id2value'])
        slot_value = f'{service_slot}--{value}--{example["label"]}'
        return self.BATCH(x=feature, y=example['label'], id=datum['id'], slot=slot_value)
    
    def __getitem__(self, idx):
        datum = self.data[idx]
        service_slot, value = self._sample_categorical_slots(datum['labels'])
        return self._convert_to_batch(datum, service_slot, value)

    def load_valid_data(self, batch_size):
        batches = []
        batch = []
        keys = ['x', 'y', 'id', 'slot']
        for datum in self.data:
            for service_slot, value in self._sample_categorical_slots(datum['labels'], test=True):
                batch.append(self._convert_to_batch(datum, service_slot, value))
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
        return batch
    
    def dataLoader(self, batch_size=1, shuffle=False, pin_memory=False, loop=False, cuda=False,
                   drop_last=False):
        while True:
            loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                                pin_memory=pin_memory, drop_last=drop_last)
            for batch in loader:
                yield (self.map_to_cuda(batch) if cuda else batch)
            
            if not loop:
                break