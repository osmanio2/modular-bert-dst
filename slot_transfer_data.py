import os
import json
import random
import torch as T
import numpy as np
import slot_class_data as D
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, namedtuple


def split_service_slot(label):
    service, slot, value = label.split('--')
    return (f'{service}--{slot}', value)

def extract_labels(dialog):
    results = D.extract_labels(dialog)
    if len(D.transfer_slots) == 0:
        return []
    from_to_slots = {split_service_slot(t)[0]: split_service_slot(f)[0] 
                     for f, t in D.transfer_slots}
    D.transfer_slots = set()
    output = []
    for result in results:
        states = dict([split_service_slot(s) for s in result['states']])
        labels = set()
        for label in result['labels']:
            to_serv_slot, value = split_service_slot(label)
            if value != 'transferred' or to_serv_slot not in from_to_slots:
                continue
            output.append({'user': result['user'], 'from_slot': from_to_slots[to_serv_slot],
                           'to_slot': to_serv_slot, 'system': result['system'], 'id': result['id']})
    return output


class SlotDataset(Dataset):

    BATCH = namedtuple('Batch', ['x', 'y', 'id', 'slot'])
    SEQUENCE = namedtuple('Sequence', ['input_ids', 'attention_mask', 'token_type_ids'])
    
    def __init__(self, data_url, tokenizer, max_len=50, random_seed=42):
        random.seed(random_seed)
        with open(os.path.join(data_url, 'schema.json'), 'r') as f:
            self.schema, self.num_class = self._extract_schema(json.load(f))
        with open(os.path.join(data_url, 'dialogues.json'), 'r') as f:
            self.data = [result for dialog in json.load(f) for result in extract_labels(dialog)]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def _extract_schema(schema):
        services = {}
        max_slots = 0
        for service in schema:
            service_name = service['service_name']
            service_desc = service['description']
            time_slots, area_slots, people_slots = 0, 0, 0
            for slot in service['slots']:
                    slot_name = slot['name']
                    normalised_slot = D.normalise_slot(slot_name)
                    if normalised_slot not in ['area', 'date_time', 'people']:
                        continue
                    if normalised_slot == 'area':
                        area_slots += 1
                    elif normalised_slot == 'date_time':
                        time_slots += 1
                    else:
                        people_slots =+ 1
                    slot_desc = slot['description']
                    services[f'{service_name}--{slot_name}'] = [service_desc, slot_desc,
                                                                                   normalised_slot]
            max_slots = max(max_slots, time_slots, area_slots, people_slots)
        return services, max_slots
    
    def _convert_to_example(self, datum):
        to_serv_slot, from_serv_slot, user = datum['to_slot'], datum['from_slot'], datum['user']
        to_service_desc, to_slot_desc, normalised_slot = self.schema[to_serv_slot]
        to_service_desc = 'To service description: ' + to_service_desc
        to_slot_desc = 'To slot description: ' + to_slot_desc
        choices, id2value = [], []
        to_service, to_slot = to_serv_slot.split('--')
        from_service, from_slot = from_serv_slot.split('--')
        for serv_slot in self.schema:
            service, slot = serv_slot.split('--')
            if service == from_service and D.normalise_slot(slot) == normalised_slot:
                srv_dsc, slot_dsc, _ = self.schema[serv_slot]
                choices.append(f'From service description: {srv_dsc} ' + \
                               f'[SEP] From slot description: {slot_dsc}')
                id2value.append(serv_slot)
                if serv_slot == from_serv_slot:
                    label = len(choices) - 1

        if len(choices) < 2:
            return None
        context = 'System: ' + datum['system'] + ' [SEP] ' + 'User: ' + user + ' [SEP] ' \
                                  + to_service_desc + ' [SEP] ' + to_slot_desc
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

    def _convert_to_batch(self, datum):
        example = self._convert_to_example(datum)
        if example is None:
            return None
        feature = self._convert_to_feature(example['context'], example['choices'])
        value = '--'.join(example['id2value'])
        slot_value = f'{datum["from_slot"]}--{datum["to_slot"]}--{example["label"]}'
        return self.BATCH(x=feature, y=example['label'], id=datum['id'], slot=slot_value)
    
    def __getitem__(self, idx):
        batch = self._convert_to_batch(self.data[idx])
        if batch is None:
            return self.__getitem__((idx + 1) % len(self))
        return batch

    def load_valid_data(self, batch_size):
        batches = []
        batch = []
        keys = ['x', 'y', 'id', 'slot']
        for datum in self.data:
            b = self._convert_to_batch(datum)
            if b is None:
                continue
            batch.append(b)
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