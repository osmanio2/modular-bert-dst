import os
import json
import random
import torch as T
import numpy as np
import slot_class_data as D
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, namedtuple


def extract_labels(dialog):
    results = []
    system = ""
    dialog_id = dialog['dialogue_id']
    servs = dialog['services']
    for i, turn in enumerate(dialog['turns']):
        if turn['speaker'] == 'USER':
            result = {'id': f'{dialog_id}-{i}'}
            result['user'] = turn['utterance']
            result['system'] = system
            frames = turn['frames']
            req_slots = {}
            for frame in frames:
                #if len(frame['state']['requested_slots']) > 0:
                req_slots[frame['service']] = set(frame['state']['requested_slots'])
            #if len(req_slots) == 0:
            #    continue
            result['labels'] = req_slots
            results.append(result)
        elif turn['speaker'] == 'SYSTEM':
            system = turn['utterance']
    return results


class RequestSlotDataset(Dataset):

    BATCH = namedtuple('Batch', ['x', 'y', 'id', 'req_slot'])
    SEQUENCE = namedtuple('Sequence', ['input_ids', 'attention_mask', 'token_type_ids'])
    
    def __init__(self, data_url, tokenizer, max_len=50, random_seed=42):
        random.seed(random_seed)
        with open(os.path.join(data_url, 'schema.json'), 'r') as f:
            self.schema, self.num_class = self._extract_req_slot_schema(json.load(f))
        with open(os.path.join(data_url, 'dialogues.json'), 'r') as f:
            self.data = [result for dialog in json.load(f) for result in extract_labels(dialog)]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def _extract_req_slot_schema(schema):
        services = {}
        max_num_classes = 0
        for service in schema:
            service_name = service['service_name']
            service_desc = service['description']
            max_num_classes = max(max_num_classes, len(service['slots']))
            services[service_name] = [service_desc, {}]
            for slot in service['slots']:
                slot_name = slot['name']
                slot_desc = slot['description']
                services[service_name][1][slot_name] = slot_desc 
        return services, max_num_classes
    
    def _sample_req_slot(self, labels, test=False):
        if test:
            return list(labels.items())
        services = list(labels.keys())
        random.shuffle(services)
        return services[0], labels[services[0]]
    
    def _convert_to_example(self, example, service, req_slots):
        service_desc, slots = self.schema[service]
        service_desc = 'Service description: ' + service_desc
        system = example['system']
        if system == '':
            system = 'Hello, how can I help?'
        #context = 'System: ' + system + ' [SEP] User: ' + example['user'] + ' [SEP] ' \
        #                                + service_desc
        context = 'User: ' + example['user'] + ' [SEP] '  + service_desc
        id2slot = list(slots.keys())
        slot2id = {k: i for i, k in enumerate(id2slot)}
        choices = [slots[k] for k in id2slot]
        label = np.array([1 if id2slot[i] in req_slots else 0 for i in range(len(id2slot))] \
                                 + [0] * (self.num_class - len(id2slot)))
        return {'context': context, 'choices': choices, 'label': label, 'id2slot': id2slot}
    
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

    def _convert_to_batch(self, datum, service, req_slots):
        example = self._convert_to_example(datum, service, req_slots)
        feature = self._convert_to_feature(example['context'], example['choices'])
        req_slots = '--'.join([f'{s}${l}' for s, l in zip(example['id2slot'], example['label'])])
        serv_slot = f'{service}--{req_slots}'
        return self.BATCH(x=feature, y=example['label'], id=datum['id'], req_slot=serv_slot)
    
    def __getitem__(self, idx):
        datum = self.data[idx]
        service, req_slots = self._sample_req_slot(datum['labels'])
        return self._convert_to_batch(datum, service, req_slots)

    def load_valid_data(self, batch_size):
        batches = []
        batch = []
        keys = ['x', 'y', 'id', 'req_slot']
        for datum in self.data:
            for service, req_slots in self._sample_req_slot(datum['labels'], test=True):
                batch.append(self._convert_to_batch(datum, service, req_slots))
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