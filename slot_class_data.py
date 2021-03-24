import sys
import os
import json
import random
import torch as T
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, namedtuple

transfer_slots = set()

def normalise_slot(slot):
    AREAS = ['city', 'location', 'destination', 'area', 'origin', 'where_to']
    PEOPLE = ['travelers', 'group_size', 'number_of_seats', 'number_of_tickets',
              'passengers', 'number_of_adults', 'party_size', 'number_of_riders']
    DATETIMES = ['date', 'time']
    if any([a in slot for a in AREAS]):
        return 'area'
    elif any([d in slot for d in DATETIMES]):
        return 'date_time'
    elif any([d in slot for d in PEOPLE]):
        return 'people'
    return slot

def extract_values(frames):
    slot_values = set()
    for frame in frames:
        service = frame['service']
        for slot, values in frame['state']['slot_values'].items():
            for value in values:
                slot_values.add(f'{service}--{slot}--{value}')
    return slot_values

def slot_value_in_other_domain(slot_value, other):
    service1, slot1, value1 = other.split('--')
    service2, slot2, value2 = slot_value.split('--')
    return service1 != service2 and (value2 == '' or value1 == value2) \
           and normalise_slot(slot1) == normalise_slot(slot2)

def slot_value_in_act(act, slot_value):
    service1, act, slot1, value1 = act.split('--')
    service2, slot2, value2 = slot_value.split('--')
    if act in ['OFFER', 'CONFIRM'] and service1 == service2 \
              and normalise_slot(slot1) == normalise_slot(slot2):
        if value2 == '' or value1 == value2:
            return True
        else:
            return None
    return False
            

def in_system_act(acts, slot_value):
    service2, slot2, value2 = slot_value.split('--')
    for act in acts:
        service1, act, slot1, value1 = act.split('--')
        if (service1 == service2 and slot1 == slot2 \
           and (value1 == value2 or act in ['REQUEST', 'OFFER'])):
           return True
    return False
    

def in_user_text(values, user, frames, acts):
    new_values = defaultdict(bool)
    for slot_value in values:
        service, slot, value = slot_value.split('--')
        in_user = False
        for frame in frames:
            if frame['service'] != service:
                continue
            for span_slot in frame['slots']:
                strt, end = span_slot['start'], span_slot['exclusive_end']
                if span_slot['slot'] == slot and user[strt:end] == value:
                    in_user = True
                    break
            if in_user or in_system_act(acts, slot_value):
                new_values[slot_value] =  True
                break

        if not in_user and not new_values[slot_value]:
            new_values[slot_value] = False
    return new_values

def is_transfer(results, slot_value):
    for result in results[::-1]:
        for act in result['acts']:
            in_act = slot_value_in_act(act, slot_value)
            if in_act:
                return 'in_act--' + ' '.join([s.replace('_', ' ') for s in act.split('--')[1:]])
        for value in result['states']:
            if slot_value_in_other_domain(slot_value, value):
                return 'other_domain--' + value
    return None

def extract_transfer_states(results, new_values):
    global transfer_slots
    transfer_states = set()
    for slot_value, filled in new_values.items():
        if not filled and 'dontcare' not in slot_value:
            state = is_transfer(results, slot_value)
            if state is None:
                continue
            if 'in_act' in state:
                transfer_states.add(slot_value)
            elif 'other_domain' in state:
                transfer_states.add(slot_value)
                transfer_slots.add((state.split('--')[1], slot_value))
                
    return transfer_states

def extract_acts(frames):
    acts = set()
    for frame in frames:
        service = frame['service']
        for action in frame['actions']:
            act = action['act']
            slot = action['slot']
            if len(action['values']) == 0:
                acts.add(f'{service}--{act}--{slot}--')
                continue
            for value in action['values']:
                acts.add(f'{service}--{act}--{slot}--{value}')
    return acts

def extract_slots(new_values, transfer_states):
    slots = set()
    non_transfer_slots = set()
    for slot_value in new_values:
        service, slot, value = slot_value.split('--')
        if value != 'dontcare':
            value = 'filled'
        slots.add(f'{service}--{slot}--{value}')
        non_transfer_slots.add(f'{service}--{slot}')
    for slot_value in transfer_states:
        service, slot, value = slot_value.split('--')
        if f'{service}--{slot}' not in non_transfer_slots:
            slots.add(f'{service}--{slot}--transferred')
    return slots

def split_service_slot(label):
    service, slot, value = label.split('--')
    return (f'{service}--{slot}', value)

def convert_schema_to_service_slot(schema):
    services = set()
    for service in schema:
        service_name = service['service_name']
        for slot in service['slots']:
            slot_name = slot['name']
            services.add(f'{service_name}--{slot_name}--')
    return services

def extract_labels(dialog, schema):
    results = []
    system = ""
    acts = set()
    prev_values = set()
    dialog_id = dialog['dialogue_id']
    servs = dialog['services']
    schema = convert_schema_to_service_slot(schema)
    for i, turn in enumerate(dialog['turns']):
        if turn['speaker'] == 'USER':
            result = {'id': f'{dialog_id}-{i}'}
            result['user'] = turn['utterance']
            result['system'] = system
            frames = turn['frames']
            values = extract_values(frames)
            new_values = values - prev_values
            other_states1 = {split_service_slot(v)[0]: is_transfer(results, v) for v in new_values}
            transfer_states = extract_transfer_states(results, in_user_text(new_values,
                                                                            result['user'],
                                                                            frames, acts))
            slots = extract_slots(new_values - transfer_states, transfer_states)
            other_states = {split_service_slot(v)[0]: is_transfer(results, v) for v in schema 
                                        if any([s in v for s in servs ])}
            other_states.update(other_states1)
            result.update({'acts': acts, 'labels': slots, 'states': values,
                          'other_states': {k: v for k, v in other_states.items() if v is not None}})
            results.append(result)
            prev_values = values
        elif turn['speaker'] == 'SYSTEM':
            system = turn['utterance']
            acts = extract_acts(turn['frames'])
            
    return results

def get_classes_counts(data):
    classes = defaultdict(int)
    for d in data:
        for r in extract_labels(d):
            for l in r['labels']:
                classes[l.split('--')[-1]] += 1
    return classes


class SlotDataset(Dataset):
    
    BATCH = namedtuple('Batch', ['x', 'y', 'id', 'slot', 'range'])
    FEATURE = namedtuple('Feature', ['input_ids', 'attention_mask', 'token_type_ids'])
    LABEL2ID = {'filled': 0, 'transferred': 1, 'dontcare': 2, 'None': 3}
    
    def __init__(self, data_url, tokenizer, max_len=50, random_seed=42, pos_prob=0.75):
        random.seed(random_seed)
        with open(os.path.join(data_url, 'schema.json'), 'r') as f:
            schema = json.load(f)
            self.schema = self._extract_categorical_schema(schema)
        with open(os.path.join(data_url, 'dialogues.json'), 'r') as f:
            self.data = [result for dialog in json.load(f) 
                               for result in extract_labels(dialog, schema) if result['labels']]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pos_prob = pos_prob

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def _extract_categorical_schema(schema):
        services = {}
        for service in schema:
            service_name = service['service_name']
            service_desc = service['description']
            for slot in service['slots']:
                slot_name = slot['name']
                slot_desc = slot['description']
                services[f'{service_name}--{slot_name}'] = [service_desc, slot_desc]
        return services
    
    def _sample_categorical_slots(self, labels, states, test=False):
        states = {split_service_slot(state)[0] for state in states}
        labels = [split_service_slot(label) for label in labels]
        services = set([label[0].split('--')[0] for label in labels])
        none_labels = [(k, 'None') for k in set(self.schema.keys()) - {k[0] for k in labels}
                        if k.split('--')[0] in services and k not in states]
        if test:
            return sorted(labels + none_labels, key=lambda x: x[0])
        random.shuffle(labels)
        random.shuffle(none_labels)
        final_labels = labels[:1] + none_labels[:1]
        return final_labels[min(len(final_labels) - 1, int(random.random()>=self.pos_prob))]
    
    def _convert_to_example(self, example, service_slot, value1):
        service_desc, slot_desc = self.schema[service_slot]
        service_desc = 'Service description: ' + service_desc
        slot_desc = 'Slot description: ' + slot_desc
        system = example['system']
        if system == '':
            system = 'Hello, how can I help?'
        utterance = 'System: ' + system + ' [SEP] User: ' + example['user']
        schema = service_desc + ' [SEP] ' + slot_desc
        if service_slot in example['other_states']:
            value = example['other_states'][service_slot]
            if 'in_act' in value:
                value = value.split('--')[1]
            else:
                value = value.split('--')[1:]
                value = ' ' + value[1].replace('_', ' ') + ' ' + value[2]
            schema += ' [SEP] Mentioned in ' + value
            
        label = self.LABEL2ID[value1]
        return {'utterance': utterance, 'schema': schema, 'label': label}
    
    def _convert_to_feature(self, utterance, schema):
        feature = self.tokenizer.encode_plus(utterance, schema, add_special_tokens=True,
                                             max_length=self.max_len, pad_to_max_length=True,
                                             return_tensors='pt')
        return self.FEATURE(**{k: v.squeeze(0) for k, v in feature.items()})

    def _convert_to_batch(self, datum, service_slot, value):
        example = self._convert_to_example(datum, service_slot, value)
        feature = self._convert_to_feature(example['utterance'], example['schema'])
        slot = f'{service_slot}--{value}--{example["label"]}'
        return self.BATCH(x=feature, y=example['label'], id=datum['id'], slot=slot, range=None)
    
    def __getitem__(self, idx):
        datum = self.data[idx]
        batch = []
        for service_slot, value in self._sample_categorical_slots(datum['labels'], datum['states'],
                                      test=True):
            batch.append(self._convert_to_batch(datum, service_slot, value))
        return batch
        
    def load_valid_data(self, batch_size):
        batches = []
        batch = []
        keys = ['x', 'y', 'id', 'slot']
        for datum in self.data:
            for service_slot, value in self._sample_categorical_slots(datum['labels'],
                                                                      datum['states'], test=True):
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
                                pin_memory=pin_memory, drop_last=drop_last, collate_fn=collate)
            for batch in loader:
                yield (self.map_to_cuda(batch) if cuda else batch)
            
            if not loop:
                break
    
def collate(batches):
    i = 0
    new_batches = []
    BATCH = type(batches[0][0])
    for batch in batches:
        for b in batch:
            vals = b._asdict()
            vals['range'] = i+len(batch)
            new_batches.append(BATCH(**vals))
        i += len(batch)
    new_batches = T.utils.data._utils.collate.default_collate(new_batches)
    return BATCH(*new_batches)

