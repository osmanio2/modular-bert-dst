import types
import torch as T
import numpy as np
from abc import ABC, abstractmethod
from transformers import BertForQuestionAnswering, BertPreTrainedModel, BertForMultipleChoice
from transformers import BertForSequenceClassification
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, namedtuple


class AdapterLayer(T.nn.Module):
    
    def __init__(self, input_size, dropout, hidden_size=64, init_scale=1e-3, act='gelu'):
        super().__init__()
        self.act = getattr(T.nn.functional, act)
        self.squeeze = T.nn.Linear(input_size, hidden_size)
        self.squeeze.weight.data.normal_(mean=0.0, std=init_scale)
        T.nn.init.zeros_(self.squeeze.bias.data)
        self.expand = T.nn.Linear(hidden_size, input_size)
        self.expand.weight.data.normal_(mean=0.0, std=init_scale)
        T.nn.init.zeros_(self.expand.bias.data)
        self.dropout = dropout
        
    def forward(self, x):
        x = self.dropout(x)
        return self.expand(self.act(self.squeeze(x))) + x

 
def add_adapter_layers(model, **kwargs):
    input_size = model.config.hidden_size
    layers = model.encoder.layer
    for layer in layers:
        adapter1 = AdapterLayer(input_size, layer.attention.output.dropout, **kwargs)
        adapter2 = AdapterLayer(input_size, layer.output.dropout, **kwargs)
        layer.attention.output.add_module('dropout', adapter1)
        layer.output.add_module('dropout', adapter2)
    return model


class DSTModule(ABC):
    def __init__(self, bert_model, tokenizer, adapter_layer, schema, device='cpu', max_len=120):
        self.model = self._load_model(bert_model)
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.adapter_layer = adapter_layer
        self.schema = self._process_schema(schema)

    def _adapter_to_device(self):
        for p in self.adapter_layer.values():
            p.to(self.device)

    def _attach_module(self):
        self.model.load_state_dict(self.adapter_layer, False)
        self.model.to(self.device)

    @classmethod
    def _load_model(cls, bert_model):
        model = cls.MODEL(bert_model)
        model.forward = types.MethodType(cls.BASEMODEL.forward, model)
        return model

    @staticmethod
    @abstractmethod
    def _process_schema(schema):
        ...
    
    @abstractmethod
    def _convert_to_batch(self, system, user, schema):
        ...

    @abstractmethod
    def predict(self, system, user, schema):
        ...

    
class SpanSlotModule(DSTModule):

    BATCH = namedtuple('Batch', ['features'])

    class SpanSlotModel(BertPreTrainedModel):
        def __init__(self, bert_model):
            super().__init__(bert_model.config)
            self.num_labels = bert_model.config.num_labels
            self.bert = bert_model
            self.qa_outputs = T.nn.Linear(self.bert.config.hidden_size, self.num_labels)
    
    MODEL = SpanSlotModel
    BASEMODEL = BertForQuestionAnswering

    @staticmethod
    def _process_schema(schema):
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

    def _convert_to_batch(self, system, user, service_slot):
        serv_desc, slot_desc = self.schema[service_slot]
        question = 'Service description: ' + serv_desc + ' [SEP] Slot description: ' + slot_desc
        answer = 'System: ' + system + ' [SEP] User: ' + user
        enc = self.tokenizer.encode_plus(question, answer, max_length=self.max_len, add_special_tokens=True,
                                pad_to_max_length=True, truncation_strategy='only_first')
        enc = {k: T.tensor(v, device=self.device).unsqueeze(0) for k, v in enc.items()}
        return self.BATCH(enc)

    def _get_span(self, input_ids, strt, end):
        if (strt == 0 and end == 0) or (strt > end):
            span = 'Not mentioned'
        else:
            span = input_ids[strt:end+1]
            span = ' '.join(self.tokenizer.convert_ids_to_tokens(span, skip_special_tokens=True))
        return span.replace(' ##', '')

    def predict(self, system, user, service_slot):
        self._attach_module()
        batch = self._convert_to_batch(system, user, service_slot)
        self.model.eval()
        with T.no_grad():
            strt_logits, end_logits  = self.model(**batch.features)
            score = T.softmax(strt_logits, dim=-1)[0].max()*T.softmax(end_logits, dim=-1).max()
            score = score.item()
        strt_index, end_index = T.max(strt_logits, dim=-1)[1][0], T.max(end_logits, dim=-1)[1][0]
        strt_index, end_index = strt_index.cpu().numpy(), end_index.cpu().numpy()    
        value = self._get_span(batch.features['input_ids'][0], strt_index, end_index)
        self.model.train()
        preds = (value, score)
        state = {service_slot: value}
        return {'state': state, 'predictions': preds}


class CategoricalSlotModule(DSTModule):

    BATCH = namedtuple('Batch', ['values', 'features'])
    FEATURE = namedtuple('Feature', ['input_ids', 'attention_mask', 'token_type_ids'])

    class CategoricalSlotModel(BertPreTrainedModel):
        def __init__(self, bert_model):
            super().__init__(bert_model.config)
            self.bert = bert_model
            self.dropout = T.nn.Dropout(bert_model.config.hidden_dropout_prob)
            self.classifier = T.nn.Linear(bert_model.config.hidden_size, 1)

    MODEL = CategoricalSlotModel
    BASEMODEL = BertForMultipleChoice

    @staticmethod
    def _process_schema(schema):
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
                            value = CategoricalSlotModule._process_year(value)
                        values[value] = value
                    services[f'{service_name}--{slot_name}'] = [service_desc, slot_desc, values]
        return services
    
    @staticmethod
    def _process_year(value):
        year = int(value)
        rel = f'{2019-year} years ago'
        if year == 2019:
            rel = f'this year'
        return f'{year} or {rel} or {year - 2000}'

    @staticmethod
    def _reverse_year(value):
        return value.split(' ')[0]

    def _convert_to_batch(self, system, user, service_slot):
        if system == '':
            system = 'Hello, how can I help?'
        service_desc, slot_desc, values = self.schema[service_slot]
        context = 'System: ' + system + ' [SEP] User: ' + user + ' [SEP] ' \
                                        + service_desc + ' [SEP] ' + slot_desc
        id2value = list(values.keys())
        value2id = {k: i for i, k in enumerate(id2value)}
        choices = [values[k] for k in id2value]
        encoded_sequence = [self.tokenizer.encode_plus(context, choice, add_special_tokens=True,
                                                max_length=self.max_len, pad_to_max_length=True)
                            for choice in choices]
        keys = list(encoded_sequence[0].keys())
        feature = dict()
        num_class = len(value2id)
        for key in keys:
            feature[key] = T.tensor([seq[key] for seq in encoded_sequence],
             device=self.device).unsqueeze(0)
        if service_slot.split('--')[1] == 'year':
            id2value = [self._reverse_year(v) for v in id2value]
        return self.BATCH(id2value, self.FEATURE(**feature))

    def _evaluate(self, features):
        self.model.eval()
        with T.no_grad():
            scores = self.model(**features._asdict())[0]
            scores = T.softmax(scores, dim=-1).squeeze(0).cpu().numpy()
        self.model.train()
        return scores
    
    def predict(self, system, user, service_slot):
        self._attach_module()
        batch = self._convert_to_batch(system, user, service_slot)
        scores = self._evaluate(batch.features)
        preds = {value: score for value, score in zip(batch.values, scores)}
        state = {service_slot: batch.values[int(np.argmax(scores))]}
        return {'state': state, 'predictions': preds}  


class TransferSlotModule(DSTModule):
    BATCH = namedtuple('Batch', ['from_slots', 'features'])
    FEATURE = namedtuple('Feature', ['input_ids', 'attention_mask', 'token_type_ids'])

    class TransferSlotModel(BertPreTrainedModel):
        def __init__(self, bert_model):
            super().__init__(bert_model.config)
            self.bert = bert_model
            self.dropout = T.nn.Dropout(bert_model.config.hidden_dropout_prob)
            self.classifier = T.nn.Linear(bert_model.config.hidden_size, 1)

    MODEL = TransferSlotModel
    BASEMODEL = BertForMultipleChoice

    @staticmethod
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

    @staticmethod
    def _process_schema(schema):
        services = {}
        for service in schema:
            service_name = service['service_name']
            service_desc = service['description']
            for slot in service['slots']:
                    slot_name = slot['name']
                    normalised_slot = TransferSlotModule.normalise_slot(slot_name)
                    if normalised_slot not in ['area', 'date_time', 'people']:
                        continue
                    slot_desc = slot['description']
                    services[f'{service_name}--{slot_name}'] = [service_desc, slot_desc,
                                                                normalised_slot]
        return services

    def _convert_to_batch(self, system, user, to_serv_slot, from_service, from_slots):
        to_service_desc, to_slot_desc, normalised_slot = self.schema[to_serv_slot]
        to_service_desc = 'To service description: ' + to_service_desc
        to_slot_desc = 'To slot description: ' + to_slot_desc
        choices, id2value = [], []
        to_service, to_slot = to_serv_slot.split('--')
        for serv_slot in self.schema:
            service, slot = serv_slot.split('--')
            in_slots = slot in from_slots if from_slots is not None else True
            if service == from_service and in_slots and \
            TransferSlotModule.normalise_slot(slot) == normalised_slot:
                srv_dsc, slot_dsc, _ = self.schema[serv_slot]
                choices.append(f'From service description: {srv_dsc} ' + \
                                f'[SEP] From slot description: {slot_dsc}')
                id2value.append(serv_slot)
        if len(choices) < 2:
            return id2value
        context = 'User: ' + user + ' [SEP] ' + to_service_desc + ' [SEP] ' + to_slot_desc
        encoded_sequence = [self.tokenizer.encode_plus(context, choice, add_special_tokens=True,
                                                    max_length=self.max_len, pad_to_max_length=True)
                                for choice in choices]
        keys = list(encoded_sequence[0].keys())
        feature = dict()
        for key in keys:
            feature[key] = T.tensor([seq[key] for seq in encoded_sequence],
             device=self.device).unsqueeze(0)
        return self.BATCH(id2value, self.FEATURE(**feature))

    def _evaluate(self, features):
        self.model.eval()
        with T.no_grad():
            scores = self.model(**features._asdict())[0]
            scores = T.softmax(scores, dim=-1).squeeze(0).cpu().numpy()
        self.model.train()
        return scores
    
    def predict(self, system, user, to_serv_slot, from_service, from_slots=None):
        if to_serv_slot not in self.schema:
            return None
        batch = self._convert_to_batch(system, user, to_serv_slot, from_service, from_slots)
        if isinstance(batch, tuple):
            self._attach_module()
            scores = self._evaluate(batch.features)
        else:
            if len(batch) == 0:
                return None
            batch = self.BATCH(batch, None)
            scores = [1]
        preds = {from_slot: score for from_slot, score in zip(batch.from_slots, scores)}
        state = {to_serv_slot: batch.from_slots[int(np.argmax(scores))]}
        return {'state': state, 'predictions': preds}  

    
class SlotClassificationModule(DSTModule):

    BATCH = namedtuple('Batch', ['slots', 'features'])
    FEATURE = namedtuple('Feature', ['input_ids', 'attention_mask', 'token_type_ids'])
    LABELS = ['filled', 'transferred', 'dontcare', 'None']

    class SlotClassificationModel(BertPreTrainedModel):
        def __init__(self, bert_model):
            super().__init__(bert_model.config)
            self.num_labels = 4
            self.hidden_size = bert_model.config.hidden_size
            self.bert = bert_model
            self.dropout = T.nn.Dropout(bert_model.config.hidden_dropout_prob)
            self.classifier = T.nn.Linear(self.hidden_size, self.num_labels)

    MODEL = SlotClassificationModel
    BASEMODEL = BertForSequenceClassification

    @staticmethod
    def _process_schema(schema):
        services = defaultdict(list)
        for service in schema:
            service_name = service['service_name']
            service_desc = service['description']
            for slot in service['slots']:
                slot_name = slot['name']
                slot_desc = slot['description']
                services[service_name].append((slot_name, service_desc, slot_desc))
        return services

    def _convert_to_batch(self, system, user, service):
        if system == '':
            system = 'Hello, how can I help?'
        utterance = 'System: ' + system + ' [SEP] User: ' + user
        batch = []
        for slot_name, service_desc, slot_desc in self.schema[service]:
            service_desc = 'Service description: ' + service_desc
            slot_desc = 'Slot description: ' + slot_desc
            schema = service_desc + ' [SEP] ' + slot_desc
            features = self.tokenizer.encode_plus(utterance, schema, add_special_tokens=True,
                                             max_length=self.max_len, pad_to_max_length=True,
                                             return_tensors='pt')
            features = self.FEATURE(**{k: v.squeeze(0).to(self.device)
                                          for k, v in features.items()})
            batch.append(self.BATCH(slot_name, features))
        return self.BATCH(*T.utils.data._utils.collate.default_collate(batch))

    def predict(self, system, user, service):
        self._attach_module()
        batch = self._convert_to_batch(system, user, service)
        self.model.eval()
        with T.no_grad():
            scores = T.softmax(self.model(**batch.features._asdict())[0], dim=-1).cpu().numpy()
        ids = np.argmax(scores, axis=-1)
        self.model.train()
        preds = {slot: {k: v for k, v in zip(self.LABELS, score)} 
                 for slot, score in zip(batch.slots, scores)}
        state = {slot: self.LABELS[idx] for slot, idx in zip(batch.slots, ids) if idx < 3}
        return {'state': state, 'predictions': preds}


class IntentClassificationModule(DSTModule):

    BATCH = namedtuple('Batch', ['values', 'features'])
    FEATURE = namedtuple('Feature', ['input_ids', 'attention_mask', 'token_type_ids'])

    class IntentClassificationModel(BertPreTrainedModel):
        def __init__(self, bert_model):
            super().__init__(bert_model.config)
            self.bert = bert_model
            self.dropout = T.nn.Dropout(bert_model.config.hidden_dropout_prob)
            self.classifier = T.nn.Linear(bert_model.config.hidden_size, 1)

    MODEL = IntentClassificationModel
    BASEMODEL = BertForMultipleChoice

    @staticmethod
    def _process_schema(schema):
        services = {}
        for service in schema:
            service_name = service['service_name']
            service_desc = service['description']
            services[service_name] = [service_desc, {'NONE': 'None of the intentions are active'}]
            for intent in service['intents']:
                intent_name = intent['name']
                intent_desc = intent['description']
                services[service_name][1][intent_name] = intent_desc
        return services

    def _convert_to_batch(self, system, user, service, prev_intents):
        if system == '':
            system = 'Hello, how can I help?'
        service_desc, intents = self.schema[service]
        service_desc = 'Service description: ' + service_desc
        context = 'System: ' + system + ' [SEP] User: ' + user + ' [SEP] ' + service_desc
        id2intent = list(intents.keys())
        intent2id = {k: i for i, k in enumerate(id2intent)}
        choices = [intents[k] + (' [SEP] Previous user intention' if service in prev_intents and \
                                k == prev_intents[service] else '') for k in id2intent]
        #print({'context': context, 'choices': choices, 'id2intent': id2intent})
        encoded_sequence = [self.tokenizer.encode_plus(context, choice, add_special_tokens=True,
                                                max_length=self.max_len, pad_to_max_length=True)
                            for choice in choices]
        keys = list(encoded_sequence[0].keys())
        feature = dict()
        num_class = len(id2intent)
        for key in keys:
            feature[key] = T.tensor([seq[key] for seq in encoded_sequence],
             device=self.device).unsqueeze(0)
        return self.BATCH(id2intent, self.FEATURE(**feature))

    def _evaluate(self, features):
        self.model.eval()
        with T.no_grad():
            scores = self.model(**features._asdict())[0]
            scores = T.softmax(scores, dim=-1).squeeze(0).cpu().numpy()
        self.model.train()
        return scores
    
    def predict(self, system, user, service, previous_intents=None):
        self._attach_module()
        batch = self._convert_to_batch(system, user, service, previous_intents)
        scores = self._evaluate(batch.features)
        preds = {value: score for value, score in zip(batch.values, scores)}
        state = {service: batch.values[int(np.argmax(scores))]}
        return {'state': state, 'predictions': preds}
    
    
class RequestSlotModule(DSTModule):

    BATCH = namedtuple('Batch', ['values', 'features'])
    FEATURE = namedtuple('Feature', ['input_ids', 'attention_mask', 'token_type_ids'])

    class RequestSlotModel(BertPreTrainedModel):
        def __init__(self, bert_model):
            super().__init__(bert_model.config)
            self.bert = bert_model
            self.dropout = T.nn.Dropout(bert_model.config.hidden_dropout_prob)
            self.classifier = T.nn.Linear(bert_model.config.hidden_size, 1)

    MODEL = RequestSlotModel
    BASEMODEL = BertForMultipleChoice

    @staticmethod
    def _process_schema(schema):
        services = {}
        for service in schema:
            service_name = service['service_name']
            service_desc = service['description']
            services[service_name] = [service_desc, {}]
            for slot in service['slots']:
                slot_name = slot['name']
                slot_desc = slot['description']
                services[service_name][1][slot_name] = slot_desc 
        return services

    def _convert_to_batch(self, system, user, service):
        if system == '':
            system = 'Hello, how can I help?'
        service_desc, slots = self.schema[service]
        #context = 'System: ' + system + ' [SEP] User: ' + user + ' [SEP] '  + service_desc 
        context = 'User: ' + user + ' [SEP] '  + service_desc 
        id2slot = list(slots.keys())
        slot2id = {k: i for i, k in enumerate(id2slot)}
        choices = [slots[k] for k in id2slot]
        encoded_sequence = [self.tokenizer.encode_plus(context, choice, add_special_tokens=True,
                                                max_length=self.max_len, pad_to_max_length=True)
                            for choice in choices]
        keys = list(encoded_sequence[0].keys())
        feature = dict()
        num_class = len(id2slot)
        for key in keys:
            feature[key] = T.tensor([seq[key] for seq in encoded_sequence],
             device=self.device).unsqueeze(0)
        return self.BATCH(id2slot, self.FEATURE(**feature))

    def _evaluate(self, features):
        self.model.eval()
        with T.no_grad():
            scores = self.model(**features._asdict())[0]
            scores = T.sigmoid(scores).squeeze(0).cpu().numpy()
        self.model.train()
        return scores
    
    def predict(self, system, user, service):
        self._attach_module()
        batch = self._convert_to_batch(system, user, service)
        scores = self._evaluate(batch.features)
        preds = {value: score for value, score in zip(batch.values, scores)}
        state = [value for i, value in enumerate(batch.values) if scores[i] > 0.99]
        return {'state': state, 'predictions': preds}