# Modular BERT for Dialogue State Tracking

A modular approach to Dialogue state tracking that utilises one pre-trained language model as a base model and employs small layers, known as adapter layers, to each specialised sub-task in the Schema Guided Dataset, detialed description [here](Modular_BERT_description.pdf).

- Slot Classification Module [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osmanio2/modular-bert-dst/blob/main/slot_classification_training.ipynb)
- Intent Classification Module [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osmanio2/modular-bert-dst/blob/main/intent_classification_training.ipynb)
- Request Slots Module [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osmanio2/modular-bert-dst/blob/main/request_slot_training.ipynb)
- Span Slots Module [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osmanio2/modular-bert-dst/blob/main/span_slots_training.ipynb)
- Transfer Slots Module [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osmanio2/modular-bert-dst/blob/main/transfer_slots_training.ipynb)
- Overall DST Model in Inference Mode [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osmanio2/modular-bert-dst/blob/main/Dialogue_State_Tracker_inference.ipynb)
