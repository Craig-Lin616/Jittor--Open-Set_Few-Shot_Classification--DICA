import jittor as jt

model_2_dict = jt.load('./weight/resnet.pkl')
aux_model_2_dict = {}
aux_adapter_2_dict = {}

for key, value in model_2_dict.items():
    if key.startswith('aux_model.'):
        new_key = key.replace('aux_model.', '')
        aux_model_2_dict[new_key] = value

for key, value in model_2_dict.items():
    if key.startswith('aux_adapter.'):
        new_key = key.replace('aux_adapter.fc.', '')
        aux_adapter_2_dict[new_key] = value

jt.save(aux_model_2_dict, './weight/resnet_aux_model.pkl')
jt.save(aux_adapter_2_dict, './weight/resnet_aux_adapter.pkl')