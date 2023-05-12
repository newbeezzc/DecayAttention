import os

model = 'Autoformer'
p_fault = 0.0015
# pred_lens = [24, 48, 168, 336, 720]
pred_lens = [720]
# pred_lens = [24,36,48,60]
datas = ['weather']
for data in datas:
    data_path = data + '.csv'
    data = 'custom'
    root_path = './dataset/weather/'
    dim = 21
    for pred_len in pred_lens:
        # attn = 'auto'
        # print("=======================p fault {}|attn {}=======================".format(p_fault, attn))
        # os.system("python run.py --is_training 1 --root_path {} --data_path {} --model_id ETTh1_48_48 \
        #   --model {} --data {} --features M --seq_len 96 --label_len 96 --pred_len {} --e_layers 2 --d_layers 1 \
        #   --factor 3 --enc_in {} --dec_in {} --c_out {} --des 'Exp' --itr 3 --p_fault {} --attn {}".format(root_path, data_path, model,
        #                                                                                                 data, pred_len, dim, dim, dim,
        #                                                                                                 p_fault, attn))

        attn = 'delay'
        print("=======================p fault {}|attn {}=======================".format(p_fault, attn))
        os.system("python run.py --is_training 1 --root_path {} --data_path {} --model_id ETTh1_48_48 \
                  --model {} --data {} --features M --seq_len 96 --label_len 96 --pred_len {} --e_layers 2 --d_layers 1 \
                  --factor 3 --enc_in {} --dec_in {} --c_out {} --des 'Exp' --itr 3 --p_fault {} --attn {}".format(
            root_path, data_path, model,
            data, pred_len, dim, dim, dim,
            p_fault, attn))
