import re
import ast
import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def get_labels(file_path):
    jsondata = json.load(open(file_path))

    D = defaultdict(list)   
    for i, temp in enumerate(jsondata):        
        key = temp['images'][-1].split('/')[-2]
                
        l_val = ast.literal_eval(re.sub(r'([a-zA-Z_]\w*)', r"'\1'", temp['messages'][-1]['content']))
        
        D[key].append(l_val)

    
    actual_label = {}
    for folder_num,image_list in D.items():   
        actual_label_each_folder = defaultdict(lambda: {'malicious': 0, 'benign': 0})     
        for each_image_dict in image_list:            
            for key, value in each_image_dict.items():
                actual_label_each_folder[key][value] += 1
        actual_label[folder_num] = actual_label_each_folder

    majority_label = {}
    for k,v in actual_label.items():
        temp_dict = {}
        for each_box_num,each_box_label in v.items():
            #Find label which has highest count
            if each_box_label['malicious'] > each_box_label['benign']:
                temp_dict[each_box_num] = 'malicious'
            elif each_box_label['benign'] > each_box_label['malicious']:
                temp_dict[each_box_num] = 'benign'
            else:
                temp_dict[each_box_num] = 'equal'
        majority_label[k] = temp_dict

    return majority_label

def get_metrics(preds, labels):
    #given two dictionaries of predictions and labels, find how many are missing, how many ar extra, and how many are correct
    #missing: in labels but not in preds
    #extra: in preds but not in labels
    #correct: in both and match
    missing = 0
    extra = 0
    correct = 0
    wrong = 0
    malicious_correct = 0
    benign_correct = 0
    malicious_wrong = 0
    benign_wrong = 0
    
    for k in labels:
        if k not in preds:
            missing += 1
        else:
            if preds[k] == labels[k]:
                correct += 1
                if preds[k] == 'malicious':
                    malicious_correct += 1
                elif preds[k] == 'benign':
                    benign_correct += 1
            else:
                wrong += 1
                if preds[k] == 'malicious':
                    malicious_wrong += 1
                elif preds[k] == 'benign':
                    benign_wrong += 1

    for k in preds:
        if k not in labels:
            extra += 1
    return missing, extra, correct, wrong, malicious_correct, benign_correct, malicious_wrong, benign_wrong 


# jsondata = json.load(open('/data1/lokesh/bep/data_gen/openphish_complete_data_sharegpt/test_sharegpt_data.json'))
# preds = pd.read_json('/data1/lokesh/LlamaFactory/generated_predictions.jsonl', lines=True)['predict'].to_list()
# all_labels = get_labels('/data1/lokesh/bep/data_gen/openphish_complete_data_sharegpt/test_sharegpt_data.json')
preds_file = pd.read_json('/data1/lokesh/LlamaFactory/generated_predictions_2epoch.jsonl', lines=True)['predict'].to_list()
labels_file = pd.read_json('/data1/lokesh/LlamaFactory/generated_predictions_2epoch.jsonl', lines=True)['label'].to_list()
missing = 0
extra = 0
correct = 0
wrong = 0
total = 0
malicious_correct = 0
benign_correct = 0
malicious_wrong = 0
benign_wrong = 0

for i, temp in enumerate(tqdm(preds_file)):        
    # key = temp['images'][-1].split('/')[-2]
    # if i==311:
    #     continue

    # try:    
        # Evaluating the prediction and label    
        #Extract everything between{}
        temp_preds = re.findall(r'\{(.*?)\}', temp, re.DOTALL)[0]
        #remove \n, extra spaces, single quotes and \"
        temp_preds = re.sub(r'\s+', ' ', temp_preds).replace("'", '"').replace('"', '').strip()
        # temp_preds = ast.literal_eval('{' + temp_preds + '}')
        formatted = "{" + ", ".join(
                    f'{k.strip()}: "{v.strip()}"'
                    for part in temp_preds.split(",")
                    for k, v in [part.split(":")]
                ) + "}"
        temp_preds = ast.literal_eval(formatted)
        # temp_preds = ast.literal_eval(temp_preds)
        # temp_preds = ast.literal_eval(re.sub(r'([a-zA-Z_]\w*)', r"'\1'", preds_file[i]))
        # l_val = ast.literal_eval(re.sub(r'([a-zA-Z_]\w*)', r"'\1'", temp['messages'][-1]['content']))
        # labels = {k:v for k, v in all_labels[key].items() if k in l_val.keys()}
        # labels = ast.literal_eval(re.sub(r'([a-zA-Z_]\w*)', r"'\1'", labels_file[i]))
        # labels = ast.literal_eval(labels_file[i])
        D = re.findall(r'```json(.*?)```', labels_file[i], re.DOTALL)[0].strip()
        D = re.sub(r'(?<=:)([A-Za-z_]+)', r"'\1'", D)
        labels = ast.literal_eval(D)        
        
        temp_miss,temp_extra,temp_correct,temp_wrong,temp_malicious_correct,temp_benign_correct,temp_malicious_wrong,temp_benign_wrong = get_metrics(temp_preds, labels)
        missing += temp_miss
        extra += temp_extra
        correct += temp_correct
        wrong += temp_wrong
        malicious_correct += temp_malicious_correct
        benign_correct += temp_benign_correct
        malicious_wrong += temp_malicious_wrong
        benign_wrong += temp_benign_wrong
        total += len(labels)
    # except Exception as e:
    #     print(f"Error processing index {i}: {e}")

print(f"Correct: {correct}")
print(f"Missing: {missing}")
print(f"Extra: {extra}")
print(f"Wrong: {wrong}")
print(f"Total: {total}")
print(f"Malicious Correct: {malicious_correct} ({(malicious_correct/total)*100:.2f}%)")
print(f"Benign Correct: {benign_correct} ({(benign_correct/total)*100:.2f}%)")
print(f"Malicious Wrong: {malicious_wrong} ({(malicious_wrong/total)*100:.2f}%)")
print(f"Benign Wrong: {benign_wrong} ({(benign_wrong/total)*100:.2f}%)")
print(f"Accuracy: {(correct/total)*100:.2f}%")

#given malicious correct, malicious wrong, benign correct, and benign wrong, find precision, recall and f1 score
#malicious is 1 and benign is 0
precision = malicious_correct / (malicious_correct + malicious_wrong) if (malicious_correct + malicious_wrong) > 0 else 0
recall = malicious_correct / (malicious_correct + benign_wrong) if (malicious_correct + benign_wrong) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1_score:.3f}")