import torch

def transform_predictions_to_labels(sequence_input_lst, wordpiece_mask, idx2label_map, input_type="logit"):
    """
    shape:
        sequence_input_lst: [batch_size, seq_len, num_labels]
        wordpiece_mask: [batch_size, seq_len, ]
    """
    wordpiece_mask = wordpiece_mask.detach().cpu().numpy().tolist()
    if input_type == "logit":
        label_sequence = torch.argmax(torch.nn.functional.softmax(sequence_input_lst, dim=2), dim=2).detach().cpu().numpy().tolist()
    elif input_type == "prob":
        label_sequence = torch.argmax(sequence_input_lst, dim=2).detach().cpu().numpy().tolist()
    elif input_type == "label":
        label_sequence = sequence_input_lst.detach().cpu().numpy().tolist()
    else:
        raise ValueError
    output_label_sequence = []
    for tmp_idx_lst, tmp_label_lst in enumerate(label_sequence):
        tmp_wordpiece_mask = wordpiece_mask[tmp_idx_lst]
        tmp_label_seq = []
        for tmp_idx, tmp_label in enumerate(tmp_label_lst):
            if tmp_wordpiece_mask[tmp_idx] != -100:
                tmp_label_seq.append(idx2label_map[tmp_label])
            
        output_label_sequence.append(tmp_label_seq)
        
    return output_label_sequence