from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score


"""
	Utility functions for evaluating the model performance
"""


def eval_deep(log, loader):
    """
    Evaluating the classification performance given mini-batch data (Corrected version)
    """
    prob_log, label_log = [], []
    pred_log = [] # Thêm mảng để lưu toàn bộ dự đoán

    for batch in log:
        pred_y = batch[0].data.cpu().numpy().argmax(axis=1)
        y = batch[1].data.cpu().numpy().tolist()
        
        prob_log.extend(batch[0].data.cpu().numpy()[:, 1].tolist())
        label_log.extend(y)
        pred_log.extend(pred_y.tolist()) # Gom tất cả dự đoán lại

    # Tính toán 1 lần duy nhất trên TOÀN BỘ dữ liệu để có kết quả chính xác
    accuracy = accuracy_score(label_log, pred_log)
    f1_binary = f1_score(label_log, pred_log)
    f1_micro = f1_score(label_log, pred_log, average='micro')
    precision = precision_score(label_log, pred_log, zero_division=0)
    recall = recall_score(label_log, pred_log, zero_division=0)
    
    auc = roc_auc_score(label_log, prob_log)
    ap = average_precision_score(label_log, prob_log)

    # Trả về đầy đủ các giá trị theo đúng thứ tự logic của bạn
    return accuracy, f1_binary, precision, recall, auc, ap


