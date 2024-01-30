from torch.utils.data import Dataset
import numpy as np 


class KTDataset(Dataset):
    def __init__(self, group, min_samples=3, max_seq=50):
        super(KTDataset, self).__init__()
        self.max_seq = max_seq
        self.samples = {}
        
        self.user_ids = []
        for user_id in group.index:
            q, qa = group[user_id]
            if len(q) < min_samples:
                continue
            
            # Main Contribution
            if len(q) > self.max_seq:
                total_questions = len(q)
                initial = total_questions % self.max_seq
                if initial >= min_samples:
                    self.user_ids.append(f"{user_id}_0")
                    self.samples[f"{user_id}_0"] = (q[:initial], qa[:initial])
                for seq in range(total_questions // self.max_seq):
                    self.user_ids.append(f"{user_id}_{seq+1}")
                    start = initial + seq * self.max_seq
                    end = start + self.max_seq
                    # qa_ = qa.copy()
                    # random.shuffle(qa_)
                    if len(qa) != 0:
                        self.samples[f"{user_id}_{seq+1}"] = (q[start:end], 
                                                          qa[start:end])
            else:
                user_id = str(user_id)
                self.user_ids.append(user_id)
                self.samples[user_id] = (q, qa)
    
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_ = self.samples[user_id]
        seq_len = len(q_)
        if len(q_) <= 1:
            q_ = np.array(q_)
        q = np.zeros((self.max_seq, q_.shape[1]))
        qa = np.zeros(self.max_seq, dtype=int)
        if seq_len == self.max_seq:
            q[:] = q_
            qa[:] = qa_
        else:
            q[-seq_len:] = q_
            qa[-seq_len:] = qa_
        
        x_emb = self.onehot(q[:-1], qa[:-1])
        q_next = q[1:]
        labels = qa[1:]
        
        return x_emb, q_next, labels

    
    def onehot(self, questions, answers):
        emb_num = questions.shape[-1]
        result = np.zeros(shape=[self.max_seq-1, 2*emb_num])
        for i in range(self.max_seq-1):
            if answers[i] > 0:
                result[i][:emb_num] = questions[i]
            elif answers[i] == 0:
                result[i][emb_num:] = questions[i]
        return result