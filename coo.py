def co_acc_probs(user_prob, prob1, prob2):
    '''共同出现的问题'''
    count = 0
    agg = 0
    for user, prob in user_prob.index:
        if prob == prob1:
            if prob2 in user_prob[user].index:
                count += 1
                agg += user_prob[user][prob1] * user_prob[user][prob2]
#                 print('user {} answered two questions, with {} answered {}, and {} answered {}'
#                   .format(user, prob1, user_prob[user][prob1], prob2, user_prob[user][prob2]))
    if count == 0:
        return -1
    else:
        return agg/count

def co_acc_skills(user_skill, skill1, skill2):
    '''共同出现的知识点'''
    count = 0
    agg = 0
    for user, skill in user_skill.index:
        if skill1 == skill:
            if skill2 in user_skill[user].index:
                count += 1
                agg += user_skill[user][skill1] * user_skill[user][skill2]
    if count == 0:
        return -1
    else:
        return agg/count