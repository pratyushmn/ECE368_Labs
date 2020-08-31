import os.path
import numpy as np
import matplotlib.pyplot as plt
import util

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    ### TODO: Write your code here


    spam_list = file_lists_by_category[0]
    ham_list = file_lists_by_category[1]

    spam_counts = util.get_counts(spam_list)
    num_spam_words = len(spam_counts)

    ham_counts = util.get_counts(ham_list)
    num_ham_words = len(ham_counts)
    
    D = len(spam_counts.keys() & ham_counts.keys())

    p_d = dict()
    q_d = dict()

    for word in spam_counts:
        p_d[word] = (spam_counts[word] + 1)/(num_spam_words + D)

    p_d["default val"] = 1/(num_spam_words + D)
    
    for word in ham_counts:
        q_d[word] = (ham_counts[word] + 1)/(num_ham_words+ D)

    q_d["default val"] = 1/(num_ham_words + D)

    probabilities_by_category = (p_d, q_d)

    return probabilities_by_category

def classify_new_email(filename,probabilities_by_category,prior_by_category):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here


    file_list = []
    file_list.append(filename)

    P_spam = np.math.log(prior_by_category[0])
    P_ham = np.math.log(prior_by_category[1])

    p_d = probabilities_by_category[0]
    q_d = probabilities_by_category[1]

    X_n = util.get_word_freq(file_list)

    spam_predictor = P_spam
    ham_predictor = P_ham

    for word in X_n:
        spam_predictor += X_n[word]*np.math.log(p_d.get(word, p_d["default val"]))
        ham_predictor += X_n[word]*np.math.log(q_d.get(word, q_d["default val"]))

    if spam_predictor > ham_predictor:
        result = "spam"
    else:
        result = "ham"

    classify_result = (result, [spam_predictor, ham_predictor])
    
    return classify_result

def classify_new_email_mod(filename, probabilities_by_category, prior_by_category, zeta):
    file_list = []
    file_list.append(filename)

    p_d = probabilities_by_category[0]
    q_d = probabilities_by_category[1]

    X_n = util.get_word_freq(file_list)

    spam_predictor = 0
    ham_predictor = 0

    for word in X_n:
        spam_predictor += X_n[word]*np.math.log(p_d.get(word, p_d["default val"]))
        ham_predictor += X_n[word]*np.math.log(q_d.get(word, q_d["default val"]))

    if spam_predictor - ham_predictor > np.math.log(zeta):
        result = "spam"
    else:
        result = "ham"

    return result

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions 

    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve

    zeta_vals = [10 ** (-150), 10 ** (-10), 10 ** (-9), 10 ** (-8), 10 ** (-7), 10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3), 10 ** (-2), 10 ** (-1), 10 ** 0, 10 ** 1, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8, 10 ** 9, 10 ** 10, 10 ** 20]

    type1 = []
    type2 = []

    for zeta in zeta_vals:   
        performance_measures = np.zeros([2,2])
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label = classify_new_email_mod(filename,
                                                    probabilities_by_category,
                                                    priors_by_category, zeta)

            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base) 
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1
        
        # Correct counts are on the diagonal
        totals = np.sum(performance_measures, 1)
        correct = np.diag(performance_measures)
        type1.append(totals[0] - correct[0])
        type2.append(totals[1] - correct[1])
        
    plt.scatter(type1, type2)
    plt.xlabel("Number of Type 1 Errors")
    plt.ylabel("Number of Type 2 Errors")
    plt.title("Variation of Type 1 and Type 2 Errors as the Parameter Zeta Changes")
    plt.show()
