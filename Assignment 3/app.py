import pandas as pd

# Load the dataset
df = pd.read_csv('Assignment 3/SpamDetection.csv')

# Split into training and testing sets
training_data = df[:20]
testing_data = df[20:]

# Calculate prior probabilities (P(spam) and P(ham)) based on the training data
spam_count = len(training_data[training_data['Target'] == 'spam'])
ham_count = len(training_data[training_data['Target'] == 'ham'])

total_train_messages = len(training_data)

P_spam = spam_count / total_train_messages
P_ham = ham_count / total_train_messages

# Create word count dictionaries for spam and ham classes based on the training data
spam_words = ' '.join(training_data[training_data['Target'] == 'spam']['data']).split()
ham_words = ' '.join(training_data[training_data['Target'] == 'ham']['data']).split()

# Create dictionaries to store word counts
spam_word_count = {word: spam_words.count(word) for word in set(spam_words)}
ham_word_count = {word: ham_words.count(word) for word in set(ham_words)}

# Vocabulary - all unique words across both classes
vocabulary = set(spam_words + ham_words)

# Calculate total word counts in spam and ham classes
spam_total = sum(spam_word_count.values())
ham_total = sum(ham_word_count.values())

# Laplace smoothing: adding 1 to the word count and the size of the vocabulary to the denominator
def perform_laplace_smoothing(word, word_count, total_count):
    return (word_count.get(word, 0) + 1) / (total_count + len(vocabulary))

# Function to classify a sentence
def classify_sentence(sentence):
    test_words = sentence.lower().split()
    
    P_test_given_spam = P_spam
    P_test_given_ham = P_ham

    for word in test_words:
        P_test_given_spam *= perform_laplace_smoothing(word, spam_word_count, spam_total)
        P_test_given_ham *= perform_laplace_smoothing(word, ham_word_count, ham_total)

    if P_test_given_spam > P_test_given_ham:
        return 'spam'
    else:
        return 'ham'

# Iterate over testing_data and predict the class
correct_predictions = 0
for index, row in testing_data.iterrows():
    sentence = row['data']
    actual_classification = row['Target']
    predicted_classification = classify_sentence(sentence)
    
    if predicted_classification == actual_classification:
        correct_predictions += 1

    print(f"Sentence: {sentence}")
    print(f"Actual Classification: {actual_classification}")
    print(f"Predicted Classification: {predicted_classification}")
    print("-" * 50)

# Calculate accuracy
accuracy = correct_predictions / len(testing_data)
print(f"Accuracy: {accuracy:.2f}")
