### Examples: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
score = sentence_bleu(reference, candidate)
print(score)


# one word different
from nltk.translate.bleu_score import sentence_bleu

reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
score = sentence_bleu(reference, candidate)
print(score)


# two words different
from nltk.translate.bleu_score import sentence_bleu

reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'sleepy', 'dog']
score = sentence_bleu(reference, candidate)
print(score)


samples = 135091
mod_20 = []
mod_20_smaller_3 = []
rest_mod_20 = []
for i, num in enumerate(range(samples)):
    if i % 20 == 0:
        mod_20.append(num)
    elif i % 20 < 3:
        mod_20_smaller_3.append(num)
    else:
        rest_mod_20.append(num)

print(len(mod_20)) # val - 6755
print(len(mod_20_smaller_3)) # test - 13510
print(len(rest_mod_20)) # train - 114826