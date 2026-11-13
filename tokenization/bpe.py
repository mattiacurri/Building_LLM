# Byte Pair Encoding (BPE)
# https://neetcode.io/problems/tokenizer-bpe
import heapq
from collections import defaultdict
from typing import List

class BPE:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = []
        corpus = list(corpus)
        i = 0
        while i < num_merges:
            count = defaultdict(int)
            j = 0
            while j < len(corpus)-1:
                # a. Count frequency of all adjacent token pairs
                count[(corpus[j], corpus[j+1])] -= 1
                j += 1
            # b. Find the most frequent pair (break ties lexicographically)
            heap = [(valore, chiave) for chiave, valore in count.items()]
            heapq.heapify(heap)  # [(2, 'b'), (5, 'a'), (8, 'c')]
            tokens.append(list(heap[0][1]))

            '''
            # Find most frequent pair (tiebreak: lexicographically smallest)
            best_count = max(count.values())
            candidates = sorted(p for p, c in count.items() if c == best_count)
            best = candidates[0]

            merges.append([best[0], best[1]])
            '''

            # c. Merge all non-overlapping occurrences left to right
            j = 0
            while j < len(corpus)-1:
                if corpus[j] == tokens[-1][0] and corpus[j+1] == tokens[-1][1]:
                    corpus[j] = str(corpus[j])+str(corpus[j+1])
                    # $O(M \cdot N^2)$, with another list $O(M \cdot N)$
                    corpus.pop(j+1)
                j += 1
            i += 1
        return tokens


def test_zero_merges():
    assert BPE().get_merges("abc", 0) == []


def test_most_frequent_pair():
    assert BPE().get_merges("aaab", 1) == [["a", "a"]]


def test_tie_broken_lexicographically():
    assert BPE().get_merges("bab", 1) == [["a", "b"]]


def test_sequential_merges():
    assert BPE().get_merges("aaaa", 2) == [["a", "a"], ["aa", "aa"]]


if __name__ == "__main__":
    test_zero_merges()
    test_most_frequent_pair()
    test_tie_broken_lexicographically()
    test_sequential_merges()
    print("ok")
