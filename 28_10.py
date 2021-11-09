from typing import List, Dict
import math


class CountVectorizer:

    def create_dict_of_words(self, corpus: List[str]) -> Dict[str, List[int]]:
        dict_of_words = {}
        corpus_new = []
        for index, string in enumerate(corpus):
            string_new = string.lower()
            corpus_new.append(string_new.split())

        for index, word_list in enumerate(corpus_new):
            for word in word_list:
                dict_of_words.setdefault(word, [0] * len(corpus)) # список из нулей
                dict_of_words[word][index] += 1

        # for index, string in enumerate(corpus):
        #     string = string + ' '  # чтобы в конце всегда был пробел
        #     new_string = ''
        #     for symbol in string:
        #         if symbol == ' ':
        #             if new_string != '':
        #                 dict_of_words.setdefault(new_string, [0] * len(corpus))
        #                 dict_of_words[new_string][index] += 1
        #                 new_string = ''
        #         else:
        #             new_string += symbol.lower()
        return dict_of_words

    def get_feature_names(self, corpus: List[str]) -> List[str]:
        return list(self.create_dict_of_words(corpus))

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        arrays = []
        for _ in range(len(corpus)):
            arrays.append([])
        for word, cnt_list in self.create_dict_of_words(corpus).items():
            for index, cnt in enumerate(cnt_list):
                arrays[index].append(cnt)
        return arrays


class TfIdfTransformer:
    # def __init__(self):
    #     self._tf = []
    #     self._idf = []

    def tf_transform(self, matrix: List[List[int]]) -> List[List[float]]:
        answer_tf: list[list[float]] = []
        for index, row in enumerate(matrix):
            answer_tf.append([])
            total_cnt = sum(row)
            for element in row:
                answer_tf[index].append(round(element / total_cnt, 5))
        return answer_tf

    def idf_transform(self, matrix: List[List[int]]) -> List[float]:
        answer_idf = []
        denominator = [1] * len(matrix[0])
        numerator = len(matrix) + 1
        for row in matrix:
            for index, value in enumerate(row):
                if value > 0:
                    denominator[index] += 1
        for el in denominator:
            answer_idf.append(round(math.log(numerator / el) + 1, 5))
        return answer_idf

    def fit_transform(self, matrix: List[List[int]]) -> List[List[float]]:
        tf = self.tf_transform(matrix)
        idf = self.idf_transform(matrix)
        answer_tf_idf = []
        for ind_tf, row in enumerate(tf):
            answer_tf_idf.append([])
            for ind_idf, el in enumerate(row):
                answer_tf_idf[ind_tf].append(round(el*idf[ind_idf], 3))
        return answer_tf_idf


class TfIdfVectorizer(CountVectorizer):
    def __init__(self):
        super().__init__()
        self._tf_idf_transformer = TfIdfTransformer()

    def fit_transform(self, corpus):
        matrix = super().fit_transform(corpus)
        tf_idf_matrix = self._tf_idf_transformer.fit_transform(matrix)
        return tf_idf_matrix


if __name__ == '__main__':
    text = [' Crock Pot    Pasta Never boil pasta again ',
            'Pasta Pomodoro Fresh   ingredients Parmesan to taste']
    # text = ['This is the first document', 'This document is the second document',
    #         'And this is the third one', 'Is  IS this the first document']
    vector = CountVectorizer()
    print(vector.get_feature_names(text))
    print(vector.fit_transform(text))
    matrix = vector.fit_transform(text)

    vec = TfIdfTransformer()
    # print(vec.tf_transform(matrix))
    # print(vec.idf_transform(matrix))
    print(vec.fit_transform(matrix))

    transformer = TfIdfVectorizer()
    print(transformer.fit_transform(text))
    print(transformer.get_feature_names(text))