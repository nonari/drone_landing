import torch
from dataloader import tugraz_classnames
from aeroscapes import aeroscapes_classnames
from os import path


class Tabulator:
    def __init__(self, header, rownames):
        self.content = ''
        self.rownames = list(rownames)
        self.space = len(sorted(header + rownames, key=lambda x: len(x))[-1])+2
        for w in header:
            self.add_word(w)
        self.content += '\n'
        self.content += ('-' * len(header) * self.space) + '\n'

    def add_line(self, line):
        self.add_word(self.rownames.pop(0))
        for w in line:
            self.add_word(w)
        self.content += '\n'

    def add_word(self, word):
        if type(word) is not str:
            word = f'{word:.2f}'
        self.content += word.ljust(self.space)


def write_table(metrics, location, classnames):
    tabulator = Tabulator(['', 'jaccard', 'f1', 'pre'], classnames)
    for j, f, p in zip(metrics['jcc'], metrics['f1'], metrics['pre']):
        tabulator.add_line([j, f, p])
    print(tabulator.content)
    text_file = open(location, "w")
    text_file.write(tabulator.content)
    text_file.close()


if __name__ == '__main__':
    loc = './executions/PSPNet_r32/test_results_alt/'
    d = torch.load(path.join(loc, 'metrics_summary'), map_location='cpu')
    write_table(d, loc, tugraz_classnames)
