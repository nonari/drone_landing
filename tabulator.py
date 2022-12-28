import torch
from dataloader import tugraz_classnames
from aeroscapes import aeroscapes_classnames


class Tabulator:
    def __ini__(self, header, rownames):
        self.content = ''
        self.rownames = list(rownames)
        self.space = len(sorted(header, key=lambda x: len(x))[-1])+2
        self.add_line(header)
        self.content += '-' * len(header) * self.space

    def add_line(self, line):
        self.add_word(self.rownames.pop())
        for w in line:
            self.add_word(w)
        self.content += '\n'

    def add_word(self, word):
        if type(word) is float:
            word = f'{word:.2f}'
        self.content += word.ljust(self.space)


if __name__ == '__main__':
    d = torch.load('./executions/UNet_r32/test_results/metrics_summary', map_location='cpu')
    pass