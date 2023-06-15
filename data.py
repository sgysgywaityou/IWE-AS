from torch.utils.data import Dataset, DataLoader
from utils.get_new_vectors import preprocess_text
from model.classifier import Model


class TextDataset(Dataset):
    '''
    file_list:  e.g. [./data/text/0_0.txt, ...]
    Return the setence.
    '''
    def __init__(self, file_list, stopwords) -> None:
        super().__init__()
        self.files = file_list
        self.stopwords = stopwords


    def __getitem__(self, index):
        file = self.files[index][0] # ./data/text/0_0.txt
        print('loading file: ', file)
        # get label
        label = int(self.files[index][1])
        # get text content, and transfer it to tensor with unified shape
        with open(file, 'r', encoding='utf-8') as f:
            txt = f.readlines()
        txt = [''.join(txt)]
        sentence = []
        preprocess_text(content_lines=txt, sentences=sentence, stopwords=self.stopwords)
        return sentence[0], label # sentence为二维列表，label为标签


    def __len__(self):
        return len(self.files)