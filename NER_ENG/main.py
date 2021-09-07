from data_process import Data

if __name__ == "__main__":
    data = Data("train.txt")
    train_word_lists, tag_lists = data.transform()
    tag_lists = data.statistics(tag_lists)
    print(tag_lists[:20])
    print(train_word_lists[:20])