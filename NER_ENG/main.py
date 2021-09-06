from data_process import Data

if __name__ == "__main__":
    data = Data("train.txt")
    train_word_lists, tag_lists = data.transform()
    print(train_word_lists[:5])
    print(tag_lists[:5])