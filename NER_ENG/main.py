from data_process import Data

if __name__ == "__main__":
    data = Data("train.txt")
    train_word_lists, tag_lists = data.transform()
    tag_lists = data.statistics(tag_lists)
    X_train, X_valid, y_train, y_valid = data.train_valid_split(train_word_lists, tag_lists)
    print(y_valid[:10])
