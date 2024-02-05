from mydata import generate_train_test_indices, MyDataset

if __name__ == '__main__':
    full_dataset = MyDataset(root="full_data/volga/", dataset="volga")
    full_data = full_dataset[0]
    print(full_data.validate(raise_on_error=False))
    generate_train_test_indices(y=full_data.y)
